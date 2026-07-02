"""FastAPI gateway: one pure-OpenAI front door over routed backend pools."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from .embeddings import Embedder, build_embedders
from .router import Router, UnknownModel
from .spec import EmbeddingsNode, Spec, SttNode, TtsNode, load_spec, SpecError
from .stt import Transcriber, build_transcribers
from .tts import Synthesizer, build_synthesizers
from .supervisor import Supervisor

_UI_FILE = Path(__file__).parent / "static" / "ui.html"


def _default_embedder_factory(node: EmbeddingsNode) -> Embedder:
    from .embeddings import FastEmbedEmbedder
    return FastEmbedEmbedder(node)


def _default_transcriber_factory(node: SttNode) -> Transcriber:
    from .stt import FasterWhisperTranscriber
    return FasterWhisperTranscriber(node)


def _default_synthesizer_factory(node: TtsNode) -> Synthesizer:
    from .tts import KokoroSynthesizer
    return KokoroSynthesizer(node)


def create_app(spec: Spec, supervisor: Supervisor,
               client: httpx.AsyncClient | None = None,
               embedder_factory=None, transcriber_factory=None,
               synthesizer_factory=None,
               spec_raw: dict | None = None, spec_path: str | None = None) -> FastAPI:
    client = client or httpx.AsyncClient(timeout=None)
    embedder_factory = embedder_factory or _default_embedder_factory
    transcriber_factory = transcriber_factory or _default_transcriber_factory
    synthesizer_factory = synthesizer_factory or _default_synthesizer_factory

    def _start(new_spec: Spec, new_raw: dict | None) -> None:
        pools = supervisor.start(new_spec)
        app.state.router = Router(pools)
        app.state.embedders = build_embedders(new_spec, embedder_factory)
        app.state.transcribers = build_transcribers(new_spec, transcriber_factory)
        app.state.synthesizers = build_synthesizers(new_spec, synthesizer_factory)
        app.state.spec = new_spec
        app.state.spec_raw = new_raw

    def _apply(new_spec: Spec, new_raw: dict) -> None:
        supervisor.stop()
        _start(new_spec, new_raw)
        if spec_path:
            Path(spec_path).write_text(json.dumps(new_raw, indent=2))

    def _status() -> dict:
        return {"llm": sorted(app.state.router.models()),
                "embeddings": sorted(app.state.embedders),
                "stt": sorted(app.state.transcribers),
                "tts": sorted(app.state.synthesizers)}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _start(spec, spec_raw)
        yield
        supervisor.stop()
        await client.aclose()

    app = FastAPI(title="ainbox-infrastructure gateway", lifespan=lifespan)

    def _router() -> Router:
        return app.state.router

    async def _proxy(request: Request, path: str) -> Response:
        body = await request.body()
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        model = payload.get("model")
        if not model:
            return JSONResponse({"error": "missing 'model'"}, status_code=400)
        try:
            backend = _router().resolve(model)
        except UnknownModel:
            return JSONResponse(
                {"error": f"model '{model}' is not raised"}, status_code=404)

        upstream = client.build_request(
            "POST", f"{backend.base_url}{path}", content=body,
            headers={"content-type": "application/json"})
        resp = await client.send(upstream, stream=True)
        return StreamingResponse(
            resp.aiter_raw(),
            status_code=resp.status_code,
            headers={"content-type": resp.headers.get("content-type", "application/json")},
            background=BackgroundTask(resp.aclose),
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await _proxy(request, "/v1/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await _proxy(request, "/v1/completions")

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> Response:
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        model = payload.get("model")
        if not model:
            return JSONResponse({"error": "missing 'model'"}, status_code=400)
        embedder = app.state.embedders.get(model)
        if embedder is None:
            return JSONResponse(
                {"error": f"embedding model '{model}' is not raised"}, status_code=404)
        raw = payload.get("input")
        texts = [raw] if isinstance(raw, str) else list(raw or [])
        vectors = await asyncio.to_thread(embedder.embed, texts)
        data = [{"object": "embedding", "index": i, "embedding": v}
                for i, v in enumerate(vectors)]
        return JSONResponse({"object": "list", "data": data, "model": model,
                             "usage": {"prompt_tokens": 0, "total_tokens": 0}})

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(file: UploadFile = File(...),
                             model: str = Form(...),
                             language: str | None = Form(None)) -> Response:
        transcriber = app.state.transcribers.get(model)
        if transcriber is None:
            return JSONResponse(
                {"error": f"stt model '{model}' is not raised"}, status_code=404)
        audio = await file.read()
        text = await asyncio.to_thread(transcriber.transcribe, audio, language)
        return JSONResponse({"text": text})

    @app.post("/v1/audio/speech")
    async def speech(request: Request) -> Response:
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        model = payload.get("model")
        if not model:
            return JSONResponse({"error": "missing 'model'"}, status_code=400)
        text = payload.get("input")
        if not text:
            return JSONResponse({"error": "missing 'input'"}, status_code=400)
        synth = app.state.synthesizers.get(model)
        if synth is None:
            return JSONResponse(
                {"error": f"tts model '{model}' is not raised"}, status_code=404)
        audio = await asyncio.to_thread(synth.synthesize, text, payload.get("voice"))
        return Response(content=audio, media_type="audio/wav")

    @app.get("/v1/models")
    async def list_models() -> Response:
        slugs = sorted(set(_router().models())
                       | set(app.state.embedders)
                       | set(app.state.transcribers)
                       | set(app.state.synthesizers))
        data = [{"id": s, "object": "model", "owned_by": "ainbox"} for s in slugs]
        return JSONResponse({"object": "list", "data": data})

    @app.get("/")
    async def ui() -> Response:
        return FileResponse(_UI_FILE)

    @app.get("/api/spec")
    async def get_spec() -> Response:
        return JSONResponse(app.state.spec_raw or {})

    @app.post("/api/spec")
    async def set_spec(request: Request) -> Response:
        try:
            raw = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        try:
            new_spec = load_spec(raw)  # validate BEFORE touching the running set
        except SpecError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        await asyncio.to_thread(_apply, new_spec, raw)
        return JSONResponse({"ok": True, "status": _status()})

    @app.get("/api/status")
    async def status() -> Response:
        return JSONResponse(_status())

    app.state.client = client
    return app
