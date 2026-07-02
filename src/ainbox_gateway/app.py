"""FastAPI gateway: one pure-OpenAI front door over routed backend pools."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from .embeddings import Embedder, build_embedders
from .router import Router, UnknownModel
from .spec import EmbeddingsNode, Spec
from .supervisor import Supervisor


def _default_embedder_factory(node: EmbeddingsNode) -> Embedder:
    from .embeddings import FastEmbedEmbedder
    return FastEmbedEmbedder(node)


def create_app(spec: Spec, supervisor: Supervisor,
               client: httpx.AsyncClient | None = None,
               embedder_factory=None) -> FastAPI:
    client = client or httpx.AsyncClient(timeout=None)
    embedder_factory = embedder_factory or _default_embedder_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pools = supervisor.start(spec)
        app.state.router = Router(pools)
        app.state.embedders = build_embedders(spec, embedder_factory)
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

    @app.get("/v1/models")
    async def list_models() -> Response:
        slugs = sorted(set(_router().models()) | set(app.state.embedders))
        data = [{"id": s, "object": "model", "owned_by": "ainbox"} for s in slugs]
        return JSONResponse({"object": "list", "data": data})

    app.state.client = client
    return app
