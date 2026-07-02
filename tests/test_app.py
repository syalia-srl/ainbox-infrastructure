import json
from contextlib import asynccontextmanager

import httpx
import pytest
import respx
from asgi_lifespan import LifespanManager
from ainbox_gateway.app import create_app
from ainbox_gateway.spec import Spec, LlmNode, EmbeddingsNode, SttNode, TtsNode
from ainbox_gateway.supervisor import build_pools


class _FakeEmbedder:
    def __init__(self, node):
        self.slug = node.slug

    def embed(self, texts):
        return [[float(len(t)), 0.5] for t in texts]


def _app_with_embeddings():
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a", replicas=2)],
                embeddings=[EmbeddingsNode(slug="emb", model="MiniLM")])
    return create_app(spec, FakeSupervisor(), embedder_factory=_FakeEmbedder)


class FakeSupervisor:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self, spec):
        self.started = True
        return build_pools(spec, base=9000)

    def stop(self):
        self.stopped = True


def _app():
    # slug "a", 2 replicas -> backends at 127.0.0.1:9000 and :9001
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a", replicas=2)])
    return create_app(spec, FakeSupervisor())


@asynccontextmanager
async def _client(app):
    """Run the app's lifespan (populates app.state.router) around the client.

    httpx.ASGITransport does not emit ASGI lifespan events on its own, so the
    startup hook must be driven explicitly via LifespanManager.
    """
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
            yield c


@pytest.mark.asyncio
@respx.mock
async def test_chat_completion_routes_and_round_robins():
    route0 = respx.post("http://127.0.0.1:9000/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"id": "0"}))
    route1 = respx.post("http://127.0.0.1:9001/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"id": "1"}))
    async with _client(_app()) as c:
        r0 = await c.post("/v1/chat/completions", json={"model": "a", "messages": []})
        r1 = await c.post("/v1/chat/completions", json={"model": "a", "messages": []})
    assert r0.json()["id"] == "0" and r1.json()["id"] == "1"
    assert route0.called and route1.called


@pytest.mark.asyncio
@respx.mock
async def test_unknown_model_returns_404():
    async with _client(_app()) as c:
        r = await c.post("/v1/chat/completions", json={"model": "nope", "messages": []})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_missing_model_returns_400():
    async with _client(_app()) as c:
        r = await c.post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_models_endpoint_lists_slugs():
    async with _client(_app()) as c:
        r = await c.get("/v1/models")
    body = r.json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["a"]
    assert body["data"][0]["object"] == "model"


@pytest.mark.asyncio
async def test_embeddings_list_input():
    async with _client(_app_with_embeddings()) as c:
        r = await c.post("/v1/embeddings", json={"model": "emb", "input": ["ab", "xyz"]})
    body = r.json()
    assert body["object"] == "list" and body["model"] == "emb"
    assert [d["embedding"] for d in body["data"]] == [[2.0, 0.5], [3.0, 0.5]]
    assert [d["index"] for d in body["data"]] == [0, 1]


@pytest.mark.asyncio
async def test_embeddings_string_input_normalized():
    async with _client(_app_with_embeddings()) as c:
        r = await c.post("/v1/embeddings", json={"model": "emb", "input": "hello"})
    data = r.json()["data"]
    assert len(data) == 1 and data[0]["embedding"] == [5.0, 0.5]


@pytest.mark.asyncio
async def test_embeddings_unknown_model_404():
    async with _client(_app_with_embeddings()) as c:
        r = await c.post("/v1/embeddings", json={"model": "nope", "input": "x"})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_models_includes_embeddings():
    async with _client(_app_with_embeddings()) as c:
        r = await c.get("/v1/models")
    assert [m["id"] for m in r.json()["data"]] == ["a", "emb"]


class _FakeTranscriber:
    def __init__(self, node):
        self.slug = node.slug

    def transcribe(self, audio, language=None):
        return f"heard {len(audio)} bytes"


def _app_with_stt():
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a", replicas=2)],
                embeddings=[EmbeddingsNode(slug="emb", model="MiniLM")],
                stt=[SttNode(slug="whisper-small", model="small")])
    return create_app(spec, FakeSupervisor(),
                      embedder_factory=_FakeEmbedder,
                      transcriber_factory=_FakeTranscriber)


@pytest.mark.asyncio
async def test_transcription_returns_text():
    async with _client(_app_with_stt()) as c:
        r = await c.post("/v1/audio/transcriptions",
                         files={"file": ("a.wav", b"1234", "audio/wav")},
                         data={"model": "whisper-small"})
    assert r.json() == {"text": "heard 4 bytes"}


@pytest.mark.asyncio
async def test_transcription_unknown_model_404():
    async with _client(_app_with_stt()) as c:
        r = await c.post("/v1/audio/transcriptions",
                         files={"file": ("a.wav", b"x", "audio/wav")},
                         data={"model": "nope"})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_models_union_includes_stt():
    async with _client(_app_with_stt()) as c:
        r = await c.get("/v1/models")
    assert [m["id"] for m in r.json()["data"]] == ["a", "emb", "whisper-small"]


class _FakeSynth:
    def __init__(self, node):
        self.slug = node.slug

    def synthesize(self, text, voice=None):
        return f"WAV[{text}|{voice}]".encode()


def _app_with_tts():
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a", replicas=2)],
                tts=[TtsNode(slug="voice", model="kokoro")])
    return create_app(spec, FakeSupervisor(), embedder_factory=_FakeEmbedder,
                      transcriber_factory=_FakeTranscriber,
                      synthesizer_factory=_FakeSynth)


@pytest.mark.asyncio
async def test_speech_returns_wav_bytes():
    async with _client(_app_with_tts()) as c:
        r = await c.post("/v1/audio/speech",
                         json={"model": "voice", "input": "hola", "voice": "ef_dora"})
    assert r.status_code == 200
    assert "audio/wav" in r.headers["content-type"]
    assert r.content == b"WAV[hola|ef_dora]"


@pytest.mark.asyncio
async def test_speech_unknown_model_404():
    async with _client(_app_with_tts()) as c:
        r = await c.post("/v1/audio/speech", json={"model": "nope", "input": "x"})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_speech_missing_input_400():
    async with _client(_app_with_tts()) as c:
        r = await c.post("/v1/audio/speech", json={"model": "voice"})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_models_union_includes_tts():
    async with _client(_app_with_tts()) as c:
        r = await c.get("/v1/models")
    assert [m["id"] for m in r.json()["data"]] == ["a", "voice"]


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_supervisor():
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a", replicas=2)])
    sup = FakeSupervisor()
    app = create_app(spec, sup)
    async with LifespanManager(app):
        assert sup.started
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
            r = await c.get("/v1/models")
            assert [m["id"] for m in r.json()["data"]] == ["a"]
    assert sup.stopped


@pytest.mark.asyncio
async def test_apply_new_spec_relaunches_registries(tmp_path):
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a", replicas=1)],
                embeddings=[EmbeddingsNode(slug="emb", model="M")])
    raw = {"gateway": {"port": 8080}, "llm": [{"slug": "a"}],
           "embeddings": [{"slug": "emb", "model": "M"}]}
    path = tmp_path / "spec.json"
    app = create_app(spec, FakeSupervisor(), embedder_factory=_FakeEmbedder,
                     transcriber_factory=_FakeTranscriber,
                     spec_raw=raw, spec_path=str(path))
    new_raw = {"gateway": {"port": 8080}, "llm": [{"slug": "b"}],
               "stt": [{"slug": "w", "model": "small"}]}
    async with _client(app) as c:
        r = await c.post("/api/spec", json=new_raw)
        assert r.status_code == 200
        st = (await c.get("/api/status")).json()
    assert st == {"llm": ["b"], "embeddings": [], "stt": ["w"], "tts": []}
    assert json.loads(path.read_text())["llm"][0]["slug"] == "b"


@pytest.mark.asyncio
async def test_apply_invalid_spec_400_and_keeps_running_set():
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a")])
    raw = {"gateway": {"port": 8080}, "llm": [{"slug": "a"}]}
    app = create_app(spec, FakeSupervisor(), embedder_factory=_FakeEmbedder,
                     transcriber_factory=_FakeTranscriber, spec_raw=raw)
    async with _client(app) as c:
        r = await c.post("/api/spec", json={"gateway": {"port": 8080}, "llm": []})
        assert r.status_code == 400
        st = (await c.get("/api/status")).json()
    assert st["llm"] == ["a"]  # unchanged


@pytest.mark.asyncio
async def test_root_serves_ui():
    app = create_app(Spec(gateway_port=8080, llm=[LlmNode(slug="a")]),
                     FakeSupervisor(), embedder_factory=_FakeEmbedder,
                     transcriber_factory=_FakeTranscriber,
                     spec_raw={"gateway": {"port": 8080}, "llm": [{"slug": "a"}]})
    async with _client(app) as c:
        r = await c.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "ainbox" in r.text.lower()


@pytest.mark.asyncio
async def test_get_spec_returns_raw():
    raw = {"gateway": {"port": 8080}, "llm": [{"slug": "a"}]}
    app = create_app(Spec(gateway_port=8080, llm=[LlmNode(slug="a")]),
                     FakeSupervisor(), embedder_factory=_FakeEmbedder,
                     transcriber_factory=_FakeTranscriber, spec_raw=raw)
    async with _client(app) as c:
        got = (await c.get("/api/spec")).json()
    assert got == raw
