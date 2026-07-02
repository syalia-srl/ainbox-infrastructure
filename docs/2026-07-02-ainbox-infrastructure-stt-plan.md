# ainbox-infrastructure STT Backend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add speech-to-text to the gateway — faster-whisper served at the standard `POST /v1/audio/transcriptions` (multipart, model in body), absorbing the standalone `whisper_api.py`.

**Architecture:** Mirrors embeddings. Whisper runs **in-process** in the gateway (faster-whisper is a library). A `Transcriber` protocol abstracts it; the real `FasterWhisperTranscriber` lazy-imports faster-whisper and is injected via a factory, so unit tests use fakes and the package still imports without faster-whisper. The raise-spec gains an optional `stt` block; the recipe already bakes whisper models.

**Tech Stack:** Same gateway package. `python-multipart` (needed to parse the multipart request) in main deps; `faster-whisper` in the `[gpu]` extra.

## Global Constraints

- **OpenAI shape:** `POST /v1/audio/transcriptions` multipart form — `file` (audio) + `model` (slug) + optional `language`; response `{"text": "..."}`. Replaces the old non-standard `/v1/audio/transcriptions/{model_name}` path route.
- **In-process, single model per slug** — no subprocess, no round-robin.
- **Lazy faster-whisper import** — the gateway imports and unit-tests without it.
- **Cut clean:** delete `whisper_api.py`; the gateway owns STT.
- **Commit style:** conventional commits; `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## File Structure

- `src/ainbox_gateway/spec.py` — add `SttNode`, parse optional `stt`.
- `src/ainbox_gateway/stt.py` — `Transcriber` protocol, `build_transcribers`, `FasterWhisperTranscriber` (lazy).
- `src/ainbox_gateway/app.py` — build transcribers in lifespan, `POST /v1/audio/transcriptions`, `/v1/models` union includes STT.
- `pyproject.toml` — `python-multipart` (main), `faster-whisper` (`[gpu]`).
- `build/`, `build.sh` — delete `whisper_api.py`, drop it from the context check and Dockerfile COPY.
- Tests: `tests/test_spec.py`, `tests/test_stt.py`, `tests/test_app.py`.
- `docs/smoke-gateway.md` — STT section.

---

### Task 1: Raise-spec `stt` block

**Files:**
- Modify: `src/ainbox_gateway/spec.py`
- Test: `tests/test_spec.py` (add cases)

**Interfaces:**
- Produces:
  - `@dataclass SttNode(slug: str, model: str, device: str = "cuda", compute_type: str = "float16")`
  - `Spec` gains `stt: list[SttNode]` (default `[]`).
  - `load_spec` parses optional `stt`; a node missing `slug` or `model` raises `SpecError`.

- [ ] **Step 1: Write failing tests** (append to `tests/test_spec.py`; add `SttNode` to the import line)

```python
def test_stt_optional_defaults_empty():
    spec = load_spec({"gateway": {"port": 8080}, "llm": [{"slug": "a"}]})
    assert spec.stt == []


def test_stt_parsed():
    spec = load_spec({"gateway": {"port": 8080}, "llm": [{"slug": "a"}],
                      "stt": [{"slug": "w", "model": "small"}]})
    assert spec.stt == [SttNode(slug="w", model="small", device="cuda",
                                compute_type="float16")]


def test_stt_node_missing_model_raises():
    with pytest.raises(SpecError):
        load_spec({"gateway": {"port": 8080}, "llm": [{"slug": "a"}],
                   "stt": [{"slug": "w"}]})
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_spec.py -k stt -v`
Expected: FAIL — `ImportError: cannot import name 'SttNode'`.

- [ ] **Step 3: Implement** — in `spec.py`, add after `EmbeddingsNode`:

```python
@dataclass
class SttNode:
    slug: str
    model: str
    device: str = "cuda"
    compute_type: str = "float16"
```

Add to `Spec`:

```python
    stt: list[SttNode] = field(default_factory=list)
```

Add loader + wire into `load_spec`:

```python
def _load_stt(raw: dict) -> SttNode:
    if "slug" not in raw or "model" not in raw:
        raise SpecError("stt node needs 'slug' and 'model'")
    return SttNode(slug=raw["slug"], model=raw["model"],
                   device=raw.get("device", "cuda"),
                   compute_type=raw.get("compute_type", "float16"))
```

```python
        embeddings=[_load_embeddings(e) for e in data.get("embeddings", [])],
        stt=[_load_stt(s) for s in data.get("stt", [])],
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_spec.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ainbox_gateway/spec.py tests/test_spec.py
git commit -m "feat(gateway): raise-spec stt block

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Transcriber protocol + registry + lazy FasterWhisperTranscriber

**Files:**
- Create: `src/ainbox_gateway/stt.py`
- Test: `tests/test_stt.py`

**Interfaces:**
- Consumes: `SttNode`, `Spec`.
- Produces:
  - `class Transcriber(Protocol)` with `slug: str` and `transcribe(self, audio: bytes, language: str | None = None) -> str`.
  - `build_transcribers(spec: Spec, factory: Callable[[SttNode], Transcriber]) -> dict[str, Transcriber]`.
  - `class FasterWhisperTranscriber` — lazy-imports faster-whisper; `transcribe` writes bytes to a temp file, runs the model, joins segment text.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_stt.py
from ainbox_gateway.spec import Spec, LlmNode, SttNode
from ainbox_gateway.stt import build_transcribers


class FakeTranscriber:
    def __init__(self, node):
        self.slug = node.slug

    def transcribe(self, audio, language=None):
        return f"transcribed:{len(audio)}:{language}"


def test_build_transcribers_maps_slug():
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a")],
                stt=[SttNode(slug="w", model="small")])
    ts = build_transcribers(spec, factory=FakeTranscriber)
    assert set(ts) == {"w"}
    assert ts["w"].transcribe(b"1234", "es") == "transcribed:4:es"


def test_build_transcribers_empty_when_no_stt():
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a")])
    assert build_transcribers(spec, factory=FakeTranscriber) == {}


def test_module_imports_without_faster_whisper():
    import importlib
    import ainbox_gateway.stt as m
    importlib.reload(m)
    assert hasattr(m, "FasterWhisperTranscriber")
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_stt.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ainbox_gateway.stt'`.

- [ ] **Step 3: Implement**

```python
# src/ainbox_gateway/stt.py
"""In-process speech-to-text backends served at /v1/audio/transcriptions.

faster-whisper is imported lazily inside FasterWhisperTranscriber so this
module (and the whole gateway) imports and unit-tests without it installed.
"""
from __future__ import annotations

import os
import tempfile
from typing import Callable, Protocol

from .spec import SttNode, Spec


class Transcriber(Protocol):
    slug: str

    def transcribe(self, audio: bytes, language: str | None = None) -> str: ...


def build_transcribers(
    spec: Spec, factory: Callable[[SttNode], "Transcriber"]
) -> dict[str, "Transcriber"]:
    return {node.slug: factory(node) for node in spec.stt}


class FasterWhisperTranscriber:
    """Real transcriber over faster-whisper (CTranslate2; CUDA-capable)."""

    def __init__(self, node: SttNode):
        from faster_whisper import WhisperModel  # lazy

        self.slug = node.slug
        self._model = WhisperModel(
            node.model, device=node.device, compute_type=node.compute_type)

    def transcribe(self, audio: bytes, language: str | None = None) -> str:
        fd, path = tempfile.mkstemp(suffix=".audio")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(audio)
            segments, _ = self._model.transcribe(
                path, language=language, vad_filter=True)
            return " ".join(s.text for s in segments).strip()
        finally:
            if os.path.exists(path):
                os.remove(path)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_stt.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ainbox_gateway/stt.py tests/test_stt.py
git commit -m "feat(gateway): transcriber protocol, registry, lazy FasterWhisperTranscriber

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `/v1/audio/transcriptions` endpoint + lifespan + `/v1/models`

**Files:**
- Modify: `src/ainbox_gateway/app.py`
- Modify: `pyproject.toml` (add `python-multipart` to main deps)
- Test: `tests/test_app.py` (add cases)

**Interfaces:**
- Consumes: `build_transcribers`, `Transcriber`.
- Produces:
  - `create_app(..., transcriber_factory=None)` — lifespan builds `app.state.transcribers = build_transcribers(spec, transcriber_factory or _default_transcriber_factory)`.
  - `POST /v1/audio/transcriptions`: multipart `file` + `model` (+ optional `language`); unknown model → 404; returns `{"text": str}`. Inference in `asyncio.to_thread`.
  - `/v1/models` union spans LLM + embeddings + STT slugs.

- [ ] **Step 1: Add `python-multipart` and install it**

In `pyproject.toml` main deps add `"python-multipart>=0.0.9"`, then:

Run: `uv pip install --python .venv "python-multipart>=0.0.9"`

- [ ] **Step 2: Write failing tests** (append to `tests/test_app.py`; add `SttNode` to the spec import)

```python
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
```

- [ ] **Step 3: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_app.py -k "transcription or union" -v`
Expected: FAIL — `create_app()` has no `transcriber_factory` / 404 on the route.

- [ ] **Step 4: Implement** — in `app.py`:

Add imports:

```python
from fastapi import File, Form, UploadFile
from .stt import build_transcribers, Transcriber
from .spec import SttNode
```

Add a default factory (module level, beside `_default_embedder_factory`):

```python
def _default_transcriber_factory(node: SttNode) -> Transcriber:
    from .stt import FasterWhisperTranscriber
    return FasterWhisperTranscriber(node)
```

Extend the signature + lifespan:

```python
def create_app(spec: Spec, supervisor: Supervisor,
               client: httpx.AsyncClient | None = None,
               embedder_factory=None, transcriber_factory=None) -> FastAPI:
    client = client or httpx.AsyncClient(timeout=None)
    embedder_factory = embedder_factory or _default_embedder_factory
    transcriber_factory = transcriber_factory or _default_transcriber_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pools = supervisor.start(spec)
        app.state.router = Router(pools)
        app.state.embedders = build_embedders(spec, embedder_factory)
        app.state.transcribers = build_transcribers(spec, transcriber_factory)
        yield
        supervisor.stop()
        await client.aclose()
```

Add the endpoint (beside `/v1/embeddings`):

```python
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
```

Update `/v1/models` union to include transcribers:

```python
    @app.get("/v1/models")
    async def list_models() -> Response:
        slugs = sorted(set(_router().models())
                       | set(app.state.embedders)
                       | set(app.state.transcribers))
        data = [{"id": s, "object": "model", "owned_by": "ainbox"} for s in slugs]
        return JSONResponse({"object": "list", "data": data})
```

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS (all green; the existing `test_models_includes_embeddings` still passes — its app has no `stt`, so the union is `["a", "emb"]`).

- [ ] **Step 6: Commit**

```bash
git add src/ainbox_gateway/app.py pyproject.toml tests/test_app.py
git commit -m "feat(gateway): /v1/audio/transcriptions endpoint + models union

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Cut clean — delete whisper_api.py, wire deps + smoke

**Files:**
- Delete: `build/whisper_api.py`
- Modify: `build.sh` (drop `whisper_api.py` from `REQUIRED_FILES`)
- Modify: `build/Dockerfile` (drop `COPY whisper_api.py`; add `faster-whisper` to the gateway `[gpu]` install — already installed via the extra)
- Modify: `pyproject.toml` (`[gpu]` extra gains `faster-whisper`)
- Modify: `docs/smoke-gateway.md` (STT section)

- [ ] **Step 1: Add faster-whisper to the `[gpu]` extra**

```toml
gpu = ["fastembed-gpu>=0.3", "faster-whisper>=1.0.1"]
```

- [ ] **Step 2: Delete the standalone whisper service**

```bash
git rm build/whisper_api.py
```

- [ ] **Step 3: Drop it from the build context check** — in `build.sh`, change:

```bash
REQUIRED_FILES=("Dockerfile" "whisper_api.py" "entrypoint.sh" "pyproject.toml")
```

to:

```bash
REQUIRED_FILES=("Dockerfile" "entrypoint.sh" "pyproject.toml")
```

- [ ] **Step 4: Drop the COPY in `build/Dockerfile`** — remove the line `COPY whisper_api.py .` (the whisper *models* are still baked by the existing `whisper_nodes` step; only the standalone FastAPI file is gone).

- [ ] **Step 5: Add the STT smoke section** to `docs/smoke-gateway.md`:

```markdown
## STT

1. Raise a spec with an `stt` block, e.g. `{"slug":"whisper-small","model":"small"}`.
2. `curl -s localhost:8080/v1/audio/transcriptions \
     -F model=whisper-small -F file=@sample.wav | jq`
   → `{"text":"..."}`.
```

- [ ] **Step 6: Verify dev suite + commit**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS.

```bash
git add pyproject.toml build.sh build/Dockerfile docs/smoke-gateway.md
git commit -m "refactor(engine): absorb whisper_api into gateway; drop standalone service

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

## Self-Review

- **Spec coverage:** STT via faster-whisper ✓ (T2), standard `/v1/audio/transcriptions` model-in-body ✓ (T3), raise-spec `stt` block ✓ (T1), `/v1/models` union ✓ (T3), standalone `whisper_api.py` retired ✓ (T4), lazy import / dev without faster-whisper ✓ (T2).
- **Placeholder scan:** all code present; smoke is a documented runbook.
- **Type consistency:** `SttNode`, `Transcriber`, `build_transcribers`, `FasterWhisperTranscriber`, `_default_transcriber_factory`, `create_app(..., transcriber_factory=None)`, `app.state.transcribers` consistent across tasks. Task 3 restates the `create_app` signature growth and the `/v1/models` three-way union.
