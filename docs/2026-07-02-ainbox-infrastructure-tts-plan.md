# ainbox-infrastructure TTS Backend (Kokoro) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline). Steps use checkbox (`- [ ]`).

**Goal:** Text-to-speech on the gateway — Kokoro-82M served at the standard `POST /v1/audio/speech`, returning WAV audio.

**Architecture:** Mirrors STT/embeddings. Kokoro runs **in-process**; a `Synthesizer` protocol abstracts it; the real `KokoroSynthesizer` lazy-imports `kokoro` and is injected via a factory, so unit tests use fakes and the package imports without kokoro. Raise-spec gains an optional `tts` block; the recipe bakes the model.

**Tech Stack:** Same gateway package. `kokoro` in the `[gpu]` extra; `espeak-ng` system dep in the image. WAV encoding via stdlib `wave` (no extra dep).

## Global Constraints

- **OpenAI shape:** `POST /v1/audio/speech`, JSON body `{"model", "input", "voice"?, "response_format"?}`; returns raw audio bytes, `Content-Type: audio/wav`. (v1 emits WAV regardless of `response_format`; mp3 is a later add.)
- **In-process, single model per slug**; the request `voice` selects the Kokoro voice (falls back to the node's default).
- **Lazy `kokoro` import**; gateway imports/unit-tests without it.
- **Commit style:** conventional commits; `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## File Structure

- `src/ainbox_gateway/spec.py` — add `TtsNode`, parse optional `tts`.
- `src/ainbox_gateway/tts.py` — `Synthesizer` protocol, `build_synthesizers`, `KokoroSynthesizer` (lazy) + WAV encode helper.
- `src/ainbox_gateway/app.py` — build synthesizers in lifespan, `POST /v1/audio/speech`, `/v1/models` union, `_status()` gains `tts`.
- `pyproject.toml` — `kokoro` in `[gpu]`.
- `recipes/rtx4060_v1.json` — add `tts_nodes`; `build/Dockerfile` — `espeak-ng` apt + bake Kokoro.
- `docs/smoke-gateway.md` — TTS section.
- Tests: `tests/test_spec.py`, `tests/test_tts.py`, `tests/test_app.py`.

---

### Task 1: Raise-spec `tts` block

**Files:** Modify `src/ainbox_gateway/spec.py`; Test `tests/test_spec.py`.

**Interfaces:** `@dataclass TtsNode(slug: str, model: str, device: str = "cuda", lang_code: str = "a", voice: str = "af_heart")`; `Spec.tts: list[TtsNode]` (default `[]`); `load_spec` parses optional `tts`, node missing `slug`/`model` → `SpecError`.

- [ ] **Step 1: Failing tests** (append to `tests/test_spec.py`; add `TtsNode` to the import)

```python
def test_tts_optional_defaults_empty():
    spec = load_spec({"gateway": {"port": 8080}, "llm": [{"slug": "a"}]})
    assert spec.tts == []


def test_tts_parsed():
    spec = load_spec({"gateway": {"port": 8080}, "llm": [{"slug": "a"}],
                      "tts": [{"slug": "voice", "model": "kokoro",
                               "lang_code": "e", "voice": "ef_dora"}]})
    assert spec.tts == [TtsNode(slug="voice", model="kokoro", device="cuda",
                                lang_code="e", voice="ef_dora")]


def test_tts_node_missing_model_raises():
    with pytest.raises(SpecError):
        load_spec({"gateway": {"port": 8080}, "llm": [{"slug": "a"}],
                   "tts": [{"slug": "voice"}]})
```

- [ ] **Step 2: Run to verify fail** — `.venv/bin/python -m pytest tests/test_spec.py -k tts -v` → `ImportError: cannot import name 'TtsNode'`.

- [ ] **Step 3: Implement** — after `SttNode` in `spec.py`:

```python
@dataclass
class TtsNode:
    slug: str
    model: str
    device: str = "cuda"
    lang_code: str = "a"
    voice: str = "af_heart"
```

Add to `Spec`: `tts: list[TtsNode] = field(default_factory=list)`. Add loader + wire into `load_spec`:

```python
def _load_tts(raw: dict) -> TtsNode:
    if "slug" not in raw or "model" not in raw:
        raise SpecError("tts node needs 'slug' and 'model'")
    return TtsNode(slug=raw["slug"], model=raw["model"],
                   device=raw.get("device", "cuda"),
                   lang_code=raw.get("lang_code", "a"),
                   voice=raw.get("voice", "af_heart"))
```

```python
        stt=[_load_stt(s) for s in data.get("stt", [])],
        tts=[_load_tts(t) for t in data.get("tts", [])],
    )
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest tests/test_spec.py -q`.

- [ ] **Step 5: Commit**

```bash
git add src/ainbox_gateway/spec.py tests/test_spec.py
git commit -m "feat(gateway): raise-spec tts block

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Synthesizer protocol + registry + lazy KokoroSynthesizer

**Files:** Create `src/ainbox_gateway/tts.py`; Test `tests/test_tts.py`.

**Interfaces:**
- `class Synthesizer(Protocol)` with `slug: str` and `synthesize(self, text: str, voice: str | None = None) -> bytes` (WAV bytes).
- `build_synthesizers(spec, factory) -> dict[str, Synthesizer]`.
- `KokoroSynthesizer` — lazy-imports `kokoro`; concatenates pipeline audio; encodes WAV via `_wav_bytes(samples, rate)` (stdlib `wave`, 24 kHz, int16).
- `_wav_bytes(samples: list[float], rate: int = 24000) -> bytes` — module-level, testable without kokoro.

- [ ] **Step 1: Failing tests**

```python
# tests/test_tts.py
import io
import wave

from ainbox_gateway.spec import Spec, LlmNode, TtsNode
from ainbox_gateway.tts import build_synthesizers, _wav_bytes


class FakeSynth:
    def __init__(self, node):
        self.slug = node.slug

    def synthesize(self, text, voice=None):
        return f"wav:{text}:{voice}".encode()


def test_build_synthesizers_maps_slug():
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a")],
                tts=[TtsNode(slug="voice", model="kokoro")])
    s = build_synthesizers(spec, factory=FakeSynth)
    assert set(s) == {"voice"}
    assert s["voice"].synthesize("hi", "ef_dora") == b"wav:hi:ef_dora"


def test_build_synthesizers_empty_when_no_tts():
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a")])
    assert build_synthesizers(spec, factory=FakeSynth) == {}


def test_wav_bytes_is_valid_riff():
    data = _wav_bytes([0.0, 0.5, -0.5, 1.0], rate=24000)
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    with wave.open(io.BytesIO(data)) as w:
        assert w.getframerate() == 24000
        assert w.getnframes() == 4


def test_module_imports_without_kokoro():
    import importlib
    import ainbox_gateway.tts as m
    importlib.reload(m)
    assert hasattr(m, "KokoroSynthesizer")
```

- [ ] **Step 2: Run to verify fail** — `ModuleNotFoundError: ainbox_gateway.tts`.

- [ ] **Step 3: Implement**

```python
# src/ainbox_gateway/tts.py
"""In-process text-to-speech backends served at /v1/audio/speech.

kokoro is imported lazily inside KokoroSynthesizer so this module (and the
whole gateway) imports and unit-tests without it installed.
"""
from __future__ import annotations

import io
import wave
from typing import Callable, Protocol

from .spec import TtsNode, Spec


class Synthesizer(Protocol):
    slug: str

    def synthesize(self, text: str, voice: str | None = None) -> bytes: ...


def build_synthesizers(
    spec: Spec, factory: Callable[[TtsNode], "Synthesizer"]
) -> dict[str, "Synthesizer"]:
    return {node.slug: factory(node) for node in spec.tts}


def _wav_bytes(samples: list[float], rate: int = 24000) -> bytes:
    """Encode float samples in [-1, 1] as 16-bit mono PCM WAV."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for s in samples:
            v = int(max(-1.0, min(1.0, float(s))) * 32767)
            frames += int(v).to_bytes(2, "little", signed=True)
        w.writeframes(bytes(frames))
    return buf.getvalue()


class KokoroSynthesizer:
    """Real synthesizer over Kokoro-82M (24 kHz)."""

    def __init__(self, node: TtsNode):
        from kokoro import KPipeline  # lazy

        self.slug = node.slug
        self._default_voice = node.voice
        self._pipeline = KPipeline(lang_code=node.lang_code)

    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        import numpy as np

        chunks = [audio for _, _, audio in
                  self._pipeline(text, voice=voice or self._default_voice)]
        samples = np.concatenate(chunks) if chunks else np.zeros(0)
        return _wav_bytes(samples.tolist(), rate=24000)
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest tests/test_tts.py -q`.

- [ ] **Step 5: Commit**

```bash
git add src/ainbox_gateway/tts.py tests/test_tts.py
git commit -m "feat(gateway): synthesizer protocol, registry, lazy KokoroSynthesizer

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `/v1/audio/speech` endpoint + lifespan + `/v1/models` + status

**Files:** Modify `src/ainbox_gateway/app.py`; Test `tests/test_app.py`.

**Interfaces:** `create_app(..., synthesizer_factory=None)`; lifespan builds `app.state.synthesizers`; `POST /v1/audio/speech` (JSON `{model, input, voice?}`, unknown model → 404, missing input → 400, returns WAV bytes `audio/wav`); `/v1/models` + `_status()` include tts.

- [ ] **Step 1: Failing tests** (append to `tests/test_app.py`; add `TtsNode` to the spec import)

```python
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
```

- [ ] **Step 2: Run to verify fail** — `create_app()` has no `synthesizer_factory` / 404.

- [ ] **Step 3: Implement** — in `app.py`:

Imports: `from .tts import Synthesizer, build_synthesizers`; add `TtsNode` to the `.spec` import.

Default factory (beside the others):

```python
def _default_synthesizer_factory(node: TtsNode) -> Synthesizer:
    from .tts import KokoroSynthesizer
    return KokoroSynthesizer(node)
```

Signature + lifespan + `_start`: add `synthesizer_factory=None`; `synthesizer_factory = synthesizer_factory or _default_synthesizer_factory`; in `_start` add `app.state.synthesizers = build_synthesizers(new_spec, synthesizer_factory)`.

`_status()` gains `"tts": sorted(app.state.synthesizers)`.

Endpoint (beside `/v1/audio/transcriptions`):

```python
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
```

`/v1/models` union: add `| set(app.state.synthesizers)`.

- [ ] **Step 4: Run whole suite** — `.venv/bin/python -m pytest tests/ -q`.

- [ ] **Step 5: Commit**

```bash
git add src/ainbox_gateway/app.py tests/test_app.py
git commit -m "feat(gateway): /v1/audio/speech endpoint + models/status union

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Bake Kokoro + deps + smoke

**Files:** `pyproject.toml`, `recipes/rtx4060_v1.json`, `build/Dockerfile`, `docs/smoke-gateway.md`.

- [ ] **Step 1: `[gpu]` extra** gains `kokoro`:

```toml
gpu = ["fastembed-gpu>=0.3", "faster-whisper>=1.0.1", "kokoro>=0.9"]
```

- [ ] **Step 2: Recipe** — add a `tts_nodes` list to `recipes/rtx4060_v1.json`:

```json
  "tts_nodes": [
    { "model": "kokoro" }
  ],
```

- [ ] **Step 3: Dockerfile** — add `espeak-ng` to the apt install line (stage 1 system deps), and bake Kokoro after the gateway `[gpu]` install (beside the embedding bake):

```dockerfile
RUN jq -r '.tts_nodes[]?.model' /tmp/recipe.json | while read -r model; do \
    if [ "$model" = "kokoro" ]; then \
        echo "[BUILD] Baking Kokoro TTS..."; \
        python3 -c "from kokoro import KPipeline; KPipeline(lang_code='a')"; \
    fi; \
done
```

(Add `espeak-ng` to the existing `apt-get install -y --no-install-recommends ...` list.)

- [ ] **Step 4: Smoke** — add to `docs/smoke-gateway.md`:

```markdown
## TTS

1. Raise a spec with a `tts` block, e.g. `{"slug":"voice","model":"kokoro","lang_code":"e","voice":"ef_dora"}`.
2. `curl -s localhost:8080/v1/audio/speech \
     -H 'content-type: application/json' \
     -d '{"model":"voice","input":"Hola, soy AInBox.","voice":"ef_dora"}' \
     -o out.wav && file out.wav`  → `RIFF (little-endian) WAVE audio`.
```

- [ ] **Step 5: Verify dev suite + commit**

```bash
.venv/bin/python -m pytest tests/ -q
git add pyproject.toml recipes/rtx4060_v1.json build/Dockerfile docs/smoke-gateway.md
git commit -m "feat(engine): bake Kokoro TTS (espeak-ng + gpu extra)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

## Self-Review

- **Coverage:** Kokoro runtime ✓ (T2), `/v1/audio/speech` OpenAI shape ✓ (T3), raise-spec `tts` ✓ (T1), `/v1/models`+status union ✓ (T3), baked + espeak-ng ✓ (T4), lazy import / dev without kokoro ✓ (T2, WAV helper unit-tested).
- **Placeholders:** none; smoke is a runbook.
- **Type consistency:** `TtsNode`, `Synthesizer`, `build_synthesizers`, `KokoroSynthesizer`, `_wav_bytes`, `_default_synthesizer_factory`, `create_app(..., synthesizer_factory=None)`, `app.state.synthesizers` consistent across tasks. T3 restates the `create_app` signature growth and the four-way `/v1/models` union.
