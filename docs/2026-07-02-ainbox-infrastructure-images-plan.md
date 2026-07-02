# ainbox-infrastructure Image-gen Backend (FLUX) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline). Steps use checkbox (`- [ ]`).

**Goal:** Image generation on the gateway — FLUX.1-schnell via `diffusers`, served at the standard `POST /v1/images/generations`, returning base64 PNGs.

**Architecture:** Mirrors the TTS/STT backends. FLUX runs **in-process** via `diffusers`; a `Generator` protocol abstracts it; the real `DiffusersFluxGenerator` lazy-imports torch/diffusers and is injected via a factory. Unit tests use a fake — the package imports and tests with no GPU/torch. FLUX is the one backend that can't co-reside with the LLMs on a small card, so its raise-spec node adds `offload` + `quant` knobs.

**Tech Stack:** Same gateway package. `diffusers`, `torch`, `transformers`, `accelerate`, `sentencepiece`, `protobuf`, `Pillow` in the `[gpu]` extra. **No real build/download on zion** — the actual weight bake + fp8 quant config are validated in the GPU smoke.

## Global Constraints

- **OpenAI shape:** `POST /v1/images/generations`, JSON `{"model","prompt","n"?,"size"?,"response_format"?}`; response `{"created":0,"data":[{"b64_json": "<png>"}]}`. `size` "WxH" → width×height (default 1024×1024). Always `b64_json` (no URL hosting).
- **schnell defaults:** few-step (`num_inference_steps=4`), `guidance_scale=0.0`.
- **In-process, single model per slug.** `n>1` loops.
- **Lazy torch/diffusers import**; gateway imports/unit-tests without them.
- **Commit style:** conventional commits; `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## File Structure

- `src/ainbox_gateway/spec.py` — add `ImagesNode`, parse optional `images`.
- `src/ainbox_gateway/images.py` — `Generator` protocol, `build_generators`, lazy `DiffusersFluxGenerator`.
- `src/ainbox_gateway/app.py` — build generators in lifespan, `POST /v1/images/generations`, `/v1/models` + `_status()` include images.
- `pyproject.toml` — `[gpu]` gains the diffusion stack.
- `recipes/rtx4060_v1.json` — add `image_nodes`; `build/Dockerfile` — bake FLUX.
- `docs/smoke-gateway.md` — image-gen section (+ fp8/offload notes).
- Tests: `tests/test_spec.py`, `tests/test_images.py`, `tests/test_app.py`.

---

### Task 1: Raise-spec `images` block

**Files:** Modify `src/ainbox_gateway/spec.py`; Test `tests/test_spec.py`.

**Interfaces:** `@dataclass ImagesNode(slug, model, device="cuda", offload=False, quant="fp8", steps=4, guidance=0.0)`; `Spec.images: list[ImagesNode]` (default `[]`); `load_spec` parses optional `images`, node missing `slug`/`model` → `SpecError`.

- [ ] **Step 1: Failing tests** (append to `tests/test_spec.py`; add `ImagesNode` to the import)

```python
def test_images_optional_defaults_empty():
    spec = load_spec({"gateway": {"port": 8080}, "llm": [{"slug": "a"}]})
    assert spec.images == []


def test_images_parsed_with_knobs():
    spec = load_spec({"gateway": {"port": 8080}, "llm": [{"slug": "a"}],
                      "images": [{"slug": "flux", "model": "black-forest-labs/FLUX.1-schnell",
                                  "offload": True, "quant": "fp8"}]})
    assert spec.images == [ImagesNode(slug="flux",
                                      model="black-forest-labs/FLUX.1-schnell",
                                      device="cuda", offload=True, quant="fp8",
                                      steps=4, guidance=0.0)]


def test_images_node_missing_model_raises():
    with pytest.raises(SpecError):
        load_spec({"gateway": {"port": 8080}, "llm": [{"slug": "a"}],
                   "images": [{"slug": "flux"}]})
```

- [ ] **Step 2: Run to verify fail** — `ImportError: cannot import name 'ImagesNode'`.

- [ ] **Step 3: Implement** — after `TtsNode` in `spec.py`:

```python
@dataclass
class ImagesNode:
    slug: str
    model: str
    device: str = "cuda"
    offload: bool = False
    quant: str = "fp8"
    steps: int = 4
    guidance: float = 0.0
```

Add to `Spec`: `images: list[ImagesNode] = field(default_factory=list)`. Loader + wire:

```python
def _load_images(raw: dict) -> ImagesNode:
    if "slug" not in raw or "model" not in raw:
        raise SpecError("images node needs 'slug' and 'model'")
    return ImagesNode(slug=raw["slug"], model=raw["model"],
                      device=raw.get("device", "cuda"),
                      offload=raw.get("offload", False),
                      quant=raw.get("quant", "fp8"),
                      steps=raw.get("steps", 4),
                      guidance=raw.get("guidance", 0.0))
```

```python
        tts=[_load_tts(t) for t in data.get("tts", [])],
        images=[_load_images(i) for i in data.get("images", [])],
    )
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest tests/test_spec.py -q`.

- [ ] **Step 5: Commit** — `feat(gateway): raise-spec images block`.

---

### Task 2: Generator protocol + registry + lazy DiffusersFluxGenerator

**Files:** Create `src/ainbox_gateway/images.py`; Test `tests/test_images.py`.

**Interfaces:**
- `class Generator(Protocol)` with `slug: str` and `generate(self, prompt: str, n: int = 1, width: int = 1024, height: int = 1024) -> list[bytes]` (list of PNG bytes).
- `build_generators(spec, factory) -> dict[str, Generator]`.
- `DiffusersFluxGenerator` — lazy-imports torch/diffusers/PIL; loads FLUX pipeline (offload if set); generates `n` PNGs.

- [ ] **Step 1: Failing tests**

```python
# tests/test_images.py
from ainbox_gateway.spec import Spec, LlmNode, ImagesNode
from ainbox_gateway.images import build_generators


class FakeGen:
    def __init__(self, node):
        self.slug = node.slug

    def generate(self, prompt, n=1, width=1024, height=1024):
        return [f"png:{prompt}:{width}x{height}:{i}".encode() for i in range(n)]


def test_build_generators_maps_slug():
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a")],
                images=[ImagesNode(slug="flux", model="m")])
    g = build_generators(spec, factory=FakeGen)
    assert set(g) == {"flux"}
    assert g["flux"].generate("cat", n=2) == [b"png:cat:1024x1024:0", b"png:cat:1024x1024:1"]


def test_build_generators_empty_when_no_images():
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a")])
    assert build_generators(spec, factory=FakeGen) == {}


def test_module_imports_without_diffusers():
    import importlib
    import ainbox_gateway.images as m
    importlib.reload(m)
    assert hasattr(m, "DiffusersFluxGenerator")
```

- [ ] **Step 2: Run to verify fail** — `ModuleNotFoundError: ainbox_gateway.images`.

- [ ] **Step 3: Implement**

```python
# src/ainbox_gateway/images.py
"""In-process image-generation backends served at /v1/images/generations.

torch/diffusers are imported lazily inside DiffusersFluxGenerator so this
module (and the whole gateway) imports and unit-tests without them installed.
"""
from __future__ import annotations

import io
from typing import Callable, Protocol

from .spec import ImagesNode, Spec


class Generator(Protocol):
    slug: str

    def generate(self, prompt: str, n: int = 1,
                 width: int = 1024, height: int = 1024) -> list[bytes]: ...


def build_generators(
    spec: Spec, factory: Callable[[ImagesNode], "Generator"]
) -> dict[str, "Generator"]:
    return {node.slug: factory(node) for node in spec.images}


class DiffusersFluxGenerator:
    """Real generator over FLUX.1-schnell via diffusers.

    NOTE: not exercised on zion. The concrete fp8/quant loading path and the
    baked checkpoint are pinned in the GPU smoke (docs/smoke-gateway.md).
    """

    def __init__(self, node: ImagesNode):
        import torch  # lazy
        from diffusers import FluxPipeline  # lazy

        self.slug = node.slug
        self._steps = node.steps
        self._guidance = node.guidance
        dtype = torch.bfloat16
        self._pipe = FluxPipeline.from_pretrained(node.model, torch_dtype=dtype)
        if node.offload:
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe = self._pipe.to(node.device)

    def generate(self, prompt: str, n: int = 1,
                 width: int = 1024, height: int = 1024) -> list[bytes]:
        out: list[bytes] = []
        for _ in range(n):
            image = self._pipe(prompt, num_inference_steps=self._steps,
                               guidance_scale=self._guidance,
                               width=width, height=height).images[0]
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            out.append(buf.getvalue())
        return out
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest tests/test_images.py -q`.

- [ ] **Step 5: Commit** — `feat(gateway): generator protocol, registry, lazy DiffusersFluxGenerator`.

---

### Task 3: `/v1/images/generations` endpoint + lifespan + `/v1/models` + status

**Files:** Modify `src/ainbox_gateway/app.py`; Test `tests/test_app.py`.

**Interfaces:** `create_app(..., generator_factory=None)`; lifespan builds `app.state.generators`; `POST /v1/images/generations` (JSON `{model, prompt, n?, size?}`, unknown model → 404, missing prompt → 400, returns `{"created":0,"data":[{"b64_json":...}]}`); `/v1/models` + `_status()` include images.

- [ ] **Step 1: Failing tests** (append to `tests/test_app.py`; add `ImagesNode` to the spec import)

```python
import base64


class _FakeGen:
    def __init__(self, node):
        self.slug = node.slug

    def generate(self, prompt, n=1, width=1024, height=1024):
        return [f"PNG[{prompt}|{width}x{height}|{i}]".encode() for i in range(n)]


def _app_with_images():
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a", replicas=2)],
                images=[ImagesNode(slug="flux", model="m")])
    return create_app(spec, FakeSupervisor(), embedder_factory=_FakeEmbedder,
                      transcriber_factory=_FakeTranscriber,
                      synthesizer_factory=_FakeSynth, generator_factory=_FakeGen)


@pytest.mark.asyncio
async def test_images_generation_b64():
    async with _client(_app_with_images()) as c:
        r = await c.post("/v1/images/generations",
                         json={"model": "flux", "prompt": "a cat", "n": 2, "size": "512x768"})
    body = r.json()
    assert len(body["data"]) == 2
    assert base64.b64decode(body["data"][0]["b64_json"]) == b"PNG[a cat|512x768|0]"


@pytest.mark.asyncio
async def test_images_unknown_model_404():
    async with _client(_app_with_images()) as c:
        r = await c.post("/v1/images/generations", json={"model": "nope", "prompt": "x"})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_images_missing_prompt_400():
    async with _client(_app_with_images()) as c:
        r = await c.post("/v1/images/generations", json={"model": "flux"})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_models_union_includes_images():
    async with _client(_app_with_images()) as c:
        r = await c.get("/v1/models")
    assert [m["id"] for m in r.json()["data"]] == ["a", "flux"]
```

- [ ] **Step 2: Run to verify fail** — `create_app()` has no `generator_factory` / 404.

- [ ] **Step 3: Implement** — in `app.py`:

Imports: `import base64`; `from .images import Generator, build_generators`; add `ImagesNode` to the `.spec` import.

Default factory:

```python
def _default_generator_factory(node: ImagesNode) -> Generator:
    from .images import DiffusersFluxGenerator
    return DiffusersFluxGenerator(node)
```

Signature/lifespan/`_start`: add `generator_factory=None`; default it; in `_start` add `app.state.generators = build_generators(new_spec, generator_factory)`.

`_status()` gains `"images": sorted(app.state.generators)`.

Endpoint (beside `/v1/audio/speech`):

```python
    @app.post("/v1/images/generations")
    async def images(request: Request) -> Response:
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        model = payload.get("model")
        if not model:
            return JSONResponse({"error": "missing 'model'"}, status_code=400)
        prompt = payload.get("prompt")
        if not prompt:
            return JSONResponse({"error": "missing 'prompt'"}, status_code=400)
        gen = app.state.generators.get(model)
        if gen is None:
            return JSONResponse(
                {"error": f"image model '{model}' is not raised"}, status_code=404)
        n = int(payload.get("n", 1))
        size = payload.get("size", "1024x1024")
        try:
            w, h = (int(x) for x in size.lower().split("x"))
        except Exception:
            return JSONResponse({"error": f"bad size '{size}'"}, status_code=400)
        pngs = await asyncio.to_thread(gen.generate, prompt, n, w, h)
        data = [{"b64_json": base64.b64encode(p).decode()} for p in pngs]
        return JSONResponse({"created": 0, "data": data})
```

`/v1/models` union: add `| set(app.state.generators)`.

- [ ] **Step 4: Run whole suite** — `.venv/bin/python -m pytest tests/ -q`.

- [ ] **Step 5: Commit** — `feat(gateway): /v1/images/generations endpoint + models/status union`.

---

### Task 4: Bake FLUX + deps + smoke

**Files:** `pyproject.toml`, `recipes/rtx4060_v1.json`, `build/Dockerfile`, `docs/smoke-gateway.md`.

- [ ] **Step 1: `[gpu]` extra** gains the diffusion stack:

```toml
gpu = ["fastembed-gpu>=0.3", "faster-whisper>=1.0.1", "kokoro>=0.9",
       "diffusers>=0.30", "torch>=2.3", "transformers>=4.44", "accelerate>=0.33",
       "sentencepiece>=0.2", "protobuf>=4", "Pillow>=10"]
```

- [ ] **Step 2: Recipe** — add `image_nodes` to `recipes/rtx4060_v1.json`:

```json
  "image_nodes": [
    { "model": "black-forest-labs/FLUX.1-schnell" }
  ],
```

- [ ] **Step 3: Dockerfile** — bake FLUX after the Kokoro bake (its own layer for blob-delta):

```dockerfile
# 6d. Pre-loading image-gen models (FLUX weights + encoders) — LARGE layer
RUN jq -r '.image_nodes[]?.model' /tmp/recipe.json | while read -r model; do \
    if [ -n "$model" ] && [ "$model" != "null" ]; then \
        echo "[BUILD] Baking FLUX image model: $model..."; \
        python3 -c "from diffusers import FluxPipeline; import torch; FluxPipeline.from_pretrained('$model', torch_dtype=torch.bfloat16)"; \
    fi; \
done
```

- [ ] **Step 4: Smoke** — add to `docs/smoke-gateway.md`:

```markdown
## Image generation (FLUX)

> FLUX wants ~16–24 GB VRAM. On a 16 GB card, raise it *instead of* the 9B, or
> set `"offload": true` on the images node. The exact fp8 checkpoint + quant
> config are finalized here (not on zion).

1. Raise a spec with an `images` block, e.g.
   `{"slug":"flux","model":"black-forest-labs/FLUX.1-schnell","quant":"fp8","offload":true}`.
2. `curl -s localhost:8080/v1/images/generations \
     -H 'content-type: application/json' \
     -d '{"model":"flux","prompt":"a red bicycle, studio photo","size":"1024x1024"}' \
     | jq -r '.data[0].b64_json' | base64 -d > out.png && file out.png`
   → `PNG image data, 1024 x 1024`.
```

- [ ] **Step 5: Verify dev suite + commit** — `.venv/bin/python -m pytest tests/ -q`; then `feat(engine): bake FLUX.1-schnell image-gen (diffusion stack in gpu extra)`.

## Self-Review

- **Coverage:** FLUX/diffusers runtime ✓ (T2), `/v1/images/generations` OpenAI shape ✓ (T3), raise-spec `images` + offload/quant knobs ✓ (T1), `/v1/models`+status union ✓ (T3), baked + heavy-layer note ✓ (T4), lazy import / dev without torch ✓ (T2).
- **Placeholders:** none in shipped code; the fp8/quant specifics are explicitly deferred to the GPU smoke (documented, not a silent gap).
- **Type consistency:** `ImagesNode`, `Generator`, `build_generators`, `DiffusersFluxGenerator`, `_default_generator_factory`, `create_app(..., generator_factory=None)`, `app.state.generators` consistent across tasks. T3 restates the `create_app` growth and the five-way `/v1/models` union.
