# ainbox-infrastructure Embeddings Backend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) or subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a GPU embedding backend to the gateway — the exact `paraphrase-multilingual-MiniLM-L12-v2` model the apps already use, served at pure-OpenAI `/v1/embeddings`, so existing magpie/superbot vectors stay valid (no reindex).

**Architecture:** The embedder is loaded **in-process** in the gateway (fastembed is a library, not a server). An `Embedder` protocol abstracts it; the real `FastEmbedEmbedder` lazy-imports fastembed and is injected via a factory, so all routing/shaping logic is unit-tested against a fake. The raise-spec gains an optional `embeddings` block; the recipe bakes the model into the image via fastembed at build time.

**Tech Stack:** Same gateway package (Python, FastAPI, httpx). `fastembed-gpu` (ONNX Runtime CUDA) as an optional `[gpu]` extra — installed in the image, not in dev.

## Global Constraints

- **Same model, no reindex:** `paraphrase-multilingual-MiniLM-L12-v2`, 384-d. GPU via fastembed-gpu produces vectors numerically equivalent to the apps' current CPU vectors.
- **In-process, not a subprocess** — no port, no round-robin for embeddings (single loaded model per slug).
- **Pure OpenAI:** `POST /v1/embeddings` with `{"model", "input"}` where `input` is a string or list of strings; response is the OpenAI embeddings shape. `/v1/models` lists embedding slugs alongside LLM slugs.
- **Lazy fastembed import:** the gateway package must import and unit-test **without** fastembed installed. Real embedder loads fastembed only when instantiated at startup.
- **Commit style:** conventional commits; `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## File Structure

- `src/ainbox_gateway/spec.py` — add `EmbeddingsNode`, parse optional `embeddings`.
- `src/ainbox_gateway/embeddings.py` — `Embedder` protocol, `build_embedders`, `FastEmbedEmbedder` (lazy).
- `src/ainbox_gateway/app.py` — build embedders in lifespan (factory param), `POST /v1/embeddings`, `/v1/models` includes embeddings.
- `pyproject.toml` — `[gpu]` extra with `fastembed-gpu`.
- `recipes/rtx4060_v1.json` — drop e5-small, add `embedding_nodes`.
- `build/Dockerfile` — bake embedding models via fastembed.
- Tests: `tests/test_spec.py`, `tests/test_embeddings.py`, `tests/test_app.py`.
- `docs/smoke-gateway.md` — add the embeddings + no-reindex equivalence check.

---

### Task 1: Raise-spec `embeddings` block

**Files:**
- Modify: `src/ainbox_gateway/spec.py`
- Test: `tests/test_spec.py` (add cases)

**Interfaces:**
- Produces:
  - `@dataclass EmbeddingsNode(slug: str, model: str, device: str = "cuda")`
  - `Spec` gains `embeddings: list[EmbeddingsNode]` (default `[]`).
  - `load_spec` parses optional `embeddings`; a node missing `slug` or `model` raises `SpecError`.

- [ ] **Step 1: Write failing tests** (append to `tests/test_spec.py`)

```python
from ainbox_gateway.spec import EmbeddingsNode


def test_embeddings_optional_defaults_empty():
    spec = load_spec({"gateway": {"port": 8080}, "llm": [{"slug": "a"}]})
    assert spec.embeddings == []


def test_embeddings_parsed():
    spec = load_spec({"gateway": {"port": 8080}, "llm": [{"slug": "a"}],
                      "embeddings": [{"slug": "emb", "model": "MiniLM"}]})
    assert spec.embeddings == [EmbeddingsNode(slug="emb", model="MiniLM", device="cuda")]


def test_embeddings_node_missing_model_raises():
    with pytest.raises(SpecError):
        load_spec({"gateway": {"port": 8080}, "llm": [{"slug": "a"}],
                   "embeddings": [{"slug": "emb"}]})
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_spec.py -k embeddings -v`
Expected: FAIL — `ImportError: cannot import name 'EmbeddingsNode'`.

- [ ] **Step 3: Implement**

In `spec.py`, add the dataclass (after `LlmNode`):

```python
@dataclass
class EmbeddingsNode:
    slug: str
    model: str
    device: str = "cuda"
```

Add `embeddings` to `Spec`:

```python
@dataclass
class Spec:
    gateway_port: int
    llm: list[LlmNode]
    embeddings: list["EmbeddingsNode"] = field(default_factory=list)
```

Add a loader + wire it into `load_spec` (before the `return`):

```python
def _load_embeddings(raw: dict) -> EmbeddingsNode:
    if "slug" not in raw or "model" not in raw:
        raise SpecError("embeddings node needs 'slug' and 'model'")
    return EmbeddingsNode(slug=raw["slug"], model=raw["model"],
                          device=raw.get("device", "cuda"))
```

```python
    return Spec(
        gateway_port=gateway["port"],
        llm=[_load_node(n) for n in raw_llm],
        embeddings=[_load_embeddings(e) for e in data.get("embeddings", [])],
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_spec.py -v`
Expected: PASS (all spec tests, old + new).

- [ ] **Step 5: Commit**

```bash
git add src/ainbox_gateway/spec.py tests/test_spec.py
git commit -m "feat(gateway): raise-spec embeddings block

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Embedder protocol + registry + lazy FastEmbedEmbedder

**Files:**
- Create: `src/ainbox_gateway/embeddings.py`
- Test: `tests/test_embeddings.py`

**Interfaces:**
- Consumes: `EmbeddingsNode`, `Spec` (Task 1).
- Produces:
  - `class Embedder(Protocol)` with attr `slug: str` and `embed(self, texts: list[str]) -> list[list[float]]`.
  - `build_embedders(spec: Spec, factory: Callable[[EmbeddingsNode], Embedder]) -> dict[str, Embedder]`.
  - `class FastEmbedEmbedder` — real impl; **imports fastembed lazily inside `__init__`** so the module imports without fastembed. `embed` returns plain lists (no numpy).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_embeddings.py
from ainbox_gateway.spec import Spec, LlmNode, EmbeddingsNode
from ainbox_gateway.embeddings import build_embedders


class FakeEmbedder:
    def __init__(self, node):
        self.slug = node.slug
        self.model = node.model

    def embed(self, texts):
        return [[float(len(t))] for t in texts]


def test_build_embedders_maps_slug_to_embedder():
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a")],
                embeddings=[EmbeddingsNode(slug="emb", model="MiniLM")])
    embedders = build_embedders(spec, factory=FakeEmbedder)
    assert set(embedders) == {"emb"}
    assert embedders["emb"].embed(["ab", "xyz"]) == [[2.0], [3.0]]


def test_build_embedders_empty_when_no_embeddings():
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a")])
    assert build_embedders(spec, factory=FakeEmbedder) == {}


def test_module_imports_without_fastembed():
    # Importing the module must not require fastembed to be installed.
    import importlib
    import ainbox_gateway.embeddings as m
    importlib.reload(m)
    assert hasattr(m, "FastEmbedEmbedder")
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_embeddings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ainbox_gateway.embeddings'`.

- [ ] **Step 3: Implement**

```python
# src/ainbox_gateway/embeddings.py
"""In-process embedding backends served at /v1/embeddings.

fastembed is imported lazily inside FastEmbedEmbedder so this module (and the
whole gateway) imports and unit-tests without fastembed installed.
"""
from __future__ import annotations

from typing import Callable, Protocol

from .spec import EmbeddingsNode, Spec


class Embedder(Protocol):
    slug: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def build_embedders(
    spec: Spec, factory: Callable[[EmbeddingsNode], "Embedder"]
) -> dict[str, "Embedder"]:
    return {node.slug: factory(node) for node in spec.embeddings}


class FastEmbedEmbedder:
    """Real embedder over fastembed (ONNX; CUDA when device='cuda')."""

    def __init__(self, node: EmbeddingsNode):
        from fastembed import TextEmbedding  # lazy

        self.slug = node.slug
        providers = ["CUDAExecutionProvider"] if node.device == "cuda" else None
        self._model = TextEmbedding(model_name=node.model, providers=providers)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.embed(texts)]
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_embeddings.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ainbox_gateway/embeddings.py tests/test_embeddings.py
git commit -m "feat(gateway): embedder protocol, registry, lazy FastEmbedEmbedder

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `/v1/embeddings` endpoint + lifespan wiring + `/v1/models`

**Files:**
- Modify: `src/ainbox_gateway/app.py`
- Test: `tests/test_app.py` (add cases)

**Interfaces:**
- Consumes: `build_embedders`, `Embedder` (Task 2).
- Produces:
  - `create_app(spec, supervisor, client=None, embedder_factory=None)` — lifespan also builds `app.state.embedders = build_embedders(spec, embedder_factory or _default_embedder_factory)`.
  - `POST /v1/embeddings`: body `{"model", "input"}` (`input` str or list[str]); resolves the embedder by slug (unknown → 404, missing model → 400); returns `{"object":"list","data":[{"object":"embedding","index":i,"embedding":vec}],"model":slug,"usage":{"prompt_tokens":0,"total_tokens":0}}`. Inference runs in `asyncio.to_thread`.
  - `GET /v1/models` includes embedding slugs (sorted union with LLM slugs).

- [ ] **Step 1: Write failing tests** (append to `tests/test_app.py`)

```python
from ainbox_gateway.spec import EmbeddingsNode


class _FakeEmbedder:
    def __init__(self, node):
        self.slug = node.slug

    def embed(self, texts):
        return [[float(len(t)), 0.5] for t in texts]


def _app_with_embeddings():
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a", replicas=2)],
                embeddings=[EmbeddingsNode(slug="emb", model="MiniLM")])
    return create_app(spec, FakeSupervisor(), embedder_factory=_FakeEmbedder)


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
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_app.py -k embeddings -v`
Expected: FAIL — `create_app()` has no `embedder_factory` / 404 on `/v1/embeddings`.

- [ ] **Step 3: Implement** — in `app.py`:

Add imports:

```python
import asyncio
from .embeddings import build_embedders, Embedder
from .spec import EmbeddingsNode
```

Add a default factory near the top (module level):

```python
def _default_embedder_factory(node: "EmbeddingsNode") -> "Embedder":
    from .embeddings import FastEmbedEmbedder
    return FastEmbedEmbedder(node)
```

Change the signature and lifespan:

```python
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
```

Add the endpoint (before `app.state.client = client`):

```python
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
```

Update `/v1/models` to union LLM + embedding slugs:

```python
    @app.get("/v1/models")
    async def list_models() -> Response:
        slugs = sorted(set(_router().models()) | set(app.state.embedders))
        data = [{"id": s, "object": "model", "owned_by": "ainbox"} for s in slugs]
        return JSONResponse({"object": "list", "data": data})
```

- [ ] **Step 4: Run the whole suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS (all green; existing `/v1/models` test still passes — LLM-only app has empty embedders so the union is just `["a"]`).

- [ ] **Step 5: Commit**

```bash
git add src/ainbox_gateway/app.py tests/test_app.py
git commit -m "feat(gateway): /v1/embeddings endpoint + models union

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Bake the model + deps + smoke (recipe, Dockerfile, pyproject)

**Files:**
- Modify: `pyproject.toml` (add `[gpu]` extra)
- Modify: `recipes/rtx4060_v1.json` (drop e5-small, add `embedding_nodes`)
- Modify: `build/Dockerfile` (bake embedding models; install `.[gpu]`)
- Modify: `docs/smoke-gateway.md` (embeddings + no-reindex check)

**Interfaces:** No Python API changes — packaging + image only.

- [ ] **Step 1: Add the `[gpu]` extra to `pyproject.toml`**

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "respx>=0.21", "asgi-lifespan>=2.1"]
gpu = ["fastembed-gpu>=0.3"]
```

- [ ] **Step 2: Update `recipes/rtx4060_v1.json`** — remove the `e5-small` entry from `llama_node`, add an `embedding_nodes` list:

```json
{
  "whisper_nodes": [
    { "model": "tiny", "alias": "fast_stt" },
    { "model": "small", "alias": "standard_stt" }
  ],
  "embedding_nodes": [
    { "model": "paraphrase-multilingual-MiniLM-L12-v2" }
  ],
  "llama_node": [
    { "url": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf?download=true", "alias": "qwen3.5-9b" },
    { "url": "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf?download=true", "alias": "qwen3.5-4b" },
    { "url": "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf?download=true", "alias": "qwen3.5-2b" }
  ]
}
```

- [ ] **Step 3: Bake embeddings in `build/Dockerfile`** — after the whisper pre-load block, add:

```dockerfile
# Pre-loading embedding models defined in the recipe (fastembed ONNX cache)
RUN jq -r '.embedding_nodes[]?.model' /tmp/recipe.json | while read -r model; do \
    if [ -n "$model" ] && [ "$model" != "null" ]; then \
        echo "[BUILD] Baking embedding model: $model..."; \
        python3 -c "from fastembed import TextEmbedding; TextEmbedding(model_name='$model')"; \
    fi; \
done
```

And change the gateway install to pull the gpu extra (fastembed-gpu):

```dockerfile
COPY gateway /app/gateway
RUN uv pip install --system '/app/gateway[gpu]'
```

(The `fastembed` used by the bake step comes from `fastembed-gpu`; ensure the gateway `[gpu]` install runs **before** the embedding-bake step, or add a dedicated `RUN uv pip install --system fastembed-gpu` ahead of the bake. Order the Dockerfile so fastembed is present when the bake runs.)

- [ ] **Step 4: Extend `docs/smoke-gateway.md`** with an embeddings section:

```markdown
## Embeddings (no-reindex equivalence)

1. Raise a spec with an `embeddings` block (see `deploy/example.json`).
2. `curl -s localhost:8080/v1/embeddings -H 'content-type: application/json' \
     -d '{"model":"text-embedding-minilm","input":["hola mundo"]}' | jq '.data[0].embedding | length'`
   → `384`.
3. **No-reindex check (GPU):** embed a fixture string here and compare against the
   same string embedded by the apps' current CPU fastembed MiniLM; cosine
   similarity must be ≈1.0 (diff < 1e-3). Confirms stored vectors stay valid.
```

- [ ] **Step 5: Verify dev suite still green + commit** (image build is the GPU smoke, not run here)

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS.

```bash
git add pyproject.toml recipes/rtx4060_v1.json build/Dockerfile docs/smoke-gateway.md
git commit -m "feat(engine): bake MiniLM embeddings, drop e5-small, gpu extra

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

## Self-Review

- **Spec coverage:** MiniLM-on-GPU ✓ (T2 FastEmbedEmbedder, CUDA provider), no-reindex ✓ (same model; equivalence in smoke), `/v1/embeddings` OpenAI shape ✓ (T3), `/v1/models` union ✓ (T3), raise-spec embeddings block ✓ (T1), model baked / e5 dropped ✓ (T4), lazy import / dev without fastembed ✓ (T2).
- **Placeholder scan:** all code present; the one manual item (no-reindex equivalence) is a documented GPU check, not a code placeholder.
- **Type consistency:** `EmbeddingsNode`, `Embedder`, `build_embedders`, `FastEmbedEmbedder`, `_default_embedder_factory`, `create_app(..., embedder_factory=None)`, `app.state.embedders` are consistent across tasks. Task 3 restates the `create_app` signature change and the `/v1/models` union so out-of-order readers stay consistent.
