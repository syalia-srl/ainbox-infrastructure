# ainbox-infrastructure Gateway Core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the gateway service that reads a raise-spec, presents one pure-OpenAI `/v1/*` front door, routes by the `model` slug to LLM backend pools, and round-robins same-slug replicas.

**Architecture:** A FastAPI service (`ainbox_gateway`) with four focused units — a raise-spec loader, a round-robin `Pool`, a slug→pool `Router`, and a `Supervisor` that turns spec nodes into `llama-server` launch commands. The app forwards OpenAI requests to the resolved backend over `httpx` (streaming passthrough). Backend process spawning sits behind a `Supervisor` protocol so all routing logic is unit-tested against fakes; a real `llama-server` is only exercised by a documented integration smoke.

**Tech Stack:** Python 3.10+, FastAPI, uvicorn, httpx, pytest, pytest-asyncio, respx (httpx mocking). llama.cpp `llama-server` (already baked into the image at `/app/llama-server`).

## Global Constraints

- **Scope:** LLM chat only (phases: gateway + `/v1/chat/completions` + `/v1/completions` + `/v1/models`). Embeddings, STT, TTS, image-gen, the tiny UI, warden integration, and the deployment swap are **out of scope** for this plan — they are follow-on plans.
- **Pure OpenAI surface:** the only public port is `gateway.port`; internal backend ports are never exposed. Callers select the target with the standard `model` field.
- **No hot-swap:** the raised set is fixed for the lifetime of the process. Changing it is a full relaunch (a later UI plan). This plan raises what the spec declares at startup and tears it down at shutdown.
- **No `--embedding` on chat models** — the argv builder must never emit it (regression from the old `entrypoint.sh`).
- **Round-robin** across replicas sharing a slug; internal ports assigned from a base (default 9000) upward, deterministically.
- **Python package name:** `ainbox_gateway`, src-layout under `src/`, tests under `tests/`.
- **Commit style:** conventional commits; end each commit body with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## File Structure

- `pyproject.toml` — gateway package metadata + deps (root).
- `src/ainbox_gateway/__init__.py` — version.
- `src/ainbox_gateway/spec.py` — raise-spec dataclasses + `load_spec`.
- `src/ainbox_gateway/pool.py` — `Backend`, `Pool` (round-robin).
- `src/ainbox_gateway/router.py` — `Router`, `UnknownModel`, `build_router`.
- `src/ainbox_gateway/supervisor.py` — `assign_ports`, `llama_argv`, `Supervisor` protocol, `LlamaSupervisor`.
- `src/ainbox_gateway/app.py` — FastAPI app, OpenAI proxy endpoints, lifespan wiring.
- `tests/test_spec.py`, `tests/test_pool.py`, `tests/test_router.py`, `tests/test_supervisor.py`, `tests/test_app.py`.
- `build/Dockerfile`, `build/entrypoint.sh` — modified in the final task to launch the gateway.

---

### Task 1: Package scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/ainbox_gateway/__init__.py`
- Test: `tests/test_import.py`

**Interfaces:**
- Produces: importable package `ainbox_gateway` with `__version__: str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_import.py
import ainbox_gateway


def test_package_has_version():
    assert isinstance(ainbox_gateway.__version__, str)
    assert ainbox_gateway.__version__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_import.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ainbox_gateway'`.

- [ ] **Step 3: Write the package + pyproject**

```toml
# pyproject.toml
[project]
name = "ainbox-gateway"
version = "0.1.0"
description = "Pure-OpenAI gateway for the AI-n-Box inference engine"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.110.0",
    "uvicorn>=0.29.0",
    "httpx>=0.27.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "respx>=0.21"]

[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
pythonpath = ["src"]
```

```python
# src/ainbox_gateway/__init__.py
__version__ = "0.1.0"
```

- [ ] **Step 4: Install dev deps + run test to verify it passes**

Run: `pip install -e '.[dev]' && pytest tests/test_import.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/ainbox_gateway/__init__.py tests/test_import.py
git commit -m "feat(gateway): scaffold ainbox_gateway package

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Raise-spec model + loader

**Files:**
- Create: `src/ainbox_gateway/spec.py`
- Test: `tests/test_spec.py`

**Interfaces:**
- Produces:
  - `@dataclass LoraSpec(file: str, alias: str, scale: float = 1.0)`
  - `@dataclass LlmNode(slug: str, replicas: int = 1, n_ctx: int = 4096, n_gpu_layers: int = -1, flash_attn: bool = False, cache_type_k: str = "f16", cache_type_v: str = "f16", loras: list[LoraSpec] = [])`
  - `@dataclass Spec(gateway_port: int, llm: list[LlmNode])`
  - `load_spec(data: dict) -> Spec` — raises `SpecError` on invalid input.
  - `class SpecError(ValueError)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_spec.py
import pytest
from ainbox_gateway.spec import load_spec, Spec, LlmNode, LoraSpec, SpecError


def test_minimal_spec():
    spec = load_spec({"gateway": {"port": 8080},
                      "llm": [{"slug": "qwen3.5-2b"}]})
    assert isinstance(spec, Spec)
    assert spec.gateway_port == 8080
    assert spec.llm == [LlmNode(slug="qwen3.5-2b")]


def test_full_node_fields_and_loras():
    spec = load_spec({"gateway": {"port": 9000}, "llm": [{
        "slug": "qwen3.5-9b", "replicas": 2, "n_ctx": 8192,
        "n_gpu_layers": -1, "flash_attn": True,
        "cache_type_k": "q8_0", "cache_type_v": "q8_0",
        "loras": [{"file": "voice.gguf", "alias": "voice", "scale": 0.8}],
    }]})
    node = spec.llm[0]
    assert node.replicas == 2 and node.n_ctx == 8192 and node.flash_attn is True
    assert node.loras == [LoraSpec(file="voice.gguf", alias="voice", scale=0.8)]


def test_missing_gateway_port_raises():
    with pytest.raises(SpecError):
        load_spec({"llm": [{"slug": "x"}]})


def test_node_without_slug_raises():
    with pytest.raises(SpecError):
        load_spec({"gateway": {"port": 8080}, "llm": [{"n_ctx": 4096}]})


def test_empty_llm_raises():
    with pytest.raises(SpecError):
        load_spec({"gateway": {"port": 8080}, "llm": []})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_spec.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ainbox_gateway.spec'`.

- [ ] **Step 3: Write the implementation**

```python
# src/ainbox_gateway/spec.py
"""Raise-spec: which fixed subset of baked models to bring up, and how."""
from __future__ import annotations

from dataclasses import dataclass, field


class SpecError(ValueError):
    """The raise-spec is structurally invalid."""


@dataclass
class LoraSpec:
    file: str
    alias: str
    scale: float = 1.0


@dataclass
class LlmNode:
    slug: str
    replicas: int = 1
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    flash_attn: bool = False
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    loras: list[LoraSpec] = field(default_factory=list)


@dataclass
class Spec:
    gateway_port: int
    llm: list[LlmNode]


def _load_node(raw: dict) -> LlmNode:
    if "slug" not in raw:
        raise SpecError("llm node missing required 'slug'")
    loras = [LoraSpec(**l) for l in raw.get("loras", [])]
    return LlmNode(
        slug=raw["slug"],
        replicas=raw.get("replicas", 1),
        n_ctx=raw.get("n_ctx", 4096),
        n_gpu_layers=raw.get("n_gpu_layers", -1),
        flash_attn=raw.get("flash_attn", False),
        cache_type_k=raw.get("cache_type_k", "f16"),
        cache_type_v=raw.get("cache_type_v", "f16"),
        loras=loras,
    )


def load_spec(data: dict) -> Spec:
    gateway = data.get("gateway")
    if not gateway or "port" not in gateway:
        raise SpecError("spec missing 'gateway.port'")
    raw_llm = data.get("llm") or []
    if not raw_llm:
        raise SpecError("spec must declare at least one 'llm' node")
    return Spec(gateway_port=gateway["port"], llm=[_load_node(n) for n in raw_llm])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_spec.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ainbox_gateway/spec.py tests/test_spec.py
git commit -m "feat(gateway): raise-spec model and loader

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Backend + round-robin Pool

**Files:**
- Create: `src/ainbox_gateway/pool.py`
- Test: `tests/test_pool.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Backend(slug: str, base_url: str)`
  - `class Pool` with `__init__(self, slug: str, backends: list[Backend])` and `next(self) -> Backend` (round-robin, thread-safe); raises `ValueError` if constructed empty.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pool.py
import pytest
from ainbox_gateway.pool import Backend, Pool


def test_single_backend_always_returned():
    b = Backend(slug="m", base_url="http://127.0.0.1:9000")
    pool = Pool(slug="m", backends=[b])
    assert [pool.next() for _ in range(3)] == [b, b, b]


def test_round_robin_cycles_in_order():
    bs = [Backend("m", f"http://127.0.0.1:{p}") for p in (9000, 9001, 9002)]
    pool = Pool(slug="m", backends=bs)
    got = [pool.next() for _ in range(7)]
    assert got == [bs[0], bs[1], bs[2], bs[0], bs[1], bs[2], bs[0]]


def test_empty_pool_rejected():
    with pytest.raises(ValueError):
        Pool(slug="m", backends=[])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pool.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ainbox_gateway.pool'`.

- [ ] **Step 3: Write the implementation**

```python
# src/ainbox_gateway/pool.py
"""A backend endpoint and a round-robin pool of same-slug replicas."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from itertools import cycle


@dataclass(frozen=True)
class Backend:
    slug: str
    base_url: str  # e.g. "http://127.0.0.1:9000"; no trailing slash


class Pool:
    def __init__(self, slug: str, backends: list[Backend]):
        if not backends:
            raise ValueError(f"pool '{slug}' has no backends")
        self.slug = slug
        self._backends = list(backends)
        self._cycle = cycle(self._backends)
        self._lock = threading.Lock()

    def next(self) -> Backend:
        with self._lock:
            return next(self._cycle)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pool.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ainbox_gateway/pool.py tests/test_pool.py
git commit -m "feat(gateway): round-robin backend pool

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Router (slug → pool resolution)

**Files:**
- Create: `src/ainbox_gateway/router.py`
- Test: `tests/test_router.py`

**Interfaces:**
- Consumes: `Backend`, `Pool` (Task 3).
- Produces:
  - `class UnknownModel(KeyError)`
  - `class Router` with `__init__(self, pools: dict[str, Pool])`, `resolve(self, model: str) -> Backend` (raises `UnknownModel`), `models(self) -> list[str]` (sorted slugs).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_router.py
import pytest
from ainbox_gateway.pool import Backend, Pool
from ainbox_gateway.router import Router, UnknownModel


def _router():
    pools = {
        "a": Pool("a", [Backend("a", "http://h:9000"), Backend("a", "http://h:9001")]),
        "b": Pool("b", [Backend("b", "http://h:9002")]),
    }
    return Router(pools)


def test_resolve_round_robins_within_slug():
    r = _router()
    urls = [r.resolve("a").base_url for _ in range(3)]
    assert urls == ["http://h:9000", "http://h:9001", "http://h:9000"]


def test_resolve_unknown_model_raises():
    with pytest.raises(UnknownModel):
        _router().resolve("nope")


def test_models_lists_sorted_slugs():
    assert _router().models() == ["a", "b"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_router.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ainbox_gateway.router'`.

- [ ] **Step 3: Write the implementation**

```python
# src/ainbox_gateway/router.py
"""Resolve an OpenAI `model` slug to a backend via its round-robin pool."""
from __future__ import annotations

from .pool import Backend, Pool


class UnknownModel(KeyError):
    """No pool serves the requested model slug."""


class Router:
    def __init__(self, pools: dict[str, Pool]):
        self._pools = pools

    def resolve(self, model: str) -> Backend:
        pool = self._pools.get(model)
        if pool is None:
            raise UnknownModel(model)
        return pool.next()

    def models(self) -> list[str]:
        return sorted(self._pools)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_router.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ainbox_gateway/router.py tests/test_router.py
git commit -m "feat(gateway): slug router over round-robin pools

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Supervisor argv + port assignment (pure functions)

**Files:**
- Create: `src/ainbox_gateway/supervisor.py`
- Test: `tests/test_supervisor.py`

**Interfaces:**
- Consumes: `LlmNode`, `Spec` (Task 2).
- Produces:
  - `LLAMA_SERVER_BIN: str = "/app/llama-server"`
  - `MODELS_DIR: str = "/models"`
  - `assign_ports(spec: Spec, base: int = 9000) -> list[tuple[LlmNode, int]]` — one `(node, port)` per replica, ports contiguous from `base` in spec order.
  - `llama_argv(node: LlmNode, port: int, bin: str = LLAMA_SERVER_BIN, models_dir: str = MODELS_DIR) -> list[str]` — the `llama-server` command; **never** includes `--embedding`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_supervisor.py
from ainbox_gateway.spec import Spec, LlmNode, LoraSpec
from ainbox_gateway.supervisor import assign_ports, llama_argv


def test_assign_ports_expands_replicas_contiguously():
    spec = Spec(gateway_port=8080, llm=[
        LlmNode(slug="a", replicas=2),
        LlmNode(slug="b", replicas=1),
    ])
    assigned = assign_ports(spec, base=9000)
    assert [(n.slug, p) for n, p in assigned] == [("a", 9000), ("a", 9001), ("b", 9002)]


def test_llama_argv_core_flags():
    argv = llama_argv(LlmNode(slug="qwen3.5-2b", n_ctx=4096, n_gpu_layers=-1), port=9000)
    assert argv[0] == "/app/llama-server"
    assert "-m" in argv and "/models/qwen3.5-2b.gguf" in argv
    assert "--port" in argv and "9000" in argv
    assert "--alias" in argv and "qwen3.5-2b" in argv
    assert argv[argv.index("-c") + 1] == "4096"
    assert argv[argv.index("-ngl") + 1] == "-1"


def test_llama_argv_never_emits_embedding():
    argv = llama_argv(LlmNode(slug="a"), port=9000)
    assert "--embedding" not in argv


def test_llama_argv_flash_attn_and_loras():
    node = LlmNode(slug="a", flash_attn=True,
                   loras=[LoraSpec(file="v.gguf", alias="v", scale=0.8)])
    argv = llama_argv(node, port=9001)
    assert argv[argv.index("--flash-attn") + 1] == "on"
    assert argv[argv.index("--lora-scaled") + 1] == "/loras/v.gguf:0.8"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_supervisor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ainbox_gateway.supervisor'`.

- [ ] **Step 3: Write the implementation** (argv + ports only; process class added in Task 8)

```python
# src/ainbox_gateway/supervisor.py
"""Turn raise-spec nodes into llama-server launch commands + port maps."""
from __future__ import annotations

from .spec import LlmNode, Spec

LLAMA_SERVER_BIN = "/app/llama-server"
MODELS_DIR = "/models"
LORAS_DIR = "/loras"


def assign_ports(spec: Spec, base: int = 9000) -> list[tuple[LlmNode, int]]:
    out: list[tuple[LlmNode, int]] = []
    port = base
    for node in spec.llm:
        for _ in range(node.replicas):
            out.append((node, port))
            port += 1
    return out


def llama_argv(
    node: LlmNode,
    port: int,
    bin: str = LLAMA_SERVER_BIN,
    models_dir: str = MODELS_DIR,
) -> list[str]:
    argv = [
        bin,
        "-m", f"{models_dir}/{node.slug}.gguf",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--alias", node.slug,
        "-c", str(node.n_ctx),
        "-ngl", str(node.n_gpu_layers),
        "--cache-type-k", node.cache_type_k,
        "--cache-type-v", node.cache_type_v,
    ]
    if node.flash_attn:
        argv += ["--flash-attn", "on"]
    if node.loras:
        scaled = ",".join(f"{LORAS_DIR}/{l.file}:{l.scale}" for l in node.loras)
        argv += ["--lora-scaled", scaled]
    return argv
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_supervisor.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ainbox_gateway/supervisor.py tests/test_supervisor.py
git commit -m "feat(gateway): llama-server argv and replica port assignment

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: OpenAI proxy endpoints (`/v1/chat/completions`, `/v1/completions`)

**Files:**
- Create: `src/ainbox_gateway/app.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `Router`, `UnknownModel` (Task 4).
- Produces:
  - `create_app(router: Router, client: httpx.AsyncClient) -> FastAPI` — app with `POST /v1/chat/completions` and `POST /v1/completions`. Reads `model` from the JSON body, resolves a backend, streams the upstream response back verbatim (status + body). Unknown model → 404. Missing `model` → 400.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_app.py
import httpx
import pytest
import respx
from ainbox_gateway.app import create_app
from ainbox_gateway.pool import Backend, Pool
from ainbox_gateway.router import Router


def _app_and_client():
    pools = {"a": Pool("a", [Backend("a", "http://h:9000"),
                             Backend("a", "http://h:9001")])}
    upstream = httpx.AsyncClient()
    return create_app(Router(pools), upstream)


@pytest.mark.asyncio
@respx.mock
async def test_chat_completion_routes_and_round_robins():
    route0 = respx.post("http://h:9000/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"id": "0"}))
    route1 = respx.post("http://h:9001/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"id": "1"}))
    app = _app_and_client()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
        r0 = await c.post("/v1/chat/completions", json={"model": "a", "messages": []})
        r1 = await c.post("/v1/chat/completions", json={"model": "a", "messages": []})
    assert r0.json()["id"] == "0" and r1.json()["id"] == "1"
    assert route0.called and route1.called


@pytest.mark.asyncio
@respx.mock
async def test_unknown_model_returns_404():
    app = _app_and_client()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
        r = await c.post("/v1/chat/completions", json={"model": "nope", "messages": []})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_missing_model_returns_400():
    app = _app_and_client()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
        r = await c.post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 400
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_app.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ainbox_gateway.app'`.

- [ ] **Step 3: Write the implementation**

```python
# src/ainbox_gateway/app.py
"""FastAPI gateway: one pure-OpenAI front door over routed backend pools."""
from __future__ import annotations

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .router import Router, UnknownModel

_OPENAI_PATHS = ("/v1/chat/completions", "/v1/completions")


def create_app(router: Router, client: httpx.AsyncClient) -> FastAPI:
    app = FastAPI(title="ainbox-infrastructure gateway")

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
            backend = router.resolve(model)
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
            background=_closer(resp),
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await _proxy(request, "/v1/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await _proxy(request, "/v1/completions")

    app.state.router = router
    app.state.client = client
    return app


def _closer(resp: httpx.Response):
    from starlette.background import BackgroundTask
    return BackgroundTask(resp.aclose)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_app.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ainbox_gateway/app.py tests/test_app.py
git commit -m "feat(gateway): OpenAI chat/completions proxy with round-robin

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: `GET /v1/models` aggregation

**Files:**
- Modify: `src/ainbox_gateway/app.py`
- Test: `tests/test_app.py` (add cases)

**Interfaces:**
- Produces: `GET /v1/models` → OpenAI-shaped `{"object": "list", "data": [{"id": slug, "object": "model", "owned_by": "ainbox"}, ...]}` from `router.models()`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_app.py`)

```python
@pytest.mark.asyncio
async def test_models_endpoint_lists_slugs():
    app = _app_and_client()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
        r = await c.get("/v1/models")
    body = r.json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["a"]
    assert body["data"][0]["object"] == "model"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_app.py::test_models_endpoint_lists_slugs -v`
Expected: FAIL — 404 (route not defined yet).

- [ ] **Step 3: Add the endpoint** (inside `create_app`, before `return app`)

```python
    @app.get("/v1/models")
    async def list_models() -> Response:
        data = [{"id": s, "object": "model", "owned_by": "ainbox"}
                for s in router.models()]
        return JSONResponse({"object": "list", "data": data})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_app.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ainbox_gateway/app.py tests/test_app.py
git commit -m "feat(gateway): GET /v1/models aggregation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: Supervisor lifecycle + startup wiring

**Files:**
- Modify: `src/ainbox_gateway/supervisor.py`
- Modify: `src/ainbox_gateway/app.py`
- Test: `tests/test_supervisor.py`, `tests/test_app.py` (add cases)

**Interfaces:**
- Consumes: `assign_ports`, `llama_argv` (Task 5), `Backend`, `Pool`, `Router`.
- Produces:
  - `class Supervisor(Protocol)` with `start(self, spec: Spec) -> dict[str, Pool]` and `stop(self) -> None`.
  - `build_pools(spec: Spec, base: int = 9000) -> dict[str, Pool]` — pure: groups `assign_ports` output into one `Pool` per slug with `Backend(slug, "http://127.0.0.1:<port>")`. (Used by real + fake supervisors and directly testable.)
  - `create_app` gains lifespan: on startup call `supervisor.start(spec)` → build `Router`; on shutdown call `supervisor.stop()` and close the client. New signature: `create_app(spec: Spec, supervisor: Supervisor, client: httpx.AsyncClient | None = None) -> FastAPI`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_supervisor.py  (append)
from ainbox_gateway.supervisor import build_pools


def test_build_pools_groups_replicas_by_slug():
    spec = Spec(gateway_port=8080, llm=[
        LlmNode(slug="a", replicas=2), LlmNode(slug="b", replicas=1)])
    pools = build_pools(spec, base=9000)
    assert set(pools) == {"a", "b"}
    assert [b.base_url for b in pools["a"]._backends] == [
        "http://127.0.0.1:9000", "http://127.0.0.1:9001"]
    assert [b.base_url for b in pools["b"]._backends] == ["http://127.0.0.1:9002"]
```

```python
# tests/test_app.py  (append)
from ainbox_gateway.spec import Spec, LlmNode
from ainbox_gateway.supervisor import build_pools


class FakeSupervisor:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self, spec):
        self.started = True
        return build_pools(spec, base=9000)

    def stop(self):
        self.stopped = True


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_supervisor():
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a", replicas=2)])
    sup = FakeSupervisor()
    app = create_app(spec, sup)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
        r = await c.get("/v1/models")
        assert [m["id"] for m in r.json()["data"]] == ["a"]
    assert sup.started and sup.stopped
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_supervisor.py::test_build_pools_groups_replicas_by_slug tests/test_app.py::test_lifespan_starts_and_stops_supervisor -v`
Expected: FAIL — `ImportError: cannot import name 'build_pools'` and `create_app` signature mismatch.

- [ ] **Step 3: Add `build_pools` + `Supervisor` protocol to `supervisor.py`**

```python
# src/ainbox_gateway/supervisor.py  (append)
from typing import Protocol

from .pool import Backend, Pool


def build_pools(spec: Spec, base: int = 9000) -> dict[str, Pool]:
    by_slug: dict[str, list[Backend]] = {}
    for node, port in assign_ports(spec, base=base):
        by_slug.setdefault(node.slug, []).append(
            Backend(slug=node.slug, base_url=f"http://127.0.0.1:{port}"))
    return {slug: Pool(slug, backends) for slug, backends in by_slug.items()}


class Supervisor(Protocol):
    def start(self, spec: Spec) -> dict[str, Pool]: ...
    def stop(self) -> None: ...
```

- [ ] **Step 4: Rewrite `create_app` to take `(spec, supervisor, client)` with lifespan**

Replace the `create_app` signature and add a lifespan; keep `_proxy`, the three routes, and `/v1/models` bodies unchanged except that they read `app.state.router` (set at startup).

```python
# src/ainbox_gateway/app.py  (replace the create_app definition head + add lifespan)
from contextlib import asynccontextmanager

from .spec import Spec
from .supervisor import Supervisor


def create_app(spec: Spec, supervisor: Supervisor,
               client: httpx.AsyncClient | None = None) -> FastAPI:
    client = client or httpx.AsyncClient(timeout=None)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pools = supervisor.start(spec)
        app.state.router = Router(pools)
        yield
        supervisor.stop()
        await client.aclose()

    app = FastAPI(title="ainbox-infrastructure gateway", lifespan=lifespan)

    def _router() -> Router:
        return app.state.router
    # ... _proxy / routes / /v1/models use _router() instead of the closed-over router
```

Update `_proxy` and `/v1/models` to call `_router().resolve(...)` / `_router().models()`. Update the Task 6 tests' `_app_and_client()` helper to the new signature:

```python
# tests/test_app.py  (update helper)
def _app_and_client():
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a", replicas=2)])
    return create_app(spec, FakeSupervisor())
```

(The respx mocks in Task 6 already target `http://127.0.0.1:9000/...` and `:9001/...` — update those two `respx.post` URLs from `http://h:900x` to `http://127.0.0.1:900x` to match `build_pools`.)

- [ ] **Step 5: Run the whole suite to verify it passes**

Run: `pytest -v`
Expected: PASS (all tasks 1–8 green).

- [ ] **Step 6: Commit**

```bash
git add src/ainbox_gateway/ tests/
git commit -m "feat(gateway): supervisor protocol, build_pools, lifespan wiring

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: Real `LlamaSupervisor` + entrypoint/Docker swap + integration smoke

**Files:**
- Modify: `src/ainbox_gateway/supervisor.py` (add `LlamaSupervisor`)
- Create: `src/ainbox_gateway/__main__.py` (module entrypoint)
- Modify: `build/Dockerfile` (COPY gateway package + install; CMD runs the gateway)
- Modify: `build/entrypoint.sh` (replace the llama launch-loop with `python -m ainbox_gateway`); keep the hardware/GPU check preamble
- Test: `tests/test_supervisor.py` (unit: launch builds the right argv via a fake spawner) + `docs/` smoke note

**Interfaces:**
- Consumes: `assign_ports`, `llama_argv`, `build_pools`.
- Produces:
  - `LlamaSupervisor(spawn=subprocess.Popen, wait_ready: Callable[[str], None] = _http_ready)` implementing `Supervisor`: on `start`, spawns one process per `(node, port)` via `llama_argv`, waits for each `/v1/models` to answer, returns `build_pools(spec)`; on `stop`, terminates all processes.
  - `python -m ainbox_gateway` reads the raise-spec path from `AINBOX_SPEC` (default `/app/config/superbot_config.json`), builds the app with `LlamaSupervisor`, and serves uvicorn on `spec.gateway_port`.

- [ ] **Step 1: Write the failing unit test (spawn is injected, not real)**

```python
# tests/test_supervisor.py  (append)
from ainbox_gateway.supervisor import LlamaSupervisor


def test_llama_supervisor_spawns_argv_per_replica():
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a", replicas=2)])
    calls = []

    class FakeProc:
        def __init__(self, argv): self.argv = argv
        def terminate(self): calls.append(("term", self.argv[self.argv.index("--port") + 1]))
        def wait(self, timeout=None): pass

    def fake_spawn(argv, **kw):
        calls.append(("spawn", argv[argv.index("--port") + 1]))
        return FakeProc(argv)

    sup = LlamaSupervisor(spawn=fake_spawn, wait_ready=lambda url: None)
    pools = sup.start(spec)
    assert set(pools) == {"a"}
    assert [c for c in calls if c[0] == "spawn"] == [("spawn", "9000"), ("spawn", "9001")]
    sup.stop()
    assert ("term", "9000") in calls and ("term", "9001") in calls
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_supervisor.py::test_llama_supervisor_spawns_argv_per_replica -v`
Expected: FAIL — `ImportError: cannot import name 'LlamaSupervisor'`.

- [ ] **Step 3: Implement `LlamaSupervisor` + `__main__`**

```python
# src/ainbox_gateway/supervisor.py  (append)
import subprocess
import time
import urllib.request
from typing import Callable


def _http_ready(base_url: str, retries: int = 60, delay: float = 1.0) -> None:
    for _ in range(retries):
        try:
            urllib.request.urlopen(f"{base_url}/v1/models", timeout=2)
            return
        except Exception:
            time.sleep(delay)
    raise RuntimeError(f"backend at {base_url} never became ready")


class LlamaSupervisor:
    def __init__(self, spawn: Callable = subprocess.Popen,
                 wait_ready: Callable[[str], None] = _http_ready):
        self._spawn = spawn
        self._wait_ready = wait_ready
        self._procs: list = []

    def start(self, spec: Spec) -> dict[str, Pool]:
        for node, port in assign_ports(spec):
            self._procs.append(self._spawn(llama_argv(node, port)))
            self._wait_ready(f"http://127.0.0.1:{port}")
        return build_pools(spec)

    def stop(self) -> None:
        for p in self._procs:
            p.terminate()
        for p in self._procs:
            p.wait(timeout=10)
        self._procs = []
```

```python
# src/ainbox_gateway/__main__.py
import json
import os

import uvicorn

from .app import create_app
from .spec import load_spec
from .supervisor import LlamaSupervisor

_SPEC_PATH = os.environ.get("AINBOX_SPEC", "/app/config/superbot_config.json")


def main() -> None:
    with open(_SPEC_PATH, encoding="utf-8") as f:
        spec = load_spec(json.load(f))
    app = create_app(spec, LlamaSupervisor())
    uvicorn.run(app, host="0.0.0.0", port=spec.gateway_port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the unit test + full suite**

Run: `pytest -v`
Expected: PASS (all green).

- [ ] **Step 5: Swap the Docker entrypoint to the gateway**

Edit `build/entrypoint.sh`: keep the hardware/GPU-detection preamble (lines through the `nvidia-smi` check + CPU stub symlink), then **replace the entire `jq … llama_node[] … llama-server …` launch loop and the final `exec uvicorn whisper_api` with**:

```bash
echo "[SYSTEM] Launching ainbox gateway..."
exec python3 -m ainbox_gateway
```

Edit `build/Dockerfile`: after the existing Python deps step, add the gateway package and install it, before `COPY entrypoint.sh`:

```dockerfile
COPY pyproject.toml /app/gateway/pyproject.toml
COPY src /app/gateway/src
RUN uv pip install --system /app/gateway
```

(The `pyproject.toml` + `src/` must be added to the build context — update `build.sh`'s context copy, or move the Docker `COPY` sources so the gateway package is reachable from the build dir. The plan's smoke step verifies the context is correct.)

- [ ] **Step 6: Document the integration smoke** (create `docs/smoke-gateway.md`)

```markdown
# Gateway integration smoke (manual, needs a GPU host + one baked GGUF)

1. Build an image whose recipe bakes a tiny model, e.g. `qwen3.5-2b`.
2. Raise-spec `deploy/smoke.json`:
   {"gateway": {"port": 8080}, "llm": [{"slug": "qwen3.5-2b", "replicas": 2, "n_gpu_layers": -1}]}
3. `make run CONFIG=deploy/smoke.json TAG=<tag> MODE=gpu`
4. `curl localhost:8080/v1/models` → lists `qwen3.5-2b`.
5. `curl localhost:8080/v1/chat/completions -d '{"model":"qwen3.5-2b","messages":[{"role":"user","content":"hi"}]}'`
   → an OpenAI chat completion; run twice and confirm both replica logs show traffic (round-robin).
```

- [ ] **Step 7: Commit**

```bash
git add src/ainbox_gateway/ tests/ build/Dockerfile build/entrypoint.sh docs/smoke-gateway.md
git commit -m "feat(gateway): LlamaSupervisor + Docker entrypoint swap

Replaces the entrypoint.sh llama launch-loop with 'python -m ainbox_gateway'.
Integration smoke documented in docs/smoke-gateway.md.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Follow-on plans (out of scope here)

Each is its own spec-phase → plan, and each produces working software:

1. **Embeddings backend** — fastembed-GPU MiniLM behind `/v1/embeddings`; equivalence test vs stored vectors; drop e5 from the recipe; remove unconditional `--embedding` (already done here).
2. **STT backend** — standard `/v1/audio/transcriptions` (model in body) behind the gateway; refactor `whisper_api.py` to OpenAI shape.
3. **Tiny UI** — edit the raise-spec + relaunch; status view of raised pools.
4. **warden integration** (warden repo) — repoint `WARDEN_LLM_BASE_URL`; make `/api/transcribe` + `/api/embed` thin proxies to the engine.
5. **Deployment swap** (ainbox + ainbox-os repos) — replace the `ollama` service; then the ainbox-os engine layer (desktop + server modes).
6. **(Later)** TTS `/v1/audio/speech`; FLUX `/v1/images/generations`.

## Self-Review

- **Spec coverage (this plan's scope = engine core / LLM):** raise-spec ✓ (T2), fixed residency + round-robin replicas ✓ (T3–T5, T8), pure-OpenAI front door ✓ (T6–T7), single public port / hidden internal ports ✓ (T5 `assign_ports`, T8 `build_pools`), no `--embedding` on chat ✓ (T5), supervisor + relaunch teardown ✓ (T8–T9), entrypoint/Docker swap ✓ (T9). Out-of-scope spec items are enumerated under Follow-on plans.
- **Placeholder scan:** every code step contains complete code; no TBD/TODO. The one manual artifact (integration smoke) is a documented runbook, not a code placeholder.
- **Type consistency:** `Backend`, `Pool`, `Router`, `UnknownModel`, `Spec`, `LlmNode`, `LoraSpec`, `assign_ports`, `llama_argv`, `build_pools`, `Supervisor`, `LlamaSupervisor`, `create_app` names/signatures are consistent across tasks. Task 8 explicitly restates the `create_app` signature change and the Task 6 test-helper/respx-URL updates it forces, so out-of-order readers stay consistent.
