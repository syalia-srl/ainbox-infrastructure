# ainbox-builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone FastAPI app that authors an inference recipe (pick models), triggers a Docker build with one button, streams progress live, and pushes the image to a registry.

**Architecture:** New `ainbox_builder` package under `src/`, sibling to `ainbox_gateway`. Pure functions render the recipe and assemble the build steps; an async `BuildRunner` shells out to `make image` + `docker push`, buffering output for an SSE log endpoint. A single static page (mirroring `ainbox_gateway/static/ui.html`) drives the endpoints and reuses the shared `syalia-ui` assets from the gateway package.

**Tech Stack:** FastAPI + uvicorn (already deps), vanilla JS, `asyncio.create_subprocess_exec`. No new dependencies.

## Global Constraints

- **No new dependencies** — FastAPI/uvicorn/httpx already in `pyproject.toml`.
- **One build at a time** — a second `POST /api/build` while a build is non-terminal → HTTP 409.
- **Injectable subprocess spawn** — `BuildRunner`/`create_app` accept a `spawn` callable (default `asyncio.create_subprocess_exec`) so tests never touch Docker.
- **Registry image path:** `<registry>/ainbox-infra/<name>:latest`; default registry `registry.syalia.dev`.
- **CUDA bases:** Ada = `12.2.2-devel-ubuntu22.04`, Blackwell = `12.8.1-devel-ubuntu22.04` (passed as `CUDA_TAG` env to `make image`).
- **Recipe schema** (consumed by `build/Dockerfile`): keys `whisper_nodes`, `embedding_nodes`, `tts_nodes`, `image_nodes`, `llama_node`; each `llama_node` entry `{"url","alias"}`; each `whisper_nodes` entry `{"model","alias"}`; each `embedding_nodes` entry `{"model"}`.
- **Reject empty LLM set** — a recipe with no `llama_node` entries is invalid (mirrors `ainbox_gateway.spec.load_spec`'s "at least one llm" rule, caught at author time).
- **Commit style:** conventional commits; end body with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Test client pattern** (reuse verbatim from `tests/test_app.py`):
  ```python
  from contextlib import asynccontextmanager
  import httpx
  from asgi_lifespan import LifespanManager

  @asynccontextmanager
  async def _client(app):
      async with LifespanManager(app):
          transport = httpx.ASGITransport(app=app)
          async with httpx.AsyncClient(transport=transport, base_url="http://b") as c:
              yield c
  ```

---

## File Structure

- `src/ainbox_builder/__init__.py` — package marker.
- `src/ainbox_builder/catalog.py` — `CATALOG` dict (llm/stt/embeddings entries with real GGUF URLs).
- `src/ainbox_builder/recipe.py` — `render_recipe(selection)` pure function + `RecipeError`.
- `src/ainbox_builder/builder.py` — `Step`, `build_command(...)`, `LogBuffer`, `BuildRunner`, `BuildManager`.
- `src/ainbox_builder/app.py` — `create_app(repo_root, catalog=CATALOG, spawn=None)` + routes.
- `src/ainbox_builder/static/build.html` — the page.
- `scripts/build_ui.py` — uvicorn launcher.
- `tests/test_builder_recipe.py`, `tests/test_builder_command.py`, `tests/test_builder_runner.py`, `tests/test_builder_app.py`.
- Modify `Makefile` (add `ui-build`), `pyproject.toml` (package-data for `ainbox_builder`).

---

### Task 1: Catalog + recipe rendering (pure)

**Files:**
- Create: `src/ainbox_builder/__init__.py` (empty), `src/ainbox_builder/catalog.py`, `src/ainbox_builder/recipe.py`
- Test: `tests/test_builder_recipe.py`

**Interfaces:**
- Produces: `render_recipe(selection: dict) -> dict`; `RecipeError(ValueError)`; `CATALOG: dict`.
  - `selection` shape: `{"llm":[{"alias","url"}], "stt":[{"alias","model"}], "embeddings":[{"model"}]}` (any list may be omitted/empty).
  - Returns the recipe dict with all five keys (`tts_nodes`/`image_nodes` always `[]`).

- [ ] **Step 1: Write the failing test** — create `tests/test_builder_recipe.py`

```python
import pytest
from ainbox_builder.recipe import render_recipe, RecipeError


def test_render_recipe_full():
    sel = {
        "llm": [{"alias": "gemma4-e4b", "url": "https://hf/gemma.gguf"},
                {"alias": "qwen3-14b", "url": "https://hf/qwen.gguf"}],
        "stt": [{"alias": "fast_stt", "model": "tiny"}],
        "embeddings": [{"model": "paraphrase-multilingual-MiniLM-L12-v2"}],
    }
    assert render_recipe(sel) == {
        "whisper_nodes": [{"model": "tiny", "alias": "fast_stt"}],
        "embedding_nodes": [{"model": "paraphrase-multilingual-MiniLM-L12-v2"}],
        "tts_nodes": [],
        "image_nodes": [],
        "llama_node": [
            {"url": "https://hf/gemma.gguf", "alias": "gemma4-e4b"},
            {"url": "https://hf/qwen.gguf", "alias": "qwen3-14b"},
        ],
    }


def test_render_recipe_llm_only():
    sel = {"llm": [{"alias": "a", "url": "https://hf/a.gguf"}]}
    out = render_recipe(sel)
    assert out["llama_node"] == [{"url": "https://hf/a.gguf", "alias": "a"}]
    assert out["whisper_nodes"] == [] and out["embedding_nodes"] == []


def test_render_recipe_rejects_empty_llm():
    with pytest.raises(RecipeError):
        render_recipe({"stt": [{"alias": "x", "model": "tiny"}]})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_builder_recipe.py -v`
Expected: FAIL — `ModuleNotFoundError: ainbox_builder`.

- [ ] **Step 3: Write minimal implementation** — `src/ainbox_builder/__init__.py` empty; `src/ainbox_builder/recipe.py`:

```python
"""Pure: turn a UI model selection into a build recipe dict."""
from __future__ import annotations


class RecipeError(ValueError):
    """The selection cannot form a valid recipe."""


def render_recipe(selection: dict) -> dict:
    llm = selection.get("llm") or []
    if not llm:
        raise RecipeError("a recipe needs at least one LLM")
    return {
        "whisper_nodes": [{"model": n["model"], "alias": n["alias"]}
                          for n in selection.get("stt") or []],
        "embedding_nodes": [{"model": n["model"]}
                            for n in selection.get("embeddings") or []],
        "tts_nodes": [],
        "image_nodes": [],
        "llama_node": [{"url": n["url"], "alias": n["alias"]} for n in llm],
    }
```

- [ ] **Step 4: Create the catalog** — `src/ainbox_builder/catalog.py`:

```python
"""Curated model catalog for the builder UI. URLs are HF GGUF resolve links."""

_HF = "https://huggingface.co"

CATALOG = {
    "llm": {
        "gemma4-e4b":   {"url": f"{_HF}/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf?download=true", "size": "5.0 GB"},
        "gemma4-e2b":   {"url": f"{_HF}/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf?download=true", "size": "~3 GB"},
        "qwen3-14b":    {"url": f"{_HF}/unsloth/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q4_K_M.gguf?download=true", "size": "9.0 GB"},
        "qwen3.5-9b":   {"url": f"{_HF}/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf?download=true", "size": "~5.5 GB"},
        "qwen3.5-4b":   {"url": f"{_HF}/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf?download=true", "size": "~2.5 GB"},
        "qwen3.5-2b":   {"url": f"{_HF}/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf?download=true", "size": "~1.5 GB"},
        "qwen3.5-0.8b": {"url": f"{_HF}/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf?download=true", "size": "~0.6 GB"},
    },
    "stt": {
        "whisper-tiny":  {"model": "tiny"},
        "whisper-small": {"model": "small"},
    },
    "embeddings": {
        "minilm": {"model": "paraphrase-multilingual-MiniLM-L12-v2"},
    },
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_builder_recipe.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add src/ainbox_builder/__init__.py src/ainbox_builder/recipe.py src/ainbox_builder/catalog.py tests/test_builder_recipe.py
git commit -m "feat(builder): catalog + pure recipe rendering"
```

---

### Task 2: Build command assembly (pure)

**Files:**
- Modify: `src/ainbox_builder/builder.py` (create)
- Test: `tests/test_builder_command.py`

**Interfaces:**
- Produces: `Step` (dataclass `label: str`, `argv: list[str]`, `env: dict[str, str] | None = None`); `build_command(name: str, cuda_tag: str, registry: str, push: bool) -> list[Step]`.

- [ ] **Step 1: Write the failing test** — `tests/test_builder_command.py`

```python
from ainbox_builder.builder import build_command, Step


def test_build_command_no_push():
    steps = build_command("smaug_v1", "12.8.1-devel-ubuntu22.04", "registry.syalia.dev", push=False)
    assert steps == [
        Step(label="build",
             argv=["make", "image", "RECIPE=recipes/smaug_v1.json"],
             env={"CUDA_TAG": "12.8.1-devel-ubuntu22.04"}),
    ]


def test_build_command_with_push():
    steps = build_command("smaug_v1", "12.2.2-devel-ubuntu22.04", "registry.syalia.dev", push=True)
    labels = [s.label for s in steps]
    assert labels == ["build", "tag", "push"]
    assert steps[1].argv == ["docker", "tag", "superbot:smaug_v1",
                             "registry.syalia.dev/ainbox-infra/smaug_v1:latest"]
    assert steps[2].argv == ["docker", "push",
                             "registry.syalia.dev/ainbox-infra/smaug_v1:latest"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_builder_command.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_command'`.

- [ ] **Step 3: Write minimal implementation** — start `src/ainbox_builder/builder.py`:

```python
"""Assemble + run the build/push shell steps, buffering output for SSE."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Step:
    label: str
    argv: list[str]
    env: dict[str, str] | None = None


def build_command(name: str, cuda_tag: str, registry: str, push: bool) -> list[Step]:
    steps = [Step(label="build",
                  argv=["make", "image", f"RECIPE=recipes/{name}.json"],
                  env={"CUDA_TAG": cuda_tag})]
    if push:
        ref = f"{registry}/ainbox-infra/{name}:latest"
        steps.append(Step(label="tag", argv=["docker", "tag", f"superbot:{name}", ref]))
        steps.append(Step(label="push", argv=["docker", "push", ref]))
    return steps
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_builder_command.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ainbox_builder/builder.py tests/test_builder_command.py
git commit -m "feat(builder): pure build/tag/push command assembly"
```

---

### Task 3: Log buffer + build runner + manager (async, injectable spawn)

**Files:**
- Modify: `src/ainbox_builder/builder.py`
- Test: `tests/test_builder_runner.py`

**Interfaces:**
- Consumes: `Step`, `build_command` (Task 2).
- Produces:
  - `LogBuffer`: `.append(line: str)`, `.close()`, `async .stream()` (async generator yielding retained then live lines until closed).
  - `BuildRunner(steps, cwd, spawn, log: LogBuffer)`, `async .run()` → sets `.status` in `{"building","done","failed"}` and `.exit_code: int`. Runs steps sequentially; stops at first non-zero exit.
  - `BuildManager(cwd, spawn)`: `.start(steps) -> str` (build_id) raising `BuildBusy` if a build is non-terminal; `.get(build_id) -> BuildRunner | None`; `class BuildBusy(RuntimeError)`.
  - `spawn` signature: `async spawn(argv, env, cwd) -> proc` where `proc.stdout` yields bytes lines and `proc.wait()` returns an int returncode. Default wraps `asyncio.create_subprocess_exec`.

- [ ] **Step 1: Write the failing tests** — `tests/test_builder_runner.py`

```python
import asyncio
import pytest
from ainbox_builder.builder import (
    Step, BuildRunner, BuildManager, BuildBusy, LogBuffer)


class _FakeProc:
    def __init__(self, lines, code):
        self._lines = [l.encode() + b"\n" for l in lines]
        self._code = code
        self.stdout = self
    def __aiter__(self):
        async def gen():
            for l in self._lines:
                yield l
        return gen()
    async def wait(self):
        return self._code


def _fake_spawn(script):
    """script: dict label -> (lines, code). Matches on argv[0]+label heuristic."""
    async def spawn(argv, env, cwd):
        # label is encoded by caller via argv; tests pass single-step runs
        key = argv[0]
        lines, code = script[key]
        return _FakeProc(lines, code)
    return spawn


@pytest.mark.asyncio
async def test_runner_streams_and_succeeds():
    log = LogBuffer()
    steps = [Step("build", ["make", "image", "RECIPE=recipes/x.json"])]
    runner = BuildRunner(steps, cwd=".", spawn=_fake_spawn({"make": (["a", "b"], 0)}), log=log)
    got = []
    async def collect():
        async for line in log.stream():
            got.append(line)
    task = asyncio.create_task(collect())
    await runner.run()
    await task
    assert runner.status == "done" and runner.exit_code == 0
    assert [g for g in got if g in ("a", "b")] == ["a", "b"]


@pytest.mark.asyncio
async def test_runner_stops_on_failure():
    log = LogBuffer()
    steps = [Step("build", ["make"]), Step("push", ["docker"])]
    runner = BuildRunner(steps, cwd=".",
                         spawn=_fake_spawn({"make": (["boom"], 2), "docker": (["nope"], 0)}),
                         log=log)
    await runner.run()
    assert runner.status == "failed" and runner.exit_code == 2


@pytest.mark.asyncio
async def test_manager_rejects_concurrent():
    mgr = BuildManager(cwd=".", spawn=_fake_spawn({"make": (["x"], 0)}))
    # a runner that blocks so the first build stays non-terminal
    slow = asyncio.Event()
    async def blocking_spawn(argv, env, cwd):
        class P:
            stdout = _FakeProc([], 0)
            async def wait(self_):
                await slow.wait()
                return 0
        return P()
    mgr._spawn = blocking_spawn
    bid = mgr.start([Step("build", ["make"])])
    with pytest.raises(BuildBusy):
        mgr.start([Step("build", ["make"])])
    slow.set()
    await mgr.get(bid).task
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_builder_runner.py -v`
Expected: FAIL — `ImportError` for `BuildRunner`/`BuildManager`/`LogBuffer`/`BuildBusy`.

- [ ] **Step 3: Write minimal implementation** — append to `src/ainbox_builder/builder.py`:

```python
import asyncio
import os


class LogBuffer:
    def __init__(self):
        self._lines: list[str] = []
        self._event = asyncio.Event()
        self._closed = False

    def append(self, line: str) -> None:
        self._lines.append(line)
        self._event.set()

    def close(self) -> None:
        self._closed = True
        self._event.set()

    async def stream(self):
        i = 0
        while True:
            while i < len(self._lines):
                yield self._lines[i]
                i += 1
            if self._closed:
                return
            self._event.clear()
            await self._event.wait()


async def _default_spawn(argv, env, cwd):
    return await asyncio.create_subprocess_exec(
        *argv, env={**os.environ, **(env or {})}, cwd=cwd,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)


class BuildRunner:
    def __init__(self, steps, cwd, spawn, log: LogBuffer):
        self.steps, self.cwd, self._spawn, self.log = steps, cwd, spawn, log
        self.status = "building"
        self.exit_code = 0
        self.task: asyncio.Task | None = None

    async def run(self):
        try:
            for step in self.steps:
                self.log.append(f"$ [{step.label}] {' '.join(step.argv)}")
                proc = await self._spawn(step.argv, step.env, self.cwd)
                async for raw in proc.stdout:
                    self.log.append(raw.decode(errors="replace").rstrip("\n"))
                code = await proc.wait()
                if code != 0:
                    self.status, self.exit_code = "failed", code
                    self.log.append(f"[{step.label}] exited {code}")
                    return
            self.status, self.exit_code = "done", 0
        finally:
            self.log.close()


class BuildBusy(RuntimeError):
    pass


class BuildManager:
    def __init__(self, cwd, spawn=None):
        self.cwd = cwd
        self._spawn = spawn or _default_spawn
        self._runners: dict[str, BuildRunner] = {}
        self._n = 0

    def _current(self) -> BuildRunner | None:
        for r in self._runners.values():
            if r.status == "building":
                return r
        return None

    def start(self, steps) -> str:
        if self._current():
            raise BuildBusy("a build is already running")
        self._n += 1
        bid = f"b{self._n}"
        runner = BuildRunner(steps, self.cwd, self._spawn, LogBuffer())
        runner.task = asyncio.create_task(runner.run())
        self._runners[bid] = runner
        return bid

    def get(self, bid) -> BuildRunner | None:
        return self._runners.get(bid)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_builder_runner.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ainbox_builder/builder.py tests/test_builder_runner.py
git commit -m "feat(builder): async build runner + log buffer + one-at-a-time manager"
```

---

### Task 4: FastAPI app + endpoints

**Files:**
- Create: `src/ainbox_builder/app.py`
- Test: `tests/test_builder_app.py`

**Interfaces:**
- Consumes: `CATALOG`, `render_recipe`/`RecipeError`, `build_command`, `BuildManager`/`BuildBusy`, `Step`.
- Produces: `create_app(repo_root: str, catalog: dict = CATALOG, spawn=None) -> FastAPI`.
  - `GET /api/catalog` → `catalog`.
  - `POST /api/recipe` body `{selection}` → 200 recipe dict, or 400 `{error}`.
  - `POST /api/build` body `{name, cuda_tag, registry, push, selection}` → writes `recipes/<name>.json` under `repo_root`, `{build_id}`; 400 invalid recipe; 409 `{error}` if busy.
  - `GET /api/build/{id}` → `{status, exit_code}`; 404 unknown.
  - `GET /api/build/{id}/log` → SSE (`text/event-stream`) of `data: <line>\n\n`; 404 unknown.
  - `GET /` → `build.html`; `/syalia-ui/*` mounted from the gateway package's static dir.

- [ ] **Step 1: Write the failing tests** — `tests/test_builder_app.py`

```python
import json
from contextlib import asynccontextmanager
import httpx
import pytest
from asgi_lifespan import LifespanManager
from ainbox_builder.app import create_app
from ainbox_builder.builder import Step


class _FakeProc:
    def __init__(self, lines, code):
        self._lines = [l.encode() + b"\n" for l in lines]
        self._code = code
        self.stdout = self
    def __aiter__(self):
        async def gen():
            for l in self._lines:
                yield l
        return gen()
    async def wait(self):
        return self._code


async def _ok_spawn(argv, env, cwd):
    return _FakeProc(["step ok"], 0)


@asynccontextmanager
async def _client(app):
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://b") as c:
            yield c


def _app(tmp_path):
    return create_app(repo_root=str(tmp_path), spawn=_ok_spawn)


@pytest.mark.asyncio
async def test_catalog(tmp_path):
    async with _client(_app(tmp_path)) as c:
        r = await c.get("/api/catalog")
        assert r.status_code == 200 and "llm" in r.json()


@pytest.mark.asyncio
async def test_recipe_valid_and_invalid(tmp_path):
    async with _client(_app(tmp_path)) as c:
        ok = await c.post("/api/recipe", json={"selection":
            {"llm": [{"alias": "a", "url": "https://hf/a.gguf"}]}})
        assert ok.status_code == 200 and ok.json()["llama_node"][0]["alias"] == "a"
        bad = await c.post("/api/recipe", json={"selection": {"stt": []}})
        assert bad.status_code == 400


@pytest.mark.asyncio
async def test_build_writes_recipe_and_runs(tmp_path):
    (tmp_path / "recipes").mkdir()
    async with _client(_app(tmp_path)) as c:
        r = await c.post("/api/build", json={
            "name": "t1", "cuda_tag": "12.8.1-devel-ubuntu22.04",
            "registry": "registry.syalia.dev", "push": False,
            "selection": {"llm": [{"alias": "a", "url": "https://hf/a.gguf"}]}})
        assert r.status_code == 200
        bid = r.json()["build_id"]
        # recipe file written
        written = json.loads((tmp_path / "recipes" / "t1.json").read_text())
        assert written["llama_node"][0]["url"] == "https://hf/a.gguf"
        # drain the SSE log
        body = (await c.get(f"/api/build/{bid}/log")).text
        assert "step ok" in body
        st = (await c.get(f"/api/build/{bid}")).json()
        assert st["status"] == "done"


@pytest.mark.asyncio
async def test_build_invalid_recipe_400(tmp_path):
    async with _client(_app(tmp_path)) as c:
        r = await c.post("/api/build", json={
            "name": "t2", "cuda_tag": "x", "registry": "r", "push": False,
            "selection": {"stt": []}})
        assert r.status_code == 400
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_builder_app.py -v`
Expected: FAIL — `ModuleNotFoundError: ainbox_builder.app`.

- [ ] **Step 3: Write minimal implementation** — `src/ainbox_builder/app.py`:

```python
"""FastAPI app: author a recipe, trigger a build, stream progress, push."""
from __future__ import annotations

import json
from pathlib import Path

import ainbox_gateway
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .builder import BuildBusy, BuildManager, build_command
from .catalog import CATALOG
from .recipe import RecipeError, render_recipe

_STATIC = Path(__file__).parent / "static"
_GATEWAY_STATIC = Path(ainbox_gateway.__file__).parent / "static" / "syalia-ui"


def create_app(repo_root: str, catalog: dict = CATALOG, spawn=None) -> FastAPI:
    app = FastAPI(title="ainbox-builder")
    manager = BuildManager(cwd=repo_root, spawn=spawn)

    @app.get("/api/catalog")
    async def get_catalog():
        return JSONResponse(catalog)

    @app.post("/api/recipe")
    async def post_recipe(request: Request):
        body = await request.json()
        try:
            return JSONResponse(render_recipe(body.get("selection") or {}))
        except RecipeError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @app.post("/api/build")
    async def post_build(request: Request):
        body = await request.json()
        try:
            recipe = render_recipe(body.get("selection") or {})
        except RecipeError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        name = body["name"]
        recipes_dir = Path(repo_root) / "recipes"
        recipes_dir.mkdir(exist_ok=True)
        (recipes_dir / f"{name}.json").write_text(json.dumps(recipe, indent=2))
        steps = build_command(name, body["cuda_tag"], body["registry"], bool(body["push"]))
        try:
            bid = manager.start(steps)
        except BuildBusy as e:
            return JSONResponse({"error": str(e)}, status_code=409)
        return JSONResponse({"build_id": bid})

    @app.get("/api/build/{bid}")
    async def build_status(bid: str):
        r = manager.get(bid)
        if not r:
            return JSONResponse({"error": "unknown build"}, status_code=404)
        return JSONResponse({"status": r.status, "exit_code": r.exit_code})

    @app.get("/api/build/{bid}/log")
    async def build_log(bid: str):
        r = manager.get(bid)
        if not r:
            return JSONResponse({"error": "unknown build"}, status_code=404)
        async def gen():
            async for line in r.log.stream():
                yield f"data: {line}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    app.mount("/syalia-ui", StaticFiles(directory=_GATEWAY_STATIC), name="syalia-ui")

    @app.get("/")
    async def index():
        return FileResponse(_STATIC / "build.html")

    return app
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_builder_app.py -v`
Expected: PASS (4 passed). (`GET /` is exercised in Task 5; the file need not exist for these tests.)

- [ ] **Step 5: Commit**

```bash
git add src/ainbox_builder/app.py tests/test_builder_app.py
git commit -m "feat(builder): FastAPI app — catalog/recipe/build/log endpoints"
```

---

### Task 5: The page + launcher + wiring

**Files:**
- Create: `src/ainbox_builder/static/build.html`, `scripts/build_ui.py`
- Modify: `Makefile`, `pyproject.toml`
- Test: `tests/test_builder_app.py` (add one served-page assertion)

**Interfaces:**
- Consumes: all endpoints from Task 4. No new Python interfaces.

- [ ] **Step 1: Write the page** — `src/ainbox_builder/static/build.html`. Adapt `src/ainbox_gateway/static/ui.html` (same `<head>`/`syalia-ui` links + CSS), replacing the body script. Keep the header brand block; swap the relaunch button for a **Build bar** and add a **log panel**. Full body script:

```html
<header>
  <img class="brandmark" src="/syalia-ui/favicon.svg" alt="SYALIA"/>
  <div class="titles">
    <h1><span class="syalia-wordmark">SYALIA</span><span class="syalia-appname">· Builder</span></h1>
    <div class="eyebrow">recipe · bake · push</div>
  </div>
  <div class="spacer"></div>
  <input id="name" class="pick" placeholder="recipe name" style="max-width:150px"/>
  <select id="cuda" class="pick" title="CUDA base"></select>
  <input id="registry" class="pick" value="registry.syalia.dev" style="max-width:190px"/>
  <label style="display:flex;align-items:center;gap:6px;font-size:12px;color:var(--text-dim)">
    <span class="toggle on" id="push" role="switch" aria-checked="true"></span>push</label>
  <button class="relaunch" id="build"><span class="material-icons">build</span>
    <span id="blabel">Build</span></button>
</header>
<main id="board"></main>
<section style="max-width:1180px;margin:0 auto;padding:0 26px 60px">
  <div class="rail-head"><span class="kind">Build log</span>
    <span class="desc" id="pill">idle</span></div>
  <pre id="log" style="background:var(--surface);border:1px solid var(--border);
    border-radius:10px;padding:14px;font-family:var(--font-mono);font-size:12px;
    color:var(--text-dim);max-height:52vh;overflow:auto;white-space:pre-wrap"></pre>
</section>
<div id="toast"></div>
<script>
const RAILS = {llm:"LLM · language models", stt:"STT · speech to text", embeddings:"Embeddings · vectors"};
const CUDA = {"Blackwell (5090)":"12.8.1-devel-ubuntu22.04", "Ada (4060)":"12.2.2-devel-ubuntu22.04"};
const $ = s => document.querySelector(s);
let CATALOG = {llm:{},stt:{},embeddings:{}};
let sel = {llm:[], stt:[], embeddings:[]};

(async function boot(){
  CATALOG = await (await fetch("/api/catalog")).json();
  const cu = $("#cuda"); Object.entries(CUDA).forEach(([k,v])=>{
    const o=document.createElement("option"); o.value=v; o.textContent=k; cu.appendChild(o);});
  $("#push").onclick = () => { const t=$("#push"); const on=!t.classList.contains("on");
    t.classList.toggle("on",on); t.setAttribute("aria-checked",on); };
  render();
})();

function render(){
  const board=$("#board"); board.innerHTML="";
  for(const kind of Object.keys(RAILS)){
    const rail=document.createElement("section"); rail.className="rail";
    rail.innerHTML=`<div class="rail-head"><span class="kind">${RAILS[kind].split(" · ")[0]}</span>
      <span class="desc">${RAILS[kind].split(" · ")[1]}</span></div>`;
    const add=document.createElement("button"); add.className="add";
    add.innerHTML=`<span class="material-icons">add</span>add`;
    add.onclick=e=>openMenu(e,kind); rail.querySelector(".rail-head").appendChild(add);
    const mods=document.createElement("div"); mods.className="modules";
    if(!sel[kind].length) mods.innerHTML=`<div class="empty">Nothing selected — <em>add</em> a model.</div>`;
    else sel[kind].forEach((m,i)=>mods.appendChild(card(kind,m,i)));
    rail.appendChild(mods); board.appendChild(rail);
  }
}
function card(kind,m,i){
  const el=document.createElement("div"); el.className="module";
  el.innerHTML=`<div class="m-top"><span class="dot"></span>
    <span class="slug" title="${m.alias}">${m.alias}</span></div>`;
  const rm=document.createElement("button"); rm.className="rm";
  rm.innerHTML=`<span class="material-icons">close</span>`;
  rm.onclick=()=>{sel[kind].splice(i,1);render();};
  el.querySelector(".m-top").appendChild(rm);
  const info=kind==="llm"?(CATALOG.llm[m.alias]?.size||"custom"):(m.model||"");
  el.insertAdjacentHTML("beforeend",`<div class="ctl"><label>${kind==="llm"?"size":"model"}</label>
    <span class="val">${info}</span></div>`);
  return el;
}
function openMenu(e,kind){
  e.stopPropagation(); document.querySelectorAll(".menu").forEach(m=>m.remove());
  const menu=document.createElement("div"); menu.className="menu";
  Object.keys(CATALOG[kind]).forEach(slug=>{
    const b=document.createElement("button");
    const present=sel[kind].some(m=>m.alias===slug);
    if(present){b.disabled=true;b.textContent=slug+"  · added";}
    else{b.textContent=slug;b.onclick=()=>{addModel(kind,slug);close();render();};}
    menu.appendChild(b);
  });
  if(kind==="llm"){ const b=document.createElement("button"); b.textContent="+ custom URL…";
    b.onclick=()=>{ const url=prompt("HF .gguf resolve URL"); if(!url)return;
      const alias=prompt("alias (slug)"); if(!alias)return;
      sel.llm.push({alias,url}); close(); render(); }; menu.appendChild(b); }
  e.currentTarget.appendChild(menu);
  setTimeout(()=>document.addEventListener("click",close,{once:true}),0);
  function close(){document.querySelectorAll(".menu").forEach(m=>m.remove());}
}
function addModel(kind,slug){
  if(kind==="llm") sel.llm.push({alias:slug, url:CATALOG.llm[slug].url});
  else sel[kind].push({alias:slug, model:CATALOG[kind][slug].model});
}
async function build(){
  const name=$("#name").value.trim();
  if(!name){toast("Name the recipe.","err");return;}
  if(!sel.llm.length){toast("Add at least one LLM.","err");return;}
  const btn=$("#build"); btn.classList.add("busy"); btn.disabled=true;
  $("#log").textContent=""; $("#pill").textContent="building";
  const body={name, cuda_tag:$("#cuda").value, registry:$("#registry").value.trim(),
    push:$("#push").classList.contains("on"), selection:sel};
  const r=await fetch("/api/build",{method:"POST",headers:{"content-type":"application/json"},
    body:JSON.stringify(body)});
  const data=await r.json();
  if(!r.ok){toast(data.error||"build failed","err"); $("#pill").textContent="failed";
    btn.classList.remove("busy"); btn.disabled=false; return;}
  const es=new EventSource(`/api/build/${data.build_id}/log`);
  const log=$("#log");
  es.onmessage=e=>{ log.textContent+=e.data+"\n"; log.scrollTop=log.scrollHeight; };
  es.onerror=async ()=>{ es.close();
    const st=await (await fetch(`/api/build/${data.build_id}`)).json();
    $("#pill").textContent=st.status; toast(st.status==="done"?"Done.":"Failed.",
      st.status==="done"?"ok":"err");
    btn.classList.remove("busy"); btn.disabled=false; };
}
function toast(msg,kind){ const t=$("#toast"); t.textContent=msg; t.className="show "+(kind||"");
  clearTimeout(toast._t); toast._t=setTimeout(()=>t.className="",2600); }
$("#build").onclick=build;
</script>
```

Copy the `<head>` (lines 1–132 of `ui.html`) and the CSS verbatim; keep `.relaunch`/`.relaunch.dirty` styles for the Build button.

- [ ] **Step 2: Write the launcher** — `scripts/build_ui.py`

```python
"""Launch the ainbox-builder UI on the local Docker host."""
import os
import uvicorn
from ainbox_builder.app import create_app


def main():
    repo_root = os.environ.get("BUILDER_REPO_ROOT", os.getcwd())
    app = create_app(repo_root=repo_root)
    uvicorn.run(app, host=os.environ.get("BUILDER_HOST", "0.0.0.0"),
                port=int(os.environ.get("BUILDER_PORT", "8090")))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Wire Makefile + pyproject**

Append to `Makefile`:

```make
# --- Builder UI ---
ui-build:
	@.venv/bin/python scripts/build_ui.py
```

In `pyproject.toml`, under `[tool.setuptools.package-data]` add:

```toml
ainbox_builder = ["static/*.html"]
```

- [ ] **Step 4: Add served-page test** — append to `tests/test_builder_app.py`

```python
@pytest.mark.asyncio
async def test_serves_page(tmp_path):
    async with _client(_app(tmp_path)) as c:
        r = await c.get("/")
        assert r.status_code == 200 and "Builder" in r.text
```

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/test_builder_recipe.py tests/test_builder_command.py tests/test_builder_runner.py tests/test_builder_app.py -v`
Expected: PASS (all builder tests green).

- [ ] **Step 6: Manual smoke** (on a Docker host with the repo)

Run: `make ui-build` then open `http://<host>:8090`, add `qwen3.5-0.8b`, name `smoke1`, push OFF, click **Build**; watch the log stream a real `make image`.

- [ ] **Step 7: Commit**

```bash
git add src/ainbox_builder/static/build.html scripts/build_ui.py Makefile pyproject.toml tests/test_builder_app.py
git commit -m "feat(builder): build.html page, launcher, make target, packaging"
```

---

## Self-Review

**Spec coverage:** catalog+recipe (Task 1) ✓; 3 rails LLM/STT/emb (Task 5) ✓; build bar name/CUDA/registry/push (Task 5) ✓; build+push orchestration (Tasks 2–3) ✓; SSE log + status pill (Tasks 4–5) ✓; endpoints table (Task 4) ✓; one-at-a-time 409 (Task 3–4) ✓; reject empty LLM (Task 1) ✓; custom URL (Task 5) ✓; syalia-ui reuse from gateway (Task 4) ✓; launcher/make/packaging (Task 5) ✓. Phase 2 (forge/infra.syalia.dev/Caddy basic-auth) is explicitly out of this plan's scope per the spec.

**Placeholder scan:** none — every step carries runnable code/commands.

**Type consistency:** `Step(label, argv, env)`, `render_recipe(selection)->dict`, `build_command(name,cuda_tag,registry,push)->list[Step]`, `BuildManager.start(steps)->bid` / `.get(bid)`, `create_app(repo_root, catalog, spawn)` — consistent across Tasks 1–5. SSE consumed via `EventSource` matching `text/event-stream` from `/api/build/{id}/log`.

**Note for implementer:** the qwen3.5 catalog URLs mirror `recipes/rtx4060_v1.json` (already in-repo and known-good); gemma4 URLs are HF-verified. If a qwen3.5 HF path 404s at build time, that surfaces in the live log — fix the catalog entry, not the plan.
