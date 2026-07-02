# ainbox-infrastructure Tiny UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline). Steps use checkbox (`- [ ]`).

**Goal:** A tiny admin UI on the gateway to view the raised model set, edit the raise-spec, and apply+relaunch — no hot-swap, a full in-process relaunch.

**Architecture:** The gateway gains a control plane. `_apply(spec, raw)` stops the supervisor, restarts it with the new spec, and rebuilds the LLM/embeddings/STT registries — all validated **before** anything is torn down. A single static HTML page (textarea + status panel) drives three JSON endpoints. Relaunch runs in `asyncio.to_thread` (supervisor start/stop is blocking).

**Tech Stack:** Same gateway package. No new deps (FastAPI `FileResponse`, vanilla JS).

## Global Constraints

- **Validate before teardown:** a bad spec returns 400 and leaves the running set untouched.
- **In-process relaunch:** `supervisor.stop()` then `supervisor.start(new_spec)`; registries rebuilt. Optional persistence to `spec_path` so a real process restart keeps the edit.
- **Tiny:** one static page, three endpoints. No auth (admin panel on the engine host, behind warden/localhost in prod).
- **Commit style:** conventional commits; `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## File Structure

- `src/ainbox_gateway/app.py` — `spec_raw`/`spec_path` params, `_apply`/`_status` helpers, `GET /`, `GET /api/spec`, `POST /api/spec`, `GET /api/status`.
- `src/ainbox_gateway/static/ui.html` — the admin page.
- `src/ainbox_gateway/__main__.py` — pass `spec_raw` + `spec_path` from the loaded file.
- `scripts/demo_ui.py` — launch the app with fakes + a sample spec for local viewing.
- Tests: `tests/test_app.py`.

---

### Task 1: Relaunch mechanism + control state

**Files:** Modify `src/ainbox_gateway/app.py`; Test `tests/test_app.py`.

**Interfaces:**
- `create_app(spec, supervisor, client=None, embedder_factory=None, transcriber_factory=None, spec_raw=None, spec_path=None)`.
- Startup stores `app.state.spec`, `app.state.spec_raw`.
- Internal `_apply(new_spec, new_raw)` (stop → start → rebuild registries → update state → persist if `spec_path`); `_status()` → `{"llm":[...], "embeddings":[...], "stt":[...]}` sorted.

- [ ] **Step 1: Write failing tests** (append to `tests/test_app.py`)

```python
import json


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
    assert st == {"llm": ["b"], "embeddings": [], "stt": ["w"]}
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
async def test_get_spec_returns_raw():
    raw = {"gateway": {"port": 8080}, "llm": [{"slug": "a"}]}
    app = create_app(Spec(gateway_port=8080, llm=[LlmNode(slug="a")]),
                     FakeSupervisor(), embedder_factory=_FakeEmbedder,
                     transcriber_factory=_FakeTranscriber, spec_raw=raw)
    async with _client(app) as c:
        got = (await c.get("/api/spec")).json()
    assert got == raw
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_app.py -k "apply or get_spec" -v`
Expected: FAIL — `create_app()` has no `spec_raw` / routes missing.

- [ ] **Step 3: Implement** — in `app.py`:

Add imports at top: `import json` and `from pathlib import Path`; `from .spec import load_spec, SpecError` (extend the existing spec import).

Change the signature + lifespan and add helpers:

```python
def create_app(spec: Spec, supervisor: Supervisor,
               client: httpx.AsyncClient | None = None,
               embedder_factory=None, transcriber_factory=None,
               spec_raw: dict | None = None, spec_path: str | None = None) -> FastAPI:
    client = client or httpx.AsyncClient(timeout=None)
    embedder_factory = embedder_factory or _default_embedder_factory
    transcriber_factory = transcriber_factory or _default_transcriber_factory

    def _start(new_spec: Spec, new_raw: dict | None) -> None:
        pools = supervisor.start(new_spec)
        app.state.router = Router(pools)
        app.state.embedders = build_embedders(new_spec, embedder_factory)
        app.state.transcribers = build_transcribers(new_spec, transcriber_factory)
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
                "stt": sorted(app.state.transcribers)}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _start(spec, spec_raw)
        yield
        supervisor.stop()
        await client.aclose()

    app = FastAPI(title="ainbox-infrastructure gateway", lifespan=lifespan)
```

(Keep the existing `_router()` helper and all `/v1/*` routes as-is.) Add control routes before `app.state.client = client`:

```python
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
```

Note the existing `/v1/*` handlers reference the closed-over `router`; change `_proxy`'s `router.resolve` to `_router().resolve` if not already (Task 8 of the core plan already switched to `_router()`), and `_router()` must read `app.state.router` (already the case).

- [ ] **Step 4: Run whole suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ainbox_gateway/app.py tests/test_app.py
git commit -m "feat(gateway): control plane — get/set spec + relaunch + status

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: The admin page (`GET /`)

**Files:** Create `src/ainbox_gateway/static/ui.html`; Modify `src/ainbox_gateway/app.py`; Test `tests/test_app.py`.

**Interfaces:** `GET /` → `text/html` (the admin page).

- [ ] **Step 1: Write failing test** (append to `tests/test_app.py`)

```python
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
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_app.py::test_root_serves_ui -v`
Expected: FAIL — 404.

- [ ] **Step 3: Create `src/ainbox_gateway/static/ui.html`** (self-contained: fetches `/api/spec` + `/api/status`, POSTs on Apply):

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ainbox · inference engine</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
         background: #0e1116; color: #d7dde5; }
  header { padding: 20px 28px; border-bottom: 1px solid #222a35;
           display: flex; align-items: baseline; gap: 14px; }
  header h1 { margin: 0; font-size: 17px; letter-spacing: .5px; }
  header .tag { color: #6b7684; font-size: 12px; }
  main { display: grid; grid-template-columns: 1fr 320px; gap: 24px; padding: 24px 28px; }
  label { display: block; color: #8b95a3; font-size: 12px; margin-bottom: 8px;
          text-transform: uppercase; letter-spacing: .8px; }
  textarea { width: 100%; height: 420px; resize: vertical; background: #11161d;
             color: #cfe3ff; border: 1px solid #222a35; border-radius: 8px;
             padding: 14px; font: 13px/1.55 ui-monospace, monospace; }
  button { margin-top: 14px; background: #2f6feb; color: #fff; border: 0;
           border-radius: 7px; padding: 10px 18px; font-size: 14px; cursor: pointer; }
  button:hover { background: #3b7bff; }
  aside { background: #11161d; border: 1px solid #222a35; border-radius: 8px; padding: 16px; }
  aside h2 { margin: 0 0 12px; font-size: 12px; text-transform: uppercase;
             letter-spacing: .8px; color: #8b95a3; }
  .kind { margin-bottom: 14px; }
  .kind b { color: #7ee0a2; font-size: 12px; }
  .pill { display: inline-block; background: #1b2430; border: 1px solid #2a3646;
          border-radius: 999px; padding: 3px 10px; margin: 4px 4px 0 0; font-size: 12px; }
  .muted { color: #58616e; }
  #msg { margin-top: 12px; font-size: 13px; min-height: 18px; }
  .ok { color: #7ee0a2; } .err { color: #ff8a8a; }
</style>
</head>
<body>
<header>
  <h1>ainbox · inference engine</h1>
  <span class="tag">pure-OpenAI gateway · fixed residency · relaunch to change</span>
</header>
<main>
  <section>
    <label for="spec">raise-spec (json)</label>
    <textarea id="spec" spellcheck="false"></textarea>
    <button id="apply">Apply &amp; Relaunch</button>
    <div id="msg"></div>
  </section>
  <aside>
    <h2>Raised now</h2>
    <div id="status"><span class="muted">loading…</span></div>
  </aside>
</main>
<script>
const $ = s => document.querySelector(s);
function renderStatus(st) {
  const kind = (name, arr) => `<div class="kind"><b>${name}</b><br/>` +
    (arr.length ? arr.map(x => `<span class="pill">${x}</span>`).join("")
                : `<span class="muted">none</span>`) + `</div>`;
  $("#status").innerHTML = kind("llm", st.llm) + kind("embeddings", st.embeddings) + kind("stt", st.stt);
}
async function load() {
  $("#spec").value = JSON.stringify(await (await fetch("/api/spec")).json(), null, 2);
  renderStatus(await (await fetch("/api/status")).json());
}
$("#apply").onclick = async () => {
  const msg = $("#msg"); msg.textContent = "relaunching…"; msg.className = "";
  let body;
  try { body = JSON.parse($("#spec").value); }
  catch (e) { msg.textContent = "invalid JSON: " + e.message; msg.className = "err"; return; }
  const r = await fetch("/api/spec", {method: "POST", headers: {"content-type": "application/json"},
                                      body: JSON.stringify(body)});
  const data = await r.json();
  if (r.ok) { msg.textContent = "relaunched ✓"; msg.className = "ok"; renderStatus(data.status); }
  else { msg.textContent = data.error || "error"; msg.className = "err"; }
};
load();
</script>
</body>
</html>
```

- [ ] **Step 4: Serve it** — in `app.py`, add near the control routes:

```python
from fastapi.responses import FileResponse
from pathlib import Path as _Path

_UI_FILE = _Path(__file__).parent / "static" / "ui.html"

    @app.get("/")
    async def ui() -> Response:
        return FileResponse(_UI_FILE)
```

(Place the `_UI_FILE` module constant at top level with the other imports; the route goes inside `create_app`.)

- [ ] **Step 5: Ensure the static file ships in the package** — in `pyproject.toml` add:

```toml
[tool.setuptools.package-data]
ainbox_gateway = ["static/*.html"]
```

- [ ] **Step 6: Run suite + commit**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS.

```bash
git add src/ainbox_gateway/app.py src/ainbox_gateway/static/ui.html pyproject.toml tests/test_app.py
git commit -m "feat(gateway): tiny admin UI (view/edit raise-spec, relaunch, status)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `__main__` wiring + demo launcher

**Files:** Modify `src/ainbox_gateway/__main__.py`; Create `scripts/demo_ui.py`.

- [ ] **Step 1: Pass raw + path from `__main__`** — update `main()`:

```python
def main() -> None:
    with open(_SPEC_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    spec = load_spec(raw)
    app = create_app(spec, LlamaSupervisor(), spec_raw=raw, spec_path=_SPEC_PATH)
    uvicorn.run(app, host="0.0.0.0", port=spec.gateway_port)
```

- [ ] **Step 2: Create `scripts/demo_ui.py`** — runs the app with fakes (no GPU/models) so the UI can be viewed locally:

```python
"""Launch the gateway UI with fake backends (no GPU) for local viewing."""
import uvicorn

from ainbox_gateway.app import create_app
from ainbox_gateway.spec import Spec, LlmNode, EmbeddingsNode, SttNode
from ainbox_gateway.supervisor import build_pools


class _FakeSup:
    def start(self, spec): return build_pools(spec)
    def stop(self): pass


class _FakeEmb:
    def __init__(self, n): self.slug = n.slug
    def embed(self, texts): return [[0.0] * 384 for _ in texts]


class _FakeTr:
    def __init__(self, n): self.slug = n.slug
    def transcribe(self, audio, language=None): return "(demo)"


RAW = {
    "gateway": {"port": 8080},
    "llm": [{"slug": "qwen3.5-9b", "replicas": 1, "n_ctx": 8192},
            {"slug": "qwen3.5-2b", "replicas": 2, "n_ctx": 4096}],
    "embeddings": [{"slug": "text-embedding-minilm",
                    "model": "paraphrase-multilingual-MiniLM-L12-v2"}],
    "stt": [{"slug": "whisper-small", "model": "small"}],
}


def main():
    spec = Spec(gateway_port=8080,
                llm=[LlmNode(slug="qwen3.5-9b", n_ctx=8192),
                     LlmNode(slug="qwen3.5-2b", replicas=2, n_ctx=4096)],
                embeddings=[EmbeddingsNode(slug="text-embedding-minilm",
                                           model="paraphrase-multilingual-MiniLM-L12-v2")],
                stt=[SttNode(slug="whisper-small", model="small")])
    app = create_app(spec, _FakeSup(), embedder_factory=_FakeEmb,
                     transcriber_factory=_FakeTr, spec_raw=RAW)
    uvicorn.run(app, host="127.0.0.1", port=8080)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add src/ainbox_gateway/__main__.py scripts/demo_ui.py
git commit -m "feat(gateway): wire spec_raw/path in __main__; add fake-backend UI demo launcher

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

## Self-Review

- **Coverage:** view raised set ✓ (`/api/status`, T1), edit+relaunch ✓ (`POST /api/spec`, T1), validate-before-teardown ✓ (T1), persistence ✓ (T1), admin page ✓ (T2), local viewing ✓ (T3 demo).
- **Placeholders:** none — full HTML/JS and code included.
- **Type consistency:** `create_app(..., spec_raw, spec_path)`, `_start`/`_apply`/`_status`, `app.state.spec_raw` consistent across tasks.
