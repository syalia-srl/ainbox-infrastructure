# ainbox-builder — Build-time recipe UI — Design

**Status:** approved (brainstorm 2026-07-03)
**Author:** Alex + Claude
**Companion runtime UI:** `src/ainbox_gateway/static/ui.html` (the raise-spec editor served *inside* the engine). This design is its **build-time** sibling.

## Goal

A tiny standalone web UI to **author a recipe** (pick which models to bake), **trigger a build** with one button, **watch progress live**, and **push** the resulting image to a registry (default `registry.syalia.dev`). It runs on a host that has Docker + a checkout of this repo, and shells out to the existing `make image` path.

## Why a separate app (not the gateway)

The runtime UI lives inside the gateway, which is baked into the appliance image and has neither Docker nor the repo. Building is a *build-host* concern: it needs the Docker daemon, the build context (`build/` + `src/`), and registry credentials. So the builder is its **own** small FastAPI app, reusing the `syalia-ui` design system.

## Non-goals (YAGNI)

- TTS / image-generation rails (future rails, same pattern).
- Auth inside the app (auth is applied at the reverse proxy for the hosted instance; local runs are trusted).
- Concurrent builds (one at a time).
- Editing the Dockerfile beyond the CUDA-base selector. The Dockerfile stays fixed and is parametrized by the `CUDA_TAG` build-arg.
- Recipe *library* management / history (v1 writes `recipes/<name>.json` and builds; browsing past recipes is future).

## Component

`src/ainbox_builder/` — new package, sibling to `ainbox_gateway`. No new dependencies (FastAPI + uvicorn already present).

    src/ainbox_builder/
      __init__.py
      app.py            # create_app(...) — routes + build orchestration
      catalog.py        # curated model catalog (or catalog.json loaded here)
      recipe.py         # pure: selection -> recipe dict (validated against gateway.spec)
      builder.py        # pure-ish: build_command(...) -> shell steps; Build runner + log buffer
      static/build.html # the page (mirrors ui.html structure + syalia-ui assets)
    scripts/build_ui.py # launcher (mirror of scripts/demo_ui.py)
    Makefile            # + `ui-build` target

The `syalia-ui/` static assets are shared with the gateway; the builder serves its own copy (or the same mount).

## The page (`build.html`)

Structure mirrors `ui.html`:

- **Three rails** — `LLM` · `STT` · `Embeddings`. Each card is added from a curated `CATALOG`; per-modality controls:
  - **LLM:** alias (editable slug), the resolved GGUF URL (read-only), and size. (No `n_ctx`/replicas here — those are *deploy*-time knobs the recipe doesn't carry; they belong to the runtime raise-spec, not the bake.) A **"+ custom URL"** entry lets you paste any HF `.gguf` resolve URL with an alias.
  - **STT:** whisper model (`tiny`/`small`), alias.
  - **Embeddings:** model (minilm), read-only.
- **Build bar** (header): recipe **name** (text) · **CUDA base** select `Ada (12.2.2-devel-ubuntu22.04)` / `Blackwell (12.8.1-devel-ubuntu22.04)` · **registry** (text, default `registry.syalia.dev`) · **push** toggle (default on) · **Build** button.
- **Log panel** — live streamed build + push output, monospace, auto-scroll; a status pill: `idle → building → pushing → done | failed`.

## Endpoints

| Method | Route | Purpose |
|---|---|---|
| `GET`  | `/` | serve `build.html` |
| `GET`  | `/api/catalog` | curated catalog JSON (llm/stt/embeddings entries) |
| `POST` | `/api/recipe` | validate selection → return rendered `recipe.json` (preview / download). 400 on invalid. |
| `POST` | `/api/build` | write `recipes/<name>.json`, start the build, return `{build_id}` |
| `GET`  | `/api/build/{id}/log` | **SSE** stream of build + push stdout/stderr |
| `GET`  | `/api/build/{id}` | `{status, exit_code, image, pushed}` |

## Recipe generation (`recipe.py` — pure, testable)

`render_recipe(selection: dict) -> dict` maps the UI selection to the recipe schema consumed by `build/Dockerfile`:

- LLM → `llama_node: [{"url": <gguf url>, "alias": <slug>}]`
- STT → `whisper_nodes: [{"model": <tiny|small>, "alias": <slug>}]`
- Embeddings → `embedding_nodes: [{"model": <hf name>}]`
- `tts_nodes: []`, `image_nodes: []` (empty — out of scope)

The rendered recipe's *deploy shape* is not produced here (that's the raise-spec / runtime UI). But we **validate model presence** and reject an empty `llama_node` (a recipe with no LLM is rejected, mirroring `load_spec`'s "at least one llm" rule at deploy time — caught early here).

## Build execution (`builder.py`)

- `build_command(name, cuda_tag, registry, push) -> list[Step]` — **pure** function returning the ordered shell steps:
  1. `env CUDA_TAG=<cuda_tag> make image RECIPE=recipes/<name>.json`
  2. if `push`: `docker tag superbot:<name> <registry>/ainbox-infra/<name>:latest`
  3. if `push`: `docker push <registry>/ainbox-infra/<name>:latest`
- `BuildRunner` — runs the steps via `asyncio.create_subprocess_exec`, streaming merged stdout/stderr line-by-line into a per-`build_id` buffer (an `asyncio.Queue` + retained list for late subscribers). Exposes an async iterator the SSE endpoint drains. Sets terminal status + exit code. A module-level lock enforces **one build at a time** (a second `POST /api/build` while busy → 409).
- Push auth is host-provided: the host is `docker login`'d to the registry (or, on forge, pushes to the in-network registry — see Deployment). A 401 surfaces verbatim in the log; status → `failed`.

## Catalog (`catalog.py` / `catalog.json`)

Curated, with real HF GGUF resolve URLs:

- **llm:** `gemma4-e4b`, `gemma4-e2b`, `qwen3-14b`, `qwen3.5-9b`, `qwen3.5-4b`, `qwen3.5-2b`, `qwen3.5-0.8b` — each `{url, size, note}`. Plus a `__custom__` affordance in the UI for an arbitrary HF `.gguf` URL.
- **stt:** `whisper-tiny`, `whisper-small`.
- **embeddings:** `minilm` → `paraphrase-multilingual-MiniLM-L12-v2`.

(Seed URLs — gemma4-e4b: `unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf`; qwen3-14b: `unsloth/Qwen3-14B-GGUF/Qwen3-14B-Q4_K_M.gguf`; qwen3.5-* from the existing `recipes/rtx4060_v1.json`.)

## Testing (TDD, kept inline — never delegated)

1. `render_recipe` — a selection → exact recipe dict; empty-LLM → error; custom URL passthrough.
2. `GET /api/catalog` — returns the catalog.
3. `build_command` — correct ordered steps for push on/off, custom registry, each CUDA base.
4. `BuildRunner` SSE — with a **fake** subprocess (echoes lines, controllable exit): log lines stream in order, terminal status + exit code correct, second concurrent build → 409.
5. `POST /api/recipe` — valid → 200 + JSON; invalid → 400.

Real Docker execution is **not** unit-tested (shell-out); the pure command assembly + streaming are.

## Deployment — Phase 2 (`infra.syalia.dev`)

Target host: **forge** (already runs `registry.syalia.dev` + the self-hosted runner). Co-locating the builder with the registry means pushes traverse the internal Docker network, avoiding the Caddy write-gate entirely.

- **Container (DooD):** a `Dockerfile.builder` + a compose service that mounts `/var/run/docker.sock` and includes the repo build context (baked in, or a mounted checkout kept current with `git pull`). Mirrors the appliance's Docker-out-of-Docker pattern.
- **Registry push on forge:** the in-network alias `registry:5000` (plain HTTP, no Caddy) is reachable, so the builder can push there directly — no `docker login`. The registry field defaults to `registry.syalia.dev` for off-forge runs but is set to the in-network target when deployed on forge.
- **Reverse proxy + auth:** a Caddy vhost `infra.syalia.dev` → the builder; **basic-auth** (or warden) gates it, since it can trigger builds and push to the *production* registry. The app itself stays authless (auth at the proxy), matching the runtime UI's stance.
- **DNS:** `infra.syalia.dev` A/AAAA → forge (Hetzner Cloud DNS, syalia project).

Phase 2 is **designed-for, not built now.** Phase 1 (local run on any Docker host) is the implementation scope; the containerization + Caddy + DNS land as a follow-up once the builder works.

## Phasing

- **Phase 1 (this plan):** the `ainbox_builder` app + page + endpoints + build/push orchestration + tests. Runs locally via `make ui-build` on a Docker host (smaug today). End-to-end: pick models → build → watch → push to `registry.syalia.dev`.
- **Phase 2 (follow-up):** dockerize + deploy to forge at `infra.syalia.dev` behind Caddy auth, pushing to the in-network registry.
