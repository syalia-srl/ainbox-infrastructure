---
type: design
status: draft
date: 2026-07-02
repos: [ainbox-infrastructure, ainbox, ainbox-os, warden]
supersedes_infra: superbot-infraestructure (engine), ollama service (ainbox + ainbox-os)
related:
  - vault/Atlas/Architecture/2026-06-09-ainbox-deployment-modes-design.md (desktop vs server modes)
  - ainbox/docs/2026-06-26-warden-shared-ml-service-design.md (ML endpoints this engine backs)
  - ainbox/docs/2026-06-30-unified-fast-fetch-and-self-hosted-models-design.md (delivery; Ollama-era)
---

# ainbox-infrastructure — unified GPU-first ML & inference backend

## Problem

The AI-n-Box suite's inference engine has churned twice — `lms` → Ollama
(ainbox-os v0.3) → and, in the compose rig, a plain `ollama` service. Each
churn re-plumbs how every app reaches a model. On top of that, ML compute
(embeddings, transcription) is a *second*, separate story: baked into apps and
partially migrated behind warden's shared-ML endpoints. There is no single
place that owns "run a model" for the whole suite.

We want **one backend** that:

- serves **LLM, embeddings, STT** today (and reserves **TTS** and
  **image generation** for later) behind **one pure OpenAI-compatible API**;
- is **GPU-first**;
- raises a **fixed, declared subset** of models (no hot-swapping) and
  **round-robins same-slug replicas**;
- is driven by a **JSON spec** editable through a **tiny UI** that relaunches
  the stack;
- keeps the suite's existing invariant: **apps never call the engine directly —
  they go through warden**, which owns the agent.

This repo (`ainbox-infrastructure`, née `superbot-infraestructure`) is that
backend. It generalizes the SuperBot engine (llama.cpp + faster-whisper,
recipe→deploy→LoRA, models baked into the image) into the suite-wide inference
substrate, replacing Ollama in **both** the `ainbox` compose rig and
`ainbox-os`.

## Goals

1. A single **pure OpenAI-compatible** front door for all model inference.
2. **Modality-agnostic** routing: adding a modality adds a backend + a route;
   the front door and spec machinery do not change.
3. **Fixed residency**, declared by a JSON spec: bake a superset, raise a
   chosen subset, all resident until relaunch.
4. **Same-slug round-robin**: N replicas sharing a slug are load-balanced.
5. **Zero embedding migration**: existing magpie/superbot MiniLM vectors stay
   valid.
6. Drop-in replacement for the Ollama service in ainbox + ainbox-os.

## Non-goals

- **No hot-swap / on-demand model loading** (explicitly rejected — llama-swap is
  the wrong tool here). Changing the raised set is a full relaunch.
- **No automatic VRAM management.** The spec author (via the UI) is responsible
  for choosing a raised set that fits the hardware. The engine raises what it's
  told and fails loudly if it doesn't fit.
- **No app-facing API changes in this spec.** How warden consumes the engine
  and how apps consume warden is warden's concern; this spec defines only the
  engine's OpenAI contract and the warden→engine boundary.
- **No build of TTS or image-gen in v1** — they are reserved in the interface
  (spec-only), designed but not implemented.

## The central invariant — one OpenAI API, N backends

Everything reaches the engine through a single OpenAI-compatible surface. The
caller selects the target with the standard `model` field; the engine routes to
the backend pool that serves that slug. The number and kind of backend runtimes
behind the door is an implementation detail.

| Modality | OpenAI route | Backend runtime | Status |
|---|---|---|---|
| LLM chat/completions | `POST /v1/chat/completions`, `/v1/completions` | llama.cpp `llama-server` | **v1 (built)** |
| Embeddings | `POST /v1/embeddings` | fastembed-GPU (ONNX/CUDA), MiniLM-L12 | **v1 (built)** |
| STT | `POST /v1/audio/transcriptions` | faster-whisper (CTranslate2) | **v1 (built)** |
| Model list | `GET /v1/models` | gateway (aggregated) | **v1 (built)** |
| TTS | `POST /v1/audio/speech` | Kokoro / Piper / XTTS (TBD) | **spec-only** |
| Image generation | `POST /v1/images/generations` | FLUX via ComfyUI or diffusers | **spec-only** |

"GPU-first" means *where compute runs*, not *one inference library*. The engine
already runs multiple GPU-capable runtimes (whisper was never llama.cpp); this
design embraces that: **llama.cpp + fastembed-GPU + faster-whisper** in v1, and
reserves diffusion + a TTS runtime behind the same door.

## Architecture

```
                       ┌─────────────────────────────────────────────┐
   apps ──▶ warden ──▶ │  ainbox-infrastructure (one container/image) │
 (magpie,  (auth +     │                                              │
  superbot, agent)     │   ┌────────── gateway ──────────┐            │
  peacock)             │   │  pure OpenAI /v1/*  + UI     │            │
                       │   │  route by `model` slug       │            │
                       │   │  round-robin same-slug pools │            │
                       │   └───┬───────┬───────┬──────────┘            │
                       │       │       │       │                       │
                       │  llama-server fastembed faster-whisper …      │
                       │   (×N pools)  (GPU)    (STT)   (TTS/FLUX rsvd) │
                       │       └───────┴───────┴──── baked models ─────┤
                       │              /models/*  (superset library)    │
                       └──────────────────────────────────────────────┘
```

Two build/run phases are preserved from the SuperBot engine, with clearer roles:

### Phase 1 — Build (bake a superset "library")

A **recipe** (`recipes/*.json`) lists everything to bake into the image: LLM
GGUFs, the MiniLM ONNX embedding model, whisper models (and later TTS voices,
FLUX weights). `build/Dockerfile` compiles `llama-server`, installs the Python
runtimes, and bakes each asset to a fixed path (`/models/<alias>.gguf` for LLMs,
an ONNX cache for MiniLM, the whisper cache, etc.). The recipe is a **superset**
— it may contain far more than any one machine will raise.

Unchanged from today: recipe→`curl`→`/models/<alias>.gguf` symlink baking. New:
the recipe also declares non-llama assets (embedding/whisper/…); the e5-small
embedding model is **dropped**, replaced by MiniLM-L12 ONNX (see Embeddings).

### Phase 2 — Raise (a fixed subset, from the raise-spec)

A **raise-spec** (the evolution of today's `deploy/*.json`) declares the subset
to bring up and how: per-slug model, replica count, port range, and llama.cpp
params (`n_ctx`, `n_gpu_layers`, `flash_attn`, cache quant, LoRAs, reasoning
on/off). The gateway reads it, supervises the backend processes, and exposes the
unified API. All raised backends stay resident until the next relaunch.

### The gateway (bespoke, ~a few hundred lines)

One service that:

1. **Reads the raise-spec** and **supervises** backend processes — spawns
   `llama-server` per LLM replica, the fastembed-GPU embedder, the whisper
   service; polls each for health (reusing today's readiness-semaphore idea).
2. **Serves the pure-OpenAI `/v1/*` surface**, routing by the `model` field to
   the pool for that slug; **round-robins** across replicas in the pool
   (least-recently-used or plain RR; streaming SSE passthrough for chat).
3. **Aggregates `GET /v1/models`** across all raised pools.
4. **Hosts the tiny UI** (same service — it already holds the spec and the
   process handles), which edits the raise-spec and triggers a **relaunch**
   (tear down backends, raise the new set).

Replacing today's `build/entrypoint.sh` bash launch-loop with this gateway is
the core code change. A "relaunch" is the engine's unit of change — there is no
partial reconfiguration.

Failure domain: the gateway supervises its children; if it restarts, backends
restart (acceptable for a single-box appliance; systemd restarts the gateway).
Splitting supervisor from gateway (e.g. supervisord + a stateless proxy) is a
reserved option if robustness demands it — see Open questions.

### The raise-spec (schema sketch)

```jsonc
{
  "gateway": { "port": 8080 },              // the ONE public OpenAI port
  "llm": [
    { "slug": "qwen3.5-9b", "replicas": 1,  // slug = OpenAI `model` name
      "n_ctx": 8192, "n_gpu_layers": -1, "flash_attn": true,
      "cache_type_k": "q8_0", "cache_type_v": "q8_0",
      "loras": [ { "file": "voice.gguf", "alias": "voice", "scale": 1.0 } ] },
    { "slug": "qwen3.5-2b", "replicas": 2, "n_ctx": 4096, "n_gpu_layers": -1 }
  ],
  "embeddings": [
    { "slug": "text-embedding-minilm", "model": "paraphrase-multilingual-MiniLM-L12-v2",
      "device": "cuda" }
  ],
  "stt": [
    { "slug": "whisper-small", "model": "small", "device": "cuda", "compute_type": "float16" }
  ]
  // reserved (spec-only): "tts": [...], "images": [...]
}
```

Notes:
- **`replicas`** raises N `llama-server` instances for the slug on distinct
  internal ports; the gateway pools them and round-robins. (Today's design has
  no replicas — this is the additive change.)
- Internal ports become an implementation detail; only `gateway.port` is public.
- A slug appearing more than once (or with `replicas > 1`) forms one round-robin
  pool.

## Embeddings — MiniLM on GPU, no reindex

Both apps store `paraphrase-multilingual-MiniLM-L12-v2` (384-d) vectors today
(`superbot/src/superbot/tools_meta.py`; `magpie/src/magpie/index.py` guards
"384-d vectors stay compatible — no reindex"). Those are live BeaverDB indexes.

Decision: **serve the exact same MiniLM model on GPU via `fastembed-gpu`**
(fastembed on ONNX Runtime's CUDA provider) behind `/v1/embeddings`. It is the
same ONNX graph and weights, so vectors are numerically equivalent to the
current CPU ones (differences <1e-4, far below any retrieval threshold).

- ✅ **No reindex** — stored vectors stay valid.
- ✅ **GPU-first** — runs on the card (~120 MB VRAM; negligible next to LLMs).
- ✅ **No query/passage prefixes** — still MiniLM, not e5.

Consequence: the recipe **drops `multilingual-e5-small`** and bakes MiniLM's
ONNX instead. The unconditional `--embedding` flag on every `llama-server`
(current `entrypoint.sh` line ~78) is **removed** — chat models generate,
embeddings are served by the fastembed backend only.

## STT — OpenAI-shaped

Today whisper is a FastAPI at `/v1/audio/transcriptions/{model_name}` (model in
the path — non-standard). Under the gateway the public route is the standard
**`POST /v1/audio/transcriptions`** with `model` in the multipart body; the
gateway maps the `model` field to the whisper backend/model. `whisper_api.py`
is refactored to the OpenAI request/response shape (or kept internal and
adapted by the gateway — implementation detail for the plan).

## warden boundary

The engine is pure OpenAI; **warden is the only client**, and the sole front
door for apps (the warden-bridge invariant). Concretely:

- warden's agent (lovelaice) points its OpenAI client's `base_url` at the
  gateway (`WARDEN_LLM_BASE_URL` → `http://<engine>:8080/v1`), selecting a slug
  as `WARDEN_LLM_MODEL`.
- warden's shared-ML endpoints (`/api/transcribe`, `/api/embed` — already the
  path magpie/superbot use) become **thin proxies** to the engine's
  `/v1/audio/transcriptions` and `/v1/embeddings` instead of running models
  in-process. The apps' client code is unchanged — only warden's implementation
  moves from in-process models to an HTTP hop to the engine.

This preserves the entire `warden-shared-ml-service` client migration (the
`mic.js` work, per-app `/api/transcribe` proxies) — the engine simply becomes
what warden's ML endpoints call.

## Deployment — replaces Ollama in two places

- **`ainbox` compose rig:** the `ollama` service is replaced by an
  `ainbox-infrastructure` service (GPU reservation, one public port). warden's
  `WARDEN_LLM_BASE_URL` repoints from `http://ollama.ainbox.local:11434` to the
  engine. Joins the `ainbox` bridge network with an alias (dropping the SuperBot
  engine's `network_mode: host`, which is incompatible with the rig's bridge +
  aliases).
- **`ainbox-os`:** the baked Ollama + `ainbox-llm-panel` are replaced by this
  engine image + its gateway UI, in both **desktop** (single-user, localhost,
  `systemd --user`) and **server** (multi-user, behind Caddy + warden) modes per
  the 2026-06-09 deployment-modes design. The raise-spec is the per-machine
  model selection the OS ships/first-boot-configures.

Delivery (the `ainbox-fetch` / registry story) is unchanged in principle: the
image (with baked models) rides the existing OCI blob-cache fetch path. Baking
models into the image **removes** the separate Ollama-model fetch driver — one
fetch path (the image) instead of two. A large baked model should sit in its own
layer so the blob-cache delta still moves only what changed.

## Reserved modalities (spec-only)

Designed into the interface now, built later. Each is "add a backend + wire its
route"; nothing about the gateway, spec, or round-robin changes.

- **TTS — `POST /v1/audio/speech`.** A speech runtime (candidate: Kokoro for
  quality/size, Piper for lightweight, XTTS for cloning — decided in its own
  spec). Baked voices declared in the recipe; raised via a `tts` block in the
  raise-spec. Pairs with whisper so the suite can speak, not only listen.
- **Image generation — `POST /v1/images/generations`.** FLUX (schnell/dev) via
  a diffusion runtime (ComfyUI or `diffusers`) fronted to the OpenAI images
  shape. Weights baked via the recipe; raised via an `images` block. Adds a 4th
  GPU runtime — exactly the "N backends, one API" case this design is built for.

## Migration from current SuperBot engine

| Current (`superbot-infraestructure`) | Target (`ainbox-infrastructure`) |
|---|---|
| `entrypoint.sh` bash launch-loop (raise all, health-poll) | gateway service: supervise + route + round-robin + UI |
| Per-model ports, `network_mode: host`, no single endpoint | one public OpenAI port; internal ports hidden; bridge network |
| `--embedding` on every `llama-server` | chat models plain; dedicated fastembed-GPU embedder |
| e5-small baked for embeddings | MiniLM-L12 ONNX baked (no reindex) |
| whisper `/v1/audio/transcriptions/{model}` | standard `/v1/audio/transcriptions` (model in body) |
| `deploy/*.json` (how to run) | raise-spec (subset + replicas + params) |
| no replicas | `replicas` per slug → round-robin pools |
| README-driven, no UI | tiny UI edits raise-spec + relaunch |

## Testing

- **Gateway routing:** unit tests for slug→pool resolution, round-robin
  distribution across replicas, `GET /v1/models` aggregation, unknown-slug 404.
- **OpenAI conformance:** golden request/response for `/v1/chat/completions`
  (incl. SSE streaming passthrough), `/v1/embeddings` (384-d), and
  `/v1/audio/transcriptions` (multipart, model-in-body) against a raised set.
- **Embedding equivalence:** assert fastembed-GPU vectors match the stored
  CPU-MiniLM vectors within tolerance on a fixture corpus (guards "no reindex").
- **Raise/relaunch:** spec with `replicas: 2` raises two backends and balances;
  a spec edit + relaunch brings up the new set and tears down the old.
- **Hardware contract:** a `cuda` backend on a CPU host fails loudly at raise.

## Phases

1. **Gateway + raise-spec (LLM only).** Replace `entrypoint.sh` with the
   gateway: read raise-spec, supervise `llama-server` pools, serve
   `/v1/chat/completions` + `/v1/models`, round-robin replicas. One public port.
2. **Embeddings.** fastembed-GPU MiniLM backend + `/v1/embeddings`; drop e5 from
   the recipe; equivalence test vs stored vectors; remove unconditional
   `--embedding`.
3. **STT.** Standard `/v1/audio/transcriptions` behind the gateway.
4. **warden integration.** Repoint `WARDEN_LLM_BASE_URL`; turn warden's
   `/api/transcribe` + `/api/embed` into thin proxies to the engine.
5. **Tiny UI.** Edit raise-spec + relaunch; status view of raised pools.
6. **Deployment swap.** Replace the `ollama` service in ainbox compose; then the
   ainbox-os engine layer (desktop + server).
7. **(Later, own specs)** TTS `/v1/audio/speech`; FLUX `/v1/images/generations`.

## Open questions (for the plan)

- **Gateway or gateway+supervisor split?** One service (simplest, appliance-fit)
  vs supervisord + stateless proxy (better failure isolation). Default: one
  service; revisit if crash-coupling bites.
- **Round-robin policy:** plain RR vs least-outstanding-requests (better for
  uneven chat latencies). Default RR; measure.
- **Backend transport:** child processes under the gateway (like today) vs
  sibling containers. Default child processes (single image, single VRAM view).
- **TTS runtime choice** and **FLUX runtime choice** — each its own spec.
- **Baked-image size vs layering** so `ainbox-fetch`'s blob-delta stays cheap
  when only one model changes.
