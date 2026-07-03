# ainbox-infrastructure — tasks

## Done (2026-07-02)

Engine feature-complete: one pure-OpenAI gateway over 5 in-process/subprocess
backends, all TDD (67 tests), shipped to `main`.

- ✅ LLM (`llama.cpp` subprocess pools, round-robin) — `/v1/chat/completions`, `/v1/completions`, `/v1/models`
- ✅ Embeddings (fastembed-GPU MiniLM, **no reindex**) — `/v1/embeddings`
- ✅ STT (faster-whisper) — `/v1/audio/transcriptions`
- ✅ TTS (Kokoro-82M) — `/v1/audio/speech`
- ✅ Image-gen (FLUX.1-schnell via diffusers) — `/v1/images/generations`
- ✅ Tiny admin UI — view/edit raise-spec + in-process relaunch + status
- Plans: `docs/2026-07-02-ainbox-infrastructure-*-plan.md`; design: `docs/2026-07-02-ainbox-infrastructure-design.md`

## Next — for tomorrow

### 1. Installer/shell recipe-selection + warden default-wiring (the end-phase)

The "prefer this engine, ask which recipe" work. **Cross-repo** (ainbox, ainbox-os, warden). Key constraints (already in the design doc's warden section):

- **warden stays provider-agnostic** — keeps `WARDEN_LLM_BASE_URL` / `WARDEN_LLM_API_KEY` / `WARDEN_LLM_MODEL` for any OpenAI endpoint. This engine is the *preferred default*, never a hardcode.
- The **"prefer local engine + ask which recipe to download" logic lives in `install.sh` / `install.ps1` / the ainbox desktop shell**, not in warden.
- warden's `/api/transcribe` + `/api/embed` become **thin proxies** to the engine's `/v1/audio/transcriptions` + `/v1/embeddings` (preserves the whole `warden-shared-ml-service` client migration — apps unchanged).
- Deployment swap: replace the `ollama` service in the ainbox compose rig (repoint `WARDEN_LLM_BASE_URL`); then the ainbox-os engine layer (desktop + server modes). See design doc "Deployment" section.
- Needs its own brainstorm → spec → plan (spans repos).

### 2. Pending GPU smoke (not runnable on zion)

Everything is unit-tested against fakes; the real backends need a GPU host. Run `docs/smoke-gateway.md` on a 4060-class box:

- Real `make image RECIPE=recipes/rtx4060_v1.json` build (llama.cpp compile + bake all models — LARGE, esp. FLUX layer).
- LLM round-robin, `/v1/embeddings` 384-d + **MiniLM CPU-vs-GPU equivalence** (< 1e-3 — guards no-reindex), STT, Kokoro WAV, FLUX PNG.
- **Finalize FLUX fp8 checkpoint + quant config** (deferred from zion — the `DiffusersFluxGenerator` currently loads bf16; wire the fp8 path here).

## Notes

- Repo ships straight to `main` (AInBox suite convention).
- Dev: `uv venv .venv && uv pip install '.[dev]'`; run `.venv/bin/python -m pytest`.
- Local UI preview (fake backends, no GPU): `PYTHONPATH=src .venv/bin/python -m scripts.demo_ui` → http://127.0.0.1:8080

## Deployed (2026-07-02) — engine.syalia.dev (CPU demo)

Real minimal engine live at **https://engine.syalia.dev** (TLS via Caddy on demos).
- Image `ainbox-engine:cpu-min` (1.1 GB, `Dockerfile.cpu`, static llama.cpp + faster-whisper), **built on demos** (fast net), not zion.
- Raised: `qwen3.5-0.8b` (LLM, CPU) + `whisper-tiny` (STT). Spec: `deploy/cpu_min.json`.
- Container: `--memory=1400m --memory-swap=2400m` (protects the ainbox prod stack), `--restart unless-stopped`, `127.0.0.1:8080`; demos has a 3 GB swapfile.
- Redeploy: `rsync` repo → `demos:/tmp/engine-build`, `./build-cpu.sh recipes/cpu_min.json cpu-min`, `docker run … ainbox-engine:cpu-min`.
- Working chat needs `"chat_template_kwargs":{"enable_thinking":false}` (0.8B is a verbose reasoner). TODO: add `disable_thinking` to the LLM raise-spec/argv so it's the default.
