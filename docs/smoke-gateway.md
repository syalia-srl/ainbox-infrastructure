# Gateway integration smoke (manual, needs a GPU host + one baked GGUF)

The gateway's routing/round-robin/lifespan logic is fully covered by unit tests
(`pytest`). This runbook is the one path those can't exercise: a real
`llama-server` behind the gateway on a GPU box.

## Steps

1. **Build** an image whose recipe bakes a small model, e.g. `qwen3.5-2b`:

   ```bash
   make image RECIPE=recipes/rtx4060_v1.json
   ```

2. **Write a raise-spec** `deploy/smoke.json` (new gateway schema — `gateway.port`
   + `llm[].slug`, *not* the legacy `server_ports`/`llama_node`/`alias` shape):

   ```json
   {
     "gateway": { "port": 8080 },
     "llm": [
       { "slug": "qwen3.5-2b", "replicas": 2, "n_ctx": 4096, "n_gpu_layers": -1 }
     ]
   }
   ```

3. **Raise it** (the raise-spec is mounted to `/app/config/superbot_config.json`,
   which the gateway reads by default via `AINBOX_SPEC`):

   ```bash
   make run CONFIG=deploy/smoke.json TAG=rtx4060_v1 MODE=gpu
   ```

4. **List models** — expect the slug:

   ```bash
   curl -s localhost:8080/v1/models | jq
   # {"object":"list","data":[{"id":"qwen3.5-2b","object":"model","owned_by":"ainbox"}]}
   ```

5. **Chat completion** — expect an OpenAI-shaped response:

   ```bash
   curl -s localhost:8080/v1/chat/completions \
     -H 'content-type: application/json' \
     -d '{"model":"qwen3.5-2b","messages":[{"role":"user","content":"hi"}]}' | jq
   ```

6. **Round-robin check** — fire the chat call several times and confirm **both**
   replica processes (ports 9000 and 9001 inside the container) log traffic:

   ```bash
   docker logs superbot_engine_gpu 2>&1 | grep -E ':9000|:9001'
   ```

## Embeddings (no-reindex equivalence)

1. Raise a spec with an `embeddings` block (see `deploy/example.json`).
2. Length check:

   ```bash
   curl -s localhost:8080/v1/embeddings -H 'content-type: application/json' \
     -d '{"model":"text-embedding-minilm","input":["hola mundo"]}' | jq '.data[0].embedding | length'
   # 384
   ```

3. **No-reindex check (GPU):** embed a fixture string via this endpoint and
   compare against the same string embedded by the apps' current CPU fastembed
   MiniLM (`paraphrase-multilingual-MiniLM-L12-v2`). Cosine similarity must be
   ≈1.0 (diff < 1e-3), confirming stored magpie/superbot vectors stay valid.

## STT

1. Raise a spec with an `stt` block, e.g. `{"slug":"whisper-small","model":"small"}`.
2. Transcribe:

   ```bash
   curl -s localhost:8080/v1/audio/transcriptions \
     -F model=whisper-small -F file=@sample.wav | jq
   # {"text":"..."}
   ```

## TTS

1. Raise a spec with a `tts` block, e.g. `{"slug":"voice","model":"kokoro","lang_code":"e","voice":"ef_dora"}`.
2. Synthesize:

   ```bash
   curl -s localhost:8080/v1/audio/speech \
     -H 'content-type: application/json' \
     -d '{"model":"voice","input":"Hola, soy AInBox.","voice":"ef_dora"}' \
     -o out.wav && file out.wav
   # out.wav: RIFF (little-endian) data, WAVE audio
   ```

## Notes

- The legacy `deploy/*.json` files (`server_ports`, `whisper_nodes`,
  `llama_node[].alias`) predate the gateway and **do not parse** under the new
  `load_spec`. Migrating them to the `gateway`+`llm` schema (and folding
  `whisper_nodes` back in) is the STT-phase plan's job.
- Streaming: add `"stream": true` to the chat body; the gateway relays the SSE
  stream verbatim.
