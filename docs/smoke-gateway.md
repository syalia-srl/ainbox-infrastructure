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

## Notes

- The legacy `deploy/*.json` files (`server_ports`, `whisper_nodes`,
  `llama_node[].alias`) predate the gateway and **do not parse** under the new
  `load_spec`. Migrating them to the `gateway`+`llm` schema (and folding
  `whisper_nodes` back in) is the STT-phase plan's job.
- Streaming: add `"stream": true` to the chat body; the gateway relays the SSE
  stream verbatim.
