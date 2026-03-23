#!/bin/bash
# ==============================================================================
# SuperBot Runtime Entrypoint
# Purpose: Orchestrates service startup using the runtime configuration.
# ==============================================================================
set -e

CONFIG_FILE="/app/config/superbot_config.json"

# 1. Validation
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[CRITICAL] Runtime configuration not found at: $CONFIG_FILE"
    exit 1
fi

echo "[SYSTEM] Starting SuperBot Services..."

# 2. Extract Network Ports
PORT_LLM=$(jq -r '.server_ports.llm // 8080' "$CONFIG_FILE")
PORT_STT=$(jq -r '.server_ports.stt // 8001' "$CONFIG_FILE")

# 3. Launch LLM/Embedding nodes
# We added the --embedding flag to enable the /v1/embeddings endpoint
jq -c '.llama_node[]' "$CONFIG_FILE" | while read -r node; do
    MODEL_FILE=$(echo "$node" | jq -r '.model_file')
    ALIAS=$(echo "$node" | jq -r '.alias // "default"')
    CTX=$(echo "$node" | jq -r '.n_ctx // 2048')
    NGL=$(echo "$node" | jq -r '.n_gpu_layers // 0')

    FULL_PATH="/models/$MODEL_FILE"

    if [ -f "$FULL_PATH" ]; then
        echo "[LLM] Launching $ALIAS (Model: $MODEL_FILE, GPU Layers: $NGL)"
        /app/llama-server \
            -m "$FULL_PATH" \
            --port "$PORT_LLM" \
            --host 0.0.0.0 \
            --alias "$ALIAS" \
            -c "$CTX" \
            -ngl "$NGL" \
            --embedding &
    else
        echo "[ERROR] Model file '$MODEL_FILE' missing in /models/."
    fi
done

# 4. Launch Whisper API
echo "[STT] Launching Whisper API on port $PORT_STT..."
uvicorn whisper_api:app --host 0.0.0.0 --port "$PORT_STT"