#!/bin/bash
# ==============================================================================
# SuperBot Runtime Entrypoint - Symlink Based Execution
# ==============================================================================
set -e

CONFIG_FILE="/app/config/superbot_config.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "[CRITICAL] Configuration file missing: $CONFIG_FILE"
    exit 1
fi

echo "[SYSTEM] Starting SuperBot Services..."

PORT_LLM=$(jq -r '.server_ports.llm // 8080' "$CONFIG_FILE")
PORT_STT=$(jq -r '.server_ports.stt // 8001' "$CONFIG_FILE")

# 3. Launch LLM nodes via Symlinks
jq -c '.llama_node[]' "$CONFIG_FILE" | while read -r node; do
    ALIAS=$(echo "$node" | jq -r '.alias')
    CTX=$(echo "$node" | jq -r '.n_ctx // 2048')
    NGL=$(echo "$node" | jq -r '.n_gpu_layers // 0')

    # The Dockerfile guaranteed that this symlink exists
    MODEL_PATH="/models/${ALIAS}.gguf"

    if [ -L "$MODEL_PATH" ] || [ -f "$MODEL_PATH" ]; then
        echo "[LLM] Launching $ALIAS (GPU Layers: $NGL)"
        /app/llama-server \
            -m "$MODEL_PATH" \
            --port "$PORT_LLM" \
            --host 0.0.0.0 \
            --alias "$ALIAS" \
            -c "$CTX" \
            -ngl "$NGL" \
            --embedding &
    else
        echo "[CRITICAL] Model link for alias '$ALIAS' not found at $MODEL_PATH"
        exit 1
    fi
done

echo "[STT] Launching Whisper API on port $PORT_STT..."
uvicorn whisper_api:app --host 0.0.0.0 --port "$PORT_STT"