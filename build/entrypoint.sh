#!/bin/bash
# ==============================================================================
# SuperBot Runtime Entrypoint
# Purpose: Orchestrates service startup using the runtime configuration.
# This script is strictly for EXECUTION and remains static inside the image.
# ==============================================================================
set -e

CONFIG_FILE="/app/config/superbot_config.json"

# 1. Check if the runtime configuration exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[CRITICAL] Runtime configuration not found at: $CONFIG_FILE"
    echo "Please mount a valid JSON configuration to /app/config/superbot_config.json"
    exit 1
fi

echo "[SYSTEM] Starting SuperBot Services..."

# 2. Extract Network Ports (Defaults: LLM=8080, STT=8001)
PORT_LLM=$(jq -r '.server_ports.llm // 8080' "$CONFIG_FILE")
PORT_STT=$(jq -r '.server_ports.stt // 8001' "$CONFIG_FILE")

# 3. Launch LLM nodes defined in the runtime config
# It looks for files already present in /models/ by 'model_file' name.
jq -c '.llama_node[]' "$CONFIG_FILE" | while read -r node; do
    MODEL_FILE=$(echo "$node" | jq -r '.model_file')
    ALIAS=$(echo "$node" | jq -r '.alias // "default"')
    CTX=$(echo "$node" | jq -r '.n_ctx // 2048')
    NGL=$(echo "$node" | jq -r '.n_gpu_layers // 0')

    FULL_PATH="/models/$MODEL_FILE"

    if [ -f "$FULL_PATH" ]; then
        echo "[LLM] Launching $ALIAS (Model: $MODEL_FILE, GPU Layers: $NGL)"
        # Launch llama-server in background
        /app/llama-server \
            -m "$FULL_PATH" \
            --port "$PORT_LLM" \
            --host 0.0.0.0 \
            --alias "$ALIAS" \
            -c "$CTX" \
            -ngl "$NGL" &
    else
        echo "[ERROR] Model file '$MODEL_FILE' missing in /models/ directory."
    fi
done

# 4. Launch Whisper API (STT)
# The Python service handles its own hardware logic (CPU/CUDA) from the JSON.
echo "[STT] Launching Whisper API on port $PORT_STT..."
uvicorn whisper_api:app --host 0.0.0.0 --port "$PORT_STT"