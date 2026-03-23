#!/bin/bash
# ==============================================================================
# SuperBot Runtime Entrypoint - Universal Orchestrator
# Purpose: Launches LLM nodes via symlinks and the Multi-Whisper STT service.
# ==============================================================================
set -e

CONFIG_FILE="/app/config/superbot_config.json"

# 1. Validation: Ensure the runtime configuration is present
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[CRITICAL] Execution config missing at: $CONFIG_FILE"
    echo "Please mount your JSON configuration to /app/config/superbot_config.json"
    exit 1
fi

echo "[SYSTEM] Initializing SuperBot Engine..."

# 2. Extract Network Settings
PORT_LLM=$(jq -r '.server_ports.llm // 8080' "$CONFIG_FILE")
PORT_STT=$(jq -r '.server_ports.stt // 8001' "$CONFIG_FILE")

# 3. LLM Layer: Launching nodes using Symlinks
# The Dockerfile created symlinks at /models/${alias}.gguf during build.
jq -c '.llama_node[]' "$CONFIG_FILE" | while read -r node; do
    ALIAS=$(echo "$node" | jq -r '.alias')
    CTX=$(echo "$node" | jq -r '.n_ctx // 2048')
    NGL=$(echo "$node" | jq -r '.n_gpu_layers // 0')

    # Strict check for the symlink identifier
    MODEL_PATH="/models/${ALIAS}.gguf"

    if [ -L "$MODEL_PATH" ] || [ -f "$MODEL_PATH" ]; then
        echo "[LLM] Launching Instance: $ALIAS"
        echo "      Path: $MODEL_PATH | GPU Layers: $NGL | Context: $CTX"
        
        # Start llama-server in background. --embedding is enabled by default.
        /app/llama-server \
            -m "$MODEL_PATH" \
            --port "$PORT_LLM" \
            --host 0.0.0.0 \
            --alias "$ALIAS" \
            -c "$CTX" \
            -ngl "$NGL" \
            --embedding &
    else
        echo "[CRITICAL] Required model alias '$ALIAS' not found at $MODEL_PATH"
        echo "Check if your build recipe matches your runtime alias."
        exit 1
    fi
done

# 4. STT Layer: Launching Multi-Model Whisper API
# The python service will internally load all models defined in 'whisper_nodes'
echo "[STT] Launching Multi-Whisper API on port $PORT_STT..."
exec uvicorn whisper_api:app --host 0.0.0.0 --port "$PORT_STT"