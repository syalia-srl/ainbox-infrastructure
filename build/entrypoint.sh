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

# ==============================================================================
# 2. HARDWARE CONTRACT VALIDATION (NEW)
# ==============================================================================
echo "[SYSTEM] Performing Hardware Capabilities Check..."

# Detect if a real GPU is accessible (NVIDIA drivers mounted via docker profile)
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    HAS_GPU="true"
    echo "[SYSTEM] Environment: NVIDIA GPU Detected."
else
    HAS_GPU="false"
    echo "[SYSTEM] Environment: CPU-Only (or GPU drivers missing)."
    ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/libcuda.so.1
fi

# 2A. Validate LLM requests against reality
jq -c '.llama_node[]?' "$CONFIG_FILE" | while read -r node; do
    NGL=$(echo "$node" | jq -r '.n_gpu_layers // 0')
    ALIAS=$(echo "$node" | jq -r '.alias')
    if [ "$NGL" -gt 0 ] && [ "$HAS_GPU" = "false" ]; then
        echo "[CRITICAL] Hardware Mismatch Abort!"
        echo "Reason: LLM node '$ALIAS' requests $NGL GPU layers, but the container is running in CPU mode."
        exit 1
    fi
done

# 2B. Validate Whisper requests against reality
jq -c '.whisper_nodes[]?' "$CONFIG_FILE" | while read -r node; do
    DEVICE=$(echo "$node" | jq -r '.device // "cpu"')
    MODEL=$(echo "$node" | jq -r '.model')
    if [ "$DEVICE" = "cuda" ] && [ "$HAS_GPU" = "false" ]; then
        echo "[CRITICAL] Hardware Mismatch Abort!"
        echo "Reason: Whisper node '$MODEL' requests 'cuda' device, but the container is running in CPU mode."
        exit 1
    fi
done

echo "[SYSTEM] Hardware validation passed. Applying constraints..."
# ==============================================================================

# 3. Extract Network Settings
PORT_LLM=$(jq -r '.server_ports.llm // 8080' "$CONFIG_FILE")
PORT_STT=$(jq -r '.server_ports.stt // 8001' "$CONFIG_FILE")

# 4. LLM Layer: Launching nodes using Symlinks
jq -c '.llama_node[]?' "$CONFIG_FILE" | while read -r node; do
    ALIAS=$(echo "$node" | jq -r '.alias')
    CTX=$(echo "$node" | jq -r '.n_ctx // 2048')
    NGL=$(echo "$node" | jq -r '.n_gpu_layers // 0')

    MODEL_PATH="/models/${ALIAS}.gguf"

    if [ -L "$MODEL_PATH" ] || [ -f "$MODEL_PATH" ]; then
        echo "[LLM] Launching Instance: $ALIAS"
        echo "      Path: $MODEL_PATH | GPU Layers: $NGL | Context: $CTX"
        
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
        exit 1
    fi
done

# 5. STT Layer: Launching Multi-Model Whisper API
echo "[STT] Launching Multi-Whisper API on port $PORT_STT..."
exec uvicorn whisper_api:app --host 0.0.0.0 --port "$PORT_STT"