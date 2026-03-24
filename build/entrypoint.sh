#!/bin/bash
# ==============================================================================
# SuperBot Runtime Entrypoint - Universal Orchestrator
# Purpose: Launches LLM nodes and Multi-Whisper STT service.
# ==============================================================================
set -e

CONFIG_FILE="/app/config/superbot_config.json"

# 1. Validation: Ensure the runtime configuration is present
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[CRITICAL] Execution config missing at: $CONFIG_FILE"
    exit 1
fi

echo "[SYSTEM] Initializing SuperBot Engine..."

# ==============================================================================
# 2. HARDWARE CONTRACT VALIDATION
# ==============================================================================
echo "[SYSTEM] Performing Hardware Capabilities Check..."

if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    HAS_GPU="true"
    echo "[SYSTEM] Environment: NVIDIA GPU Detected."
else
    HAS_GPU="false"
    echo "[SYSTEM] Environment: CPU-Only."
    ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/libcuda.so.1
fi

# Validation for Whisper Nodes
jq -c '.whisper_nodes[]?' "$CONFIG_FILE" | while read -r node; do
    DEVICE=$(echo "$node" | jq -r '.device // "cpu"')
    if [ "$DEVICE" = "cuda" ] && [ "$HAS_GPU" = "false" ]; then
        echo "[CRITICAL] Hardware Mismatch: Whisper node requests 'cuda' on CPU host."
        exit 1
    fi
done

# ==============================================================================

# 3. Extract Global STT Port
PORT_STT=$(jq -r '.server_ports.stt // 8001' "$CONFIG_FILE")

# 4. LLM Layer: Sequential Launch
jq -c '.llama_node[]?' "$CONFIG_FILE" | while read -r node; do
    
    # --- 4A. PARAMETER EXTRACTION ---
    ALIAS=$(echo "$node" | jq -r '.alias')
    NODE_PORT=$(echo "$node" | jq -r '.port')
    CTX=$(echo "$node" | jq -r '.n_ctx // 2048')
    NGL=$(echo "$node" | jq -r '.n_gpu_layers // 0')
    BATCH=$(echo "$node" | jq -r '.n_batch // 512')
    FLASH=$(echo "$node" | jq -r '.flash_attn // false')
    D_THINK=$(echo "$node" | jq -r '.disable_thinking // false')
    CACHE_K=$(echo "$node" | jq -r '.cache_type_k // "f16"')
    CACHE_V=$(echo "$node" | jq -r '.cache_type_v // "f16"')
    THREADS=$(echo "$node" | jq -r '.threads // null')

    MODEL_PATH="/models/${ALIAS}.gguf"

    if [ ! -f "$MODEL_PATH" ] && [ ! -L "$MODEL_PATH" ]; then
        echo "[CRITICAL] Model '$ALIAS' not found at $MODEL_PATH"
        exit 1
    fi

    # CPU Safety Overrides
    if [ "$HAS_GPU" = "false" ]; then
        [ "$NGL" != "0" ] && { echo "[CRITICAL] GPU layers requested on CPU mode."; exit 1; }
        FLASH="false"
    fi

    echo "[LLM] Launching: $ALIAS on Port: $NODE_PORT"
    
    # --- 4D. COMMAND CONSTRUCTION ---
    CMD_ARGS=("-m" "$MODEL_PATH" "--port" "$NODE_PORT" "--host" "0.0.0.0" "--alias" "$ALIAS" "-c" "$CTX" "-b" "$BATCH" "-ngl" "$NGL" "--cache-type-k" "$CACHE_K" "--cache-type-v" "$CACHE_V" "--embedding")

    [ "$FLASH" = "true" ] && CMD_ARGS+=("--flash-attn" "on")
    [ "$THREADS" != "null" ] && CMD_ARGS+=("-t" "$THREADS")

    # Disable reasoning budget logic
    if [ "$D_THINK" = "true" ]; then
        CMD_ARGS+=("--reasoning-budget" "0" "--chat-template" "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}{% endfor %}{{'<|im_start|>assistant\n'}}")
    fi

    # Launch server in background and capture logs for the wait loop
    /app/llama-server "${CMD_ARGS[@]}" > "/var/log/${ALIAS}.log" 2>&1 &

    # --- 4E. ROBUST SEMAPHORE ---
    echo "      [WAIT] Polling 127.0.0.1:${NODE_PORT}..."
    
    MAX_RETRIES=60
    RETRY_COUNT=0
    
    while ! curl -s -f -o /dev/null "http://127.0.0.1:${NODE_PORT}/v1/models"; do
        if [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; then
            echo "[CRITICAL] Timeout: $ALIAS is not responding."
            echo "      [DEBUG] Check /var/log/${ALIAS}.log for llama-server errors."
            exit 1
        fi
        sleep 1
        RETRY_COUNT=$((RETRY_COUNT + 1))
    done
    
    echo "      [SUCCESS] $ALIAS is ready."
    echo "---------------------------------------------------"
done

# 5. STT Layer: FINAL EXECUTION
echo "[STT] Launching Whisper API on port $PORT_STT..."
exec uvicorn whisper_api:app --host 0.0.0.0 --port "$PORT_STT"