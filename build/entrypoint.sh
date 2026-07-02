#!/bin/bash
# ==============================================================================
# ainbox-infrastructure Runtime Entrypoint
# Purpose: Detect hardware, then hand off to the ainbox gateway, which reads the
#          raise-spec, supervises the llama-server pools, and serves the single
#          pure-OpenAI /v1/* front door.
# ==============================================================================
set -e

CONFIG_FILE="/app/config/superbot_config.json"

# 1. Validation: Ensure the raise-spec is present
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[CRITICAL] Raise-spec missing at: $CONFIG_FILE"
    exit 1
fi

echo "[SYSTEM] Initializing ainbox inference engine..."

# ==============================================================================
# 2. HARDWARE CAPABILITIES CHECK
# ==============================================================================
echo "[SYSTEM] Performing Hardware Capabilities Check..."

if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "[SYSTEM] Environment: NVIDIA GPU Detected."
else
    echo "[SYSTEM] Environment: CPU-Only."
    ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/libcuda.so.1
fi

# ==============================================================================
# 3. Hand off to the gateway (reads AINBOX_SPEC, default = $CONFIG_FILE)
# ==============================================================================
echo "[SYSTEM] Launching ainbox gateway..."
exec python3 -m ainbox_gateway
