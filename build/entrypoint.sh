#!/bin/bash
# ==============================================================================
# Script: entrypoint.sh
# Purpose: Primary orchestration for the AI Engine (PID 1).
#
# Description:
#   This script reads the unified 'recipe.json' mounted at runtime. It performs 
#   two critical tasks:
#   1. Maps model URLs to their local paths in the '/models' directory.
#   2. Extracts network ports and launches Llama.cpp and Whisper API.
#
# Constraints:
#   - No environment variables are used for configuration.
#   - Strictly offline: it assumes all models were baked during the build phase.
# ==============================================================================

set -e

# Path where the single source of truth is mounted
CONFIG_FILE="/app/config/superbot_config.json"
# Temporary file for llama-server (requires an array-only JSON)
LLAMA_CONFIG_TEMP="/tmp/llama_runtime.json"

echo "========================================"
echo "   Starting SuperBot AI Engine...       "
echo "========================================"

# ------------------------------------------------------------------------------
# 1. Validation
# ------------------------------------------------------------------------------
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[CRITICAL ERROR] Configuration file not found at: $CONFIG_FILE"
    exit 1
fi

# ------------------------------------------------------------------------------
# 2. Port Parsing
# ------------------------------------------------------------------------------
# Extracting ports for service binding
PORT_LLM=$(jq -r '.server_ports.llm // 8080' "$CONFIG_FILE")
PORT_STT=$(jq -r '.server_ports.stt // 8001' "$CONFIG_FILE")

# ------------------------------------------------------------------------------
# 3. Llama.cpp Configuration Mapping
# ------------------------------------------------------------------------------
# We transform the 'llama_node' array: 
# For each object, we take the 'url', extract the filename, and create the 
# 'model' key pointing to the local path. We then remove the 'url' key.
jq '[.llama_node[] | . + {model: ("/models/" + (.url | split("/") | last))} | del(.url)]' "$CONFIG_FILE" > "$LLAMA_CONFIG_TEMP"

echo "[INFO] Network Configuration:"
echo "       - Llama.cpp Port : $PORT_LLM"
echo "       - Whisper Port   : $PORT_STT"
echo "[INFO] Generated runtime config for Llama.cpp at $LLAMA_CONFIG_TEMP"

# ------------------------------------------------------------------------------
# 4. Launch Services
# ------------------------------------------------------------------------------

# Launch Llama.cpp server in background
echo "[INFO] Launching Llama.cpp server..."
/app/llama-server \
    --config-file "$LLAMA_CONFIG_TEMP" \
    --port "$PORT_LLM" \
    --host 0.0.0.0 &

# Launch Whisper FastAPI server in foreground
# The Python script 'whisper_api.py' will read $CONFIG_FILE directly
echo "[INFO] Launching Whisper API server..."
uvicorn whisper_api:app \
    --host 0.0.0.0 \
    --port "$PORT_STT"