#!/bin/bash
set -e

CONFIG_FILE="/app/config/superbot_config.json"
LLAMA_CONFIG_TEMP="/tmp/llama_runtime.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "[CRITICAL ERROR] Configuration file not found at: $CONFIG_FILE"
    exit 1
fi

PORT_LLM=$(jq -r '.server_ports.llm // 8080' "$CONFIG_FILE")
PORT_STT=$(jq -r '.server_ports.stt // 8001' "$CONFIG_FILE")

# Map URLs to local paths
jq '[.llama_node[] | . + {model: ("/models/" + (.url | split("/") | last))} | del(.url)]' "$CONFIG_FILE" > "$LLAMA_CONFIG_TEMP"

# Launch Llama.cpp (Background)
/app/llama-server --config-file "$LLAMA_CONFIG_TEMP" --port "$PORT_LLM" --host 0.0.0.0 &

# Launch Whisper API (Foreground)
uvicorn whisper_api:app --host 0.0.0.0 --port "$PORT_STT"