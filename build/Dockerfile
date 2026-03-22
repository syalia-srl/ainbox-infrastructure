# Stage 1: Build environment with CUDA support
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# 1. System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl wget git build-essential jq cmake libcurl4-openssl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/* [cite: 19]

# 2. Python Package Manager (uv) - Global Installation
ADD https://astral.sh/uv/install.sh /tmp/install_uv.sh
RUN chmod +x /tmp/install_uv.sh && \
    UV_INSTALL_DIR="/usr/local/bin" INSTALLER_NO_MODIFY_PATH=1 /tmp/install_uv.sh && \
    rm /tmp/install_uv.sh [cite: 19]

WORKDIR /app

# 3. Compile Llama.cpp (ESTA CAPA NO SE TOCA)
# Mantenemos exactamente tu estructura para que Docker use el CACHE 
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    mkdir build && cd build && \
    cmake .. -DGGML_CUDA=ON \
             -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
             -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" && \
    cmake --build . --config Release -j$(nproc) --target llama-server && \
    cp bin/llama-server /app/ && \
    cd /app && rm -rf llama.cpp 

# --- TODO LO QUE CAMBIE DEBE IR AQUÍ ABAJO ---

# 4. Application Dependencies
COPY pyproject.toml . [cite: 22]
RUN uv pip install --system . [cite: 22]

# 5. Model Vault (Descarga con curl para evitar error 8 de wget)
COPY recipe.json /tmp/recipe.json [cite: 23]
RUN mkdir -p /models && \
    jq -r '.llama_node[].url' /tmp/recipe.json | while read url; do \
        if [ -n "$url" ] && [ "$url" != "null" ]; then \
            filename=$(basename "$url" | cut -d'?' -f1); \
            if [ ! -f "/models/$filename" ]; then \
                echo "[BUILD] Baking GGUF model: $filename..."; \
                curl -L "$url" -o "/models/$filename" --retry 5 --retry-delay 2 -C -; \
            fi; \
        fi; \
    done [cite: 23, 24, 25, 26, 28, 29]

# Pre-loading Whisper models
RUN WHISPER_MODEL=$(jq -r '.whisper_node.model' /tmp/recipe.json) && \
    if [ -n "$WHISPER_MODEL" ] && [ "$WHISPER_MODEL" != "null" ]; then \
        echo "[BUILD] Baking Whisper model: $WHISPER_MODEL..."; \
        python3 -c "from faster_whisper import WhisperModel; WhisperModel('$WHISPER_MODEL', device='cpu')"; \
    fi [cite: 29, 30, 31, 32]

# 6. Runtime Assets
COPY whisper_api.py . 
COPY entrypoint.sh . 
RUN chmod +x entrypoint.sh && rm /tmp/recipe.json && mkdir -p /app/config 

ENTRYPOINT ["/app/entrypoint.sh"]