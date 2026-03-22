# Stage 1: Build environment with CUDA support
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# 1. System Dependencies (Capa estática)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl wget git build-essential jq cmake libcurl4-openssl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/* 

# 2. Python Package Manager (uv) (Capa estática)
ADD https://astral.sh/uv/install.sh /tmp/install_uv.sh
RUN chmod +x /tmp/install_uv.sh && \
    UV_INSTALL_DIR="/usr/local/bin" INSTALLER_NO_MODIFY_PATH=1 /tmp/install_uv.sh && \
    rm /tmp/install_uv.sh 

WORKDIR /app

# 3. Compile Llama.cpp (CAPA CONGELADA)
# Importante: No hay ningún comando COPY antes de esto. 
# Docker usará la caché de la compilación que ya hiciste.
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    mkdir build && cd build && \
    cmake .. -DGGML_CUDA=ON \
             -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
             -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" && \
    cmake --build . --config Release -j$(nproc) --target llama-server && \
    cp bin/llama-server /app/ && \
    cd /app && rm -rf llama.cpp [cite: 20, 21]

# --- A PARTIR DE AQUÍ DOCKER PROCESARÁ TUS CAMBIOS LOCALES ---

# 4. Application Dependencies (Se invalida si cambias pyproject.toml)
COPY pyproject.toml .
RUN uv pip install --system . 

# 5. Model Vault (Capa Pesada - Se invalida si cambias la receta)
COPY recipe.json /tmp/recipe.json
RUN mkdir -p /models && \
    jq -r '.llama_node[].url' /tmp/recipe.json | while read url; do \
        if [ -n "$url" ] && [ "$url" != "null" ]; then \
            # Limpieza del nombre para evitar errores con parámetros de URL
            filename=$(basename "$url" | cut -d'?' -f1); \
            if [ ! -f "/models/$filename" ]; then \
                echo "[BUILD] Baking GGUF model: $filename..."; \
                # Cambiado a curl -L para mayor estabilidad con redirecciones [cite: 27]
                curl -L "$url" -o "/models/$filename" --retry 5 --retry-delay 2 -C -; \
            fi; \
        fi; \
    done [cite: 23, 24, 25, 27]

# Pre-loading Whisper models
RUN WHISPER_MODEL=$(jq -r '.whisper_node.model' /tmp/recipe.json) && \
    if [ -n "$WHISPER_MODEL" ] && [ "$WHISPER_MODEL" != "null" ]; then \
        echo "[BUILD] Baking Whisper model: $WHISPER_MODEL..."; \
        python3 -c "from faster_whisper import WhisperModel; WhisperModel('$WHISPER_MODEL', device='cpu')"; \
    fi [cite: 30, 31, 32]

# 6. Runtime Assets (Tu código: lo que más cambia)
COPY whisper_api.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh && rm /tmp/recipe.json && mkdir -p /app/config [cite: 33]

ENTRYPOINT ["/app/entrypoint.sh"]