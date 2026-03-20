# ==============================================================================
# Dockerfile: SuperBot AI Engine
# Stage 1: Build environment con soporte CUDA
# ==============================================================================
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS builder

# Evitar prompts interactivos durante la instalación
ENV DEBIAN_FRONTEND=noninteractive

# 1. Dependencias del Sistema
# Se incluye libcurl4-openssl-dev para soporte de red en llama-server si fuera necesario
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    curl \
    wget \
    git \
    build-essential \
    jq \
    cmake \
    libcurl4-openssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Instalación de Python Package Manager (uv)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# 3. Compilación de Llama.cpp (Capa Estática)
# SOLUCIÓN AL ERROR DE LINKER: 
# -DCMAKE_LIBRARY_PATH apunta a los stubs de CUDA para resolver cuGetErrorString, etc.
# -DCMAKE_EXE_LINKER_FLAGS permite símbolos no definidos que el driver resolverá en runtime.
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    mkdir build && cd build && \
    cmake .. -DGGML_CUDA=ON \
             -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
             -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" && \
    cmake --build . --config Release -j$(nproc) --target llama-server && \
    cp bin/llama-server /app/ && \
    cd /app && rm -rf llama.cpp

# 4. Dependencias de la Aplicación (Whisper API)
COPY pyproject.toml .
RUN uv pip install --system .

# 5. Model Vault (Capa Pesada - Descarga de modelos)
# Se asume que 'recipe.json' se inyecta mediante --build-context en el comando de build
COPY recipe.json /tmp/recipe.json

RUN mkdir -p /models && \
    jq -r '.llama_node[].url' /tmp/recipe.json | while read url; do \
        if [ -n "$url" ] && [ "$url" != "null" ]; then \
            filename=$(basename "$url"); \
            if [ ! -f "/models/$filename" ]; then \
                echo "[BUILD] Horneando modelo GGUF: $filename..."; \
                wget -q --show-progress --continue --tries=0 --retry-connrefused "$url" -O "/models/$filename"; \
            fi; \
        fi; \
    done

# Pre-descarga de modelos Whisper para caché offline
RUN WHISPER_MODEL=$(jq -r '.whisper_node.model' /tmp/recipe.json) && \
    if [ -n "$WHISPER_MODEL" ] && [ "$WHISPER_MODEL" != "null" ]; then \
        echo "[BUILD] Horneando modelo Whisper: $WHISPER_MODEL..."; \
        python3 -c "from faster_whisper import WhisperModel; WhisperModel('$WHISPER_MODEL', device='cpu')"; \
    fi

# ------------------------------------------------------------------------------
# 6. Runtime Assets (Capa Volátil)
# ------------------------------------------------------------------------------
COPY whisper_api.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Limpieza de manifiesto temporal
RUN rm /tmp/recipe.json

# Exponer puertos por defecto (serán sobreescritos por el config runtime)
EXPOSE 8080 8001

ENTRYPOINT ["./entrypoint.sh"]