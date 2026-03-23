# Stage 1: Build environment with CUDA support
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# 1. System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl wget git build-essential jq cmake libcurl4-openssl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Python Package Manager (uv) - Global Installation
ADD https://astral.sh/uv/install.sh /tmp/install_uv.sh
RUN chmod +x /tmp/install_uv.sh && \
    UV_INSTALL_DIR="/usr/local/bin" INSTALLER_NO_MODIFY_PATH=1 /tmp/install_uv.sh && \
    rm /tmp/install_uv.sh

WORKDIR /app

# 3. Compile Llama.cpp (ACTUALIZADO PARA LIBRERÍAS DINÁMICAS - FIX)
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    mkdir build && cd build && \
    cmake .. -DGGML_CUDA=ON \
             -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
             -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" && \
    cmake --build . --config Release -j$(nproc) --target llama-server && \
    find . -name "lib*.so*" -exec cp -a {} /usr/lib/ \; && \
    ldconfig && \
    cp bin/llama-server /app/ && \
    cd /app && rm -rf llama.cpp 

# Forzamos la ruta de las librerías dinámicas
ENV LD_LIBRARY_PATH=/usr/lib

# --- TODO LO QUE CAMBIE DEBE IR AQUÍ ABAJO ---

# 4. Application Dependencies
COPY pyproject.toml .
RUN uv pip install --system .

# 5. Model Vault (Descarga con curl para evitar error 8 de wget)
# Step 5: Model Vault - Download and Symlink
COPY recipe.json /tmp/recipe.json
RUN mkdir -p /models && \
    jq -c '.llama_node[]' /tmp/recipe.json | while read -r node; do \
        url=$(echo "$node" | jq -r '.url'); \
        alias=$(echo "$node" | jq -r '.alias'); \
        if [ "$url" != "null" ] && [ -n "$alias" ]; then \
            # Get original filename from URL
            filename=$(basename "$url" | cut -d'?' -f1); \
            echo "[BUILD] Downloading $filename..."; \
            curl -L -f "$url" -o "/models/$filename" --retry 5; \
            # Create the symlink: /models/<alias>.gguf -> /models/<original_name>
            ln -s "/models/$filename" "/models/${alias}.gguf"; \
            echo "[BUILD] Symlink created: /models/${alias}.gguf -> $filename"; \
        fi; \
    done

# Pre-loading Whisper models
RUN WHISPER_MODEL=$(jq -r '.whisper_node.model' /tmp/recipe.json) && \
    if [ -n "$WHISPER_MODEL" ] && [ "$WHISPER_MODEL" != "null" ]; then \
        echo "[BUILD] Baking Whisper model: $WHISPER_MODEL..."; \
        python3 -c "from faster_whisper import WhisperModel; WhisperModel('$WHISPER_MODEL', device='cpu')"; \
    fi

# 6. Runtime Assets
COPY whisper_api.py . 
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh && rm /tmp/recipe.json && mkdir -p /app/config 

ENTRYPOINT ["/app/entrypoint.sh"]