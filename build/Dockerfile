# Stage 1: Build environment with CUDA support
# CUDA_TAG is overridable at build time (see build.sh). Default 12.2.2 keeps the
# Ada/rtx4060 appliance targets on their known-good driver floor; Blackwell hosts
# (e.g. sm_120 RTX 5090) pass CUDA_TAG=12.8.1-devel-ubuntu22.04 for native support.
ARG CUDA_TAG=12.2.2-devel-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG} AS builder

ENV DEBIAN_FRONTEND=noninteractive

# 1. System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl wget git build-essential jq cmake libcurl4-openssl-dev espeak-ng \
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

# 4. Application Dependencies
COPY pyproject.toml .
RUN uv pip install --system .

# 5. Model Vault - Optimized for Multi-model and Symlinks
COPY recipe.json /tmp/recipe.json

# Download LLM models and create symlinks based on Alias
RUN mkdir -p /models && \
    jq -c '.llama_node[]' /tmp/recipe.json | while read -r node; do \
        url=$(echo "$node" | jq -r '.url'); \
        alias=$(echo "$node" | jq -r '.alias'); \
        if [ "$url" != "null" ] && [ -n "$alias" ]; then \
            filename=$(basename "$url" | cut -d'?' -f1); \
            echo "[BUILD] Downloading $filename for alias $alias..."; \
            curl -L -f "$url" -o "/models/$filename" --retry 5; \
            ln -s "/models/$filename" "/models/${alias}.gguf"; \
            echo "[BUILD] Symlink created: /models/${alias}.gguf -> $filename"; \
        fi; \
    done

# Pre-loading MULTIPLE Whisper models defined in the recipe
RUN jq -r '.whisper_nodes[].model' /tmp/recipe.json | while read -r model; do \
    if [ -n "$model" ] && [ "$model" != "null" ]; then \
        echo "[BUILD] Baking Whisper model: $model..."; \
        python3 -c "from faster_whisper import WhisperModel; WhisperModel('$model', device='cpu')"; \
    fi; \
done

# 6. Gateway package — the pure-OpenAI front door + supervisor.
# Installed WITH the [gpu] extra (fastembed-gpu) so embeddings run on CUDA and
# so fastembed is present for the embedding-bake step below.
COPY gateway /app/gateway
RUN uv pip install --system '/app/gateway[gpu]'

# 6b. Pre-loading embedding models defined in the recipe (fastembed ONNX cache)
RUN jq -r '.embedding_nodes[]?.model' /tmp/recipe.json | while read -r model; do \
    if [ -n "$model" ] && [ "$model" != "null" ]; then \
        echo "[BUILD] Baking embedding model: $model..."; \
        python3 -c "from fastembed import TextEmbedding; TextEmbedding(model_name='$model')"; \
    fi; \
done

# 6c. Pre-loading TTS models defined in the recipe (Kokoro weights + voices)
RUN jq -r '.tts_nodes[]?.model' /tmp/recipe.json | while read -r model; do \
    if [ "$model" = "kokoro" ]; then \
        echo "[BUILD] Baking Kokoro TTS..."; \
        python3 -c "from kokoro import KPipeline; KPipeline(lang_code='a')"; \
    fi; \
done

# 6d. Pre-loading image-gen models (FLUX weights + encoders) — LARGE layer
RUN jq -r '.image_nodes[]?.model' /tmp/recipe.json | while read -r model; do \
    if [ -n "$model" ] && [ "$model" != "null" ]; then \
        echo "[BUILD] Baking FLUX image model: $model..."; \
        python3 -c "from diffusers import FluxPipeline; import torch; FluxPipeline.from_pretrained('$model', torch_dtype=torch.bfloat16)"; \
    fi; \
done

# 7. Runtime Assets
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh && rm /tmp/recipe.json && mkdir -p /app/config

ENTRYPOINT ["/app/entrypoint.sh"]