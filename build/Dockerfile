# syntax=docker/dockerfile:1
# ============================================================================
# Multi-stage: a heavy CUDA-devel *builder* compiles llama.cpp; a slim CUDA-
# *runtime* image carries only what the recipe actually needs. build.sh derives
# the CUDA tags + WITH_LLAMA + EXTRAS from the recipe JSON and passes them here.
# ============================================================================

# ---- Stage 1: builder — compile llama.cpp (only when the recipe has an LLM) --
# CUDA_TAG default 12.2.2 keeps Ada/rtx4060 on their known-good floor; Blackwell
# (sm_120 RTX 5090) hosts pass CUDA_TAG=12.8.1-devel-ubuntu22.04.
ARG CUDA_TAG=12.2.2-devel-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG} AS builder

ENV DEBIAN_FRONTEND=noninteractive
ARG WITH_LLAMA=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake libcurl4-openssl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Compile llama.cpp into /out (server binary + its shared libs). LLM-less
# recipes (WITH_LLAMA=0) skip the whole compile and leave /out empty.
RUN mkdir -p /out/libs && \
    if [ "$WITH_LLAMA" = "1" ]; then \
      git clone --depth 1 https://github.com/ggerganov/llama.cpp.git && \
      cd llama.cpp && mkdir build && cd build && \
      cmake .. -DGGML_CUDA=ON \
               -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
               -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" && \
      cmake --build . --config Release -j"$(nproc)" --target llama-server && \
      find . -name "lib*.so*" -exec cp -a {} /out/libs/ \; && \
      cp bin/llama-server /out/llama-server; \
    else echo "[BUILD] LLM-less recipe — skipping llama.cpp compile"; fi

# ---- Stage 2: runtime — slim base, only the recipe's backends ---------------
ARG CUDA_RUNTIME_TAG=12.2.2-runtime-ubuntu22.04
FROM nvidia/cuda:${CUDA_RUNTIME_TAG} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/lib

# Runtime-only system deps: python + jq + curl, espeak-ng (kokoro phonemizer),
# libgomp1 (llama-server OpenMP), libcurl4 (llama.cpp curl). No compilers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl jq espeak-ng libgomp1 libcurl4 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /tmp/install_uv.sh
RUN chmod +x /tmp/install_uv.sh && \
    UV_INSTALL_DIR="/usr/local/bin" INSTALLER_NO_MODIFY_PATH=1 /tmp/install_uv.sh && \
    rm /tmp/install_uv.sh

WORKDIR /app

# Pull llama-server + libs from the builder — present only when WITH_LLAMA=1.
# The bind mount keeps the install conditional on what the builder produced.
RUN --mount=type=bind,from=builder,source=/out,target=/builder-out \
    if [ -f /builder-out/llama-server ]; then \
      cp /builder-out/llama-server /app/llama-server && \
      cp -a /builder-out/libs/. /usr/lib/ && ldconfig && \
      echo "[BUILD] llama-server installed"; \
    else echo "[BUILD] no llama-server (LLM-less image)"; fi

# Gateway + supervisor. EXTRAS (from build.sh) selects backend deps, e.g.
# "stt,embeddings"; empty → core only (no torch/diffusers/whisper/fastembed).
ARG EXTRAS=""
COPY gateway /app/gateway
RUN if [ -n "$EXTRAS" ]; then uv pip install --system "/app/gateway[$EXTRAS]"; \
    else uv pip install --system /app/gateway; fi

COPY recipe.json /tmp/recipe.json

# Bake model weights — each loop is a no-op when its recipe array is empty, and
# its Python deps are present only if the matching EXTRA was installed above.
RUN mkdir -p /models && \
    jq -c '.llama_node[]?' /tmp/recipe.json | while read -r node; do \
        url=$(echo "$node" | jq -r '.url'); alias=$(echo "$node" | jq -r '.alias'); \
        if [ "$url" != "null" ] && [ -n "$alias" ]; then \
            filename=$(basename "$url" | cut -d'?' -f1); \
            echo "[BUILD] Downloading $filename for alias $alias..."; \
            curl -L -f "$url" -o "/models/$filename" --retry 5; \
            ln -s "/models/$filename" "/models/${alias}.gguf"; \
        fi; \
    done

RUN jq -r '.whisper_nodes[]?.model' /tmp/recipe.json | while read -r model; do \
    if [ -n "$model" ] && [ "$model" != "null" ]; then \
        echo "[BUILD] Baking Whisper: $model"; \
        python3 -c "from faster_whisper import WhisperModel; WhisperModel('$model', device='cpu')"; \
    fi; \
done

RUN jq -r '.embedding_nodes[]?.model' /tmp/recipe.json | while read -r model; do \
    if [ -n "$model" ] && [ "$model" != "null" ]; then \
        echo "[BUILD] Baking embedding: $model"; \
        python3 -c "from fastembed import TextEmbedding; TextEmbedding(model_name='$model')"; \
    fi; \
done

RUN jq -r '.tts_nodes[]?.model' /tmp/recipe.json | while read -r model; do \
    if [ "$model" = "kokoro" ]; then \
        echo "[BUILD] Baking Kokoro TTS"; \
        python3 -c "from kokoro import KPipeline; KPipeline(lang_code='a')"; \
    fi; \
done

RUN jq -r '.image_nodes[]?.model' /tmp/recipe.json | while read -r model; do \
    if [ -n "$model" ] && [ "$model" != "null" ]; then \
        echo "[BUILD] Baking FLUX image model: $model"; \
        python3 -c "from diffusers import FluxPipeline; import torch; FluxPipeline.from_pretrained('$model', torch_dtype=torch.bfloat16)"; \
    fi; \
done

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh && rm /tmp/recipe.json && mkdir -p /app/config

ENTRYPOINT ["/app/entrypoint.sh"]
