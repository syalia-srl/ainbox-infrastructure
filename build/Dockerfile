# Stage 1: Build environment with CUDA support
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# 1. System Dependencies (Added cmake)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl wget git build-essential jq cmake \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Python Package Manager (uv)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /app

# 3. Compile Llama.cpp (Using CMake for CUDA)
# GGML_CUDA=ON is the new flag for NVIDIA acceleration
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    mkdir build && cd build && \
    cmake .. -DGGML_CUDA=ON && \
    cmake --build . --config Release -j$(nproc) --target llama-server && \
    cp bin/llama-server /app/ && \
    cd /app && rm -rf llama.cpp

# 4. Application Dependencies
COPY pyproject.toml .
RUN uv pip install --system .

# 5. Model Baking (Heavy Layer)
COPY recipe.json /tmp/recipe.json

RUN mkdir -p /models && \
    jq -r '.llama_node[].url' /tmp/recipe.json | while read url; do \
        if [ -n "$url" ] && [ "$url" != "null" ]; then \
            filename=$(basename "$url"); \
            if [ ! -f "/models/$filename" ]; then \
                echo "[BUILD] Downloading GGUF: $filename..."; \
                wget -q --show-progress --continue --tries=0 --retry-connrefused "$url" -O "/models/$filename"; \
            fi; \
        fi; \
    done

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