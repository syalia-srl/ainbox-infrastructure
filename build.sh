#!/bin/bash
# ==============================================================================
# SuperBot Build Orchestrator
# Purpose: Prepares the build context and triggers the Docker image creation.
# ==============================================================================
set -e

RECIPE_PATH=$1
BUILD_DIR="./build"

# 1. Input Validation
if [ -z "$RECIPE_PATH" ] || [ ! -f "$RECIPE_PATH" ]; then
    echo "[ERROR] Usage: make image RECIPE=path/to/recipe.json"
    exit 1
fi

# build.sh modification
IMAGE_TAG=${2:-$(basename "$RECIPE_PATH" .json)} # Use 2nd arg or fallback to filename

# 2. Context Integrity Check
# README.md is no longer required for the build context
REQUIRED_FILES=("Dockerfile" "entrypoint.sh" "pyproject.toml")
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$BUILD_DIR/$f" ]; then
        echo "[ERROR] Missing required file in $BUILD_DIR: $f"
        exit 1
    fi
done

# 3. Recipe Injection: Renames the specific recipe to a generic name for Docker
cp "$RECIPE_PATH" "$BUILD_DIR/recipe.json"

# 3b. Gateway package staging: the pure-OpenAI front door lives at the repo
# root (pyproject.toml + src/), outside the build/ context, so stage a copy in.
rm -rf "$BUILD_DIR/gateway"
mkdir -p "$BUILD_DIR/gateway"
cp pyproject.toml "$BUILD_DIR/gateway/pyproject.toml"
cp -r src "$BUILD_DIR/gateway/src"

echo "[FACTORY] Starting build for superbot:$IMAGE_TAG"

# 4. Build Execution — derive per-recipe build knobs, then build (BuildKit).
# CUDA_TAG (env, optional) picks the devel base; the runtime base is derived
# from it. Default 12.2.2 keeps Ada/rtx4060 on their floor; Blackwell hosts
# export CUDA_TAG=12.8.1-devel-ubuntu22.04.
CUDA_TAG="${CUDA_TAG:-12.2.2-devel-ubuntu22.04}"
CUDA_RUNTIME_TAG="${CUDA_TAG/devel/runtime}"

command -v jq >/dev/null || { echo "[ERROR] jq is required on the build host"; exit 1; }
_has() { [ "$(jq "(.$1 // []) | length" "$RECIPE_PATH")" -gt 0 ]; }
WITH_LLAMA=0; _has llama_node && WITH_LLAMA=1
EXTRAS=""
_has whisper_nodes   && EXTRAS="$EXTRAS,stt"
_has embedding_nodes && EXTRAS="$EXTRAS,embeddings"
_has tts_nodes       && EXTRAS="$EXTRAS,tts"
_has image_nodes     && EXTRAS="$EXTRAS,images"
EXTRAS="${EXTRAS#,}"

echo "[FACTORY] llama=$WITH_LLAMA  extras='${EXTRAS:-<core-only>}'  cuda=$CUDA_TAG  runtime=$CUDA_RUNTIME_TAG"

DOCKER_BUILDKIT=1 docker build --progress=plain \
    --build-arg CUDA_TAG="$CUDA_TAG" \
    --build-arg CUDA_RUNTIME_TAG="$CUDA_RUNTIME_TAG" \
    --build-arg WITH_LLAMA="$WITH_LLAMA" \
    --build-arg EXTRAS="$EXTRAS" \
    -t "superbot:$IMAGE_TAG" \
    -f "$BUILD_DIR/Dockerfile" \
    "$BUILD_DIR"

# 5. Cleanup
rm "$BUILD_DIR/recipe.json"
rm -rf "$BUILD_DIR/gateway"