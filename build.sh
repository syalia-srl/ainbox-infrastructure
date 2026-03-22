#!/bin/bash
# ==============================================================================
# SuperBot Build Orchestrator
# Purpose: Prepares context and triggers Docker build without unnecessary assets.
# ==============================================================================
set -e

RECIPE_PATH=$1
BUILD_DIR="./build"

# 1. Input Validation
if [ -z "$RECIPE_PATH" ] || [ ! -f "$RECIPE_PATH" ]; then
    echo "[ERROR] Usage: make image RECIPE=path/to/recipe.json"
    exit 1
fi

IMAGE_TAG=$(basename "$RECIPE_PATH" .json)

# 2. Context Integrity Check
# README.md is no longer required for the build context
REQUIRED_FILES=("Dockerfile" "whisper_api.py" "entrypoint.sh" "pyproject.toml")
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$BUILD_DIR/$f" ]; then
        echo "[ERROR] Missing required file in $BUILD_DIR: $f"
        exit 1
    fi
done

# 3. Recipe Injection
cp "$RECIPE_PATH" "$BUILD_DIR/recipe.json"

echo "[FACTORY] Starting build for superbot:$IMAGE_TAG"

# 4. Build Execution
docker build --progress=plain \
    -t "superbot:$IMAGE_TAG" \
    -f "$BUILD_DIR/Dockerfile" \
    "$BUILD_DIR"

# 5. Cleanup
rm "$BUILD_DIR/recipe.json"