#!/bin/bash
set -e

RECIPE_PATH=$1
BUILD_DIR="./build"

# 1. Validation
if [ -z "$RECIPE_PATH" ] || [ ! -f "$RECIPE_PATH" ]; then
    echo "[ERROR] Usage: make image RECIPE=path/to/recipe.json"
    exit 1
fi

# 2. Extract TAG from filename
IMAGE_TAG=$(basename "$RECIPE_PATH" .json)

# 3. Context Integrity Check
# Ensuring all required assets are in the build directory
REQUIRED_FILES=("Dockerfile" "whisper_api.py" "entrypoint.sh" "pyproject.toml")
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$BUILD_DIR/$f" ]; then
        echo "[ERROR] Build context error: '$f' missing in $BUILD_DIR"
        exit 1
    fi
done

echo "[FACTORY] Starting build for superbot:$IMAGE_TAG"

# 4. Docker Build using the recipe as a build-context
docker build --progress=plain \
    -t "superbot:$IMAGE_TAG" \
    -f "$BUILD_DIR/Dockerfile" \
    --build-context recipe="$RECIPE_PATH" \
    "$BUILD_DIR"