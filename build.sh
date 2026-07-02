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

# 4. Build Execution: Standard docker build
docker build --progress=plain \
    -t "superbot:$IMAGE_TAG" \
    -f "$BUILD_DIR/Dockerfile" \
    "$BUILD_DIR"

# 5. Cleanup
rm "$BUILD_DIR/recipe.json"
rm -rf "$BUILD_DIR/gateway"