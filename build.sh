#!/bin/bash
# ==============================================================================
# SuperBot Build Orchestrator
# Purpose: Prepares the build context and triggers the Docker image creation.
# ==============================================================================
set -e

RECIPE_PATH=$1
BUILD_DIR="./build"

# 1. Validation: Ensure the recipe file exists
if [ -z "$RECIPE_PATH" ] || [ ! -f "$RECIPE_PATH" ]; then
    echo "[ERROR] Usage: make image RECIPE=path/to/recipe.json"
    exit 1
fi

# 2. Extract Tag: Use the filename as the image tag
IMAGE_TAG=$(basename "$RECIPE_PATH" .json)

# 3. Structural Integrity: Verify that all factory assets are present
REQUIRED_FILES=("Dockerfile" "whisper_api.py" "entrypoint.sh" "pyproject.toml")
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$BUILD_DIR/$f" ]; then
        echo "[ERROR] Missing required file in $BUILD_DIR: $f"
        exit 1
    fi
done

# 4. Recipe Injection: Copy the selected recipe into the build context
# This resolves the 'not a directory' error by making the file available to COPY.
cp "$RECIPE_PATH" "$BUILD_DIR/recipe.json"

echo "[FACTORY] Starting build for superbot:$IMAGE_TAG"

# 5. Build Execution: Standard docker build without invalid build-context flags
docker build --progress=plain \
    -t "superbot:$IMAGE_TAG" \
    -f "$BUILD_DIR/Dockerfile" \
    "$BUILD_DIR"

# 6. Cleanup: Remove the temporary recipe from the build directory
rm "$BUILD_DIR/recipe.json"