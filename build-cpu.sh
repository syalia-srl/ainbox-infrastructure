#!/bin/bash
# CPU-slim build orchestrator (mirrors build.sh, uses Dockerfile.cpu).
# Usage: ./build-cpu.sh recipes/cpu_min.json [tag]
set -e
RECIPE_PATH=$1
BUILD_DIR="./build"
[ -z "$RECIPE_PATH" ] || [ ! -f "$RECIPE_PATH" ] && { echo "[ERROR] usage: ./build-cpu.sh recipes/x.json"; exit 1; }
IMAGE_TAG=${2:-$(basename "$RECIPE_PATH" .json)-cpu}

cp "$RECIPE_PATH" "$BUILD_DIR/recipe.json"
rm -rf "$BUILD_DIR/gateway"; mkdir -p "$BUILD_DIR/gateway"
cp pyproject.toml "$BUILD_DIR/gateway/pyproject.toml"
cp -r src "$BUILD_DIR/gateway/src"

echo "[FACTORY] building ainbox-engine:$IMAGE_TAG (CPU-slim)"
docker build --progress=plain -t "ainbox-engine:$IMAGE_TAG" -f "$BUILD_DIR/Dockerfile.cpu" "$BUILD_DIR"

rm "$BUILD_DIR/recipe.json"; rm -rf "$BUILD_DIR/gateway"
echo "[FACTORY] done: ainbox-engine:$IMAGE_TAG"
