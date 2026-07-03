"""FastAPI app: author a recipe, trigger a build, stream progress, push."""
from __future__ import annotations

import json
from pathlib import Path

import ainbox_gateway
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .builder import BuildBusy, BuildManager, build_command
from .catalog import CATALOG
from .recipe import RecipeError, render_recipe

_STATIC = Path(__file__).parent / "static"
_GATEWAY_STATIC = Path(ainbox_gateway.__file__).parent / "static" / "syalia-ui"


def create_app(repo_root: str, catalog: dict = CATALOG, spawn=None) -> FastAPI:
    app = FastAPI(title="ainbox-builder")
    manager = BuildManager(cwd=repo_root, spawn=spawn)

    @app.get("/api/catalog")
    async def get_catalog():
        return JSONResponse(catalog)

    @app.post("/api/recipe")
    async def post_recipe(request: Request):
        body = await request.json()
        try:
            return JSONResponse(render_recipe(body.get("selection") or {}))
        except RecipeError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @app.post("/api/build")
    async def post_build(request: Request):
        body = await request.json()
        try:
            recipe = render_recipe(body.get("selection") or {})
        except RecipeError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        name = body["name"]
        recipes_dir = Path(repo_root) / "recipes"
        recipes_dir.mkdir(exist_ok=True)
        (recipes_dir / f"{name}.json").write_text(json.dumps(recipe, indent=2))
        steps = build_command(name, body["cuda_tag"], body["registry"], bool(body["push"]))
        try:
            bid = manager.start(steps)
        except BuildBusy as e:
            return JSONResponse({"error": str(e)}, status_code=409)
        return JSONResponse({"build_id": bid})

    @app.get("/api/build/{bid}")
    async def build_status(bid: str):
        r = manager.get(bid)
        if not r:
            return JSONResponse({"error": "unknown build"}, status_code=404)
        return JSONResponse({"status": r.status, "exit_code": r.exit_code})

    @app.get("/api/build/{bid}/log")
    async def build_log(bid: str):
        r = manager.get(bid)
        if not r:
            return JSONResponse({"error": "unknown build"}, status_code=404)
        async def gen():
            async for line in r.log.stream():
                yield f"data: {line}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    app.mount("/syalia-ui", StaticFiles(directory=_GATEWAY_STATIC), name="syalia-ui")

    @app.get("/")
    async def index():
        return FileResponse(_STATIC / "build.html")

    return app
