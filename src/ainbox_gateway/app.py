"""FastAPI gateway: one pure-OpenAI front door over routed backend pools."""
from __future__ import annotations

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from .router import Router, UnknownModel


def create_app(router: Router, client: httpx.AsyncClient) -> FastAPI:
    app = FastAPI(title="ainbox-infrastructure gateway")

    async def _proxy(request: Request, path: str) -> Response:
        body = await request.body()
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        model = payload.get("model")
        if not model:
            return JSONResponse({"error": "missing 'model'"}, status_code=400)
        try:
            backend = router.resolve(model)
        except UnknownModel:
            return JSONResponse(
                {"error": f"model '{model}' is not raised"}, status_code=404)

        upstream = client.build_request(
            "POST", f"{backend.base_url}{path}", content=body,
            headers={"content-type": "application/json"})
        resp = await client.send(upstream, stream=True)
        return StreamingResponse(
            resp.aiter_raw(),
            status_code=resp.status_code,
            headers={"content-type": resp.headers.get("content-type", "application/json")},
            background=BackgroundTask(resp.aclose),
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await _proxy(request, "/v1/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await _proxy(request, "/v1/completions")

    app.state.router = router
    app.state.client = client
    return app
