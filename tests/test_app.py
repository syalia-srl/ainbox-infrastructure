import httpx
import pytest
import respx
from ainbox_gateway.app import create_app
from ainbox_gateway.pool import Backend, Pool
from ainbox_gateway.router import Router


def _app_and_client():
    pools = {"a": Pool("a", [Backend("a", "http://h:9000"),
                             Backend("a", "http://h:9001")])}
    upstream = httpx.AsyncClient()
    return create_app(Router(pools), upstream)


@pytest.mark.asyncio
@respx.mock
async def test_chat_completion_routes_and_round_robins():
    route0 = respx.post("http://h:9000/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"id": "0"}))
    route1 = respx.post("http://h:9001/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"id": "1"}))
    app = _app_and_client()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
        r0 = await c.post("/v1/chat/completions", json={"model": "a", "messages": []})
        r1 = await c.post("/v1/chat/completions", json={"model": "a", "messages": []})
    assert r0.json()["id"] == "0" and r1.json()["id"] == "1"
    assert route0.called and route1.called


@pytest.mark.asyncio
@respx.mock
async def test_unknown_model_returns_404():
    app = _app_and_client()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
        r = await c.post("/v1/chat/completions", json={"model": "nope", "messages": []})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_missing_model_returns_400():
    app = _app_and_client()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
        r = await c.post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_models_endpoint_lists_slugs():
    app = _app_and_client()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
        r = await c.get("/v1/models")
    body = r.json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["a"]
    assert body["data"][0]["object"] == "model"
