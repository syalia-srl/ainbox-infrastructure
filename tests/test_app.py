from contextlib import asynccontextmanager

import httpx
import pytest
import respx
from asgi_lifespan import LifespanManager
from ainbox_gateway.app import create_app
from ainbox_gateway.spec import Spec, LlmNode
from ainbox_gateway.supervisor import build_pools


class FakeSupervisor:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self, spec):
        self.started = True
        return build_pools(spec, base=9000)

    def stop(self):
        self.stopped = True


def _app():
    # slug "a", 2 replicas -> backends at 127.0.0.1:9000 and :9001
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a", replicas=2)])
    return create_app(spec, FakeSupervisor())


@asynccontextmanager
async def _client(app):
    """Run the app's lifespan (populates app.state.router) around the client.

    httpx.ASGITransport does not emit ASGI lifespan events on its own, so the
    startup hook must be driven explicitly via LifespanManager.
    """
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
            yield c


@pytest.mark.asyncio
@respx.mock
async def test_chat_completion_routes_and_round_robins():
    route0 = respx.post("http://127.0.0.1:9000/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"id": "0"}))
    route1 = respx.post("http://127.0.0.1:9001/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"id": "1"}))
    async with _client(_app()) as c:
        r0 = await c.post("/v1/chat/completions", json={"model": "a", "messages": []})
        r1 = await c.post("/v1/chat/completions", json={"model": "a", "messages": []})
    assert r0.json()["id"] == "0" and r1.json()["id"] == "1"
    assert route0.called and route1.called


@pytest.mark.asyncio
@respx.mock
async def test_unknown_model_returns_404():
    async with _client(_app()) as c:
        r = await c.post("/v1/chat/completions", json={"model": "nope", "messages": []})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_missing_model_returns_400():
    async with _client(_app()) as c:
        r = await c.post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_models_endpoint_lists_slugs():
    async with _client(_app()) as c:
        r = await c.get("/v1/models")
    body = r.json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["a"]
    assert body["data"][0]["object"] == "model"


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_supervisor():
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a", replicas=2)])
    sup = FakeSupervisor()
    app = create_app(spec, sup)
    async with LifespanManager(app):
        assert sup.started
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
            r = await c.get("/v1/models")
            assert [m["id"] for m in r.json()["data"]] == ["a"]
    assert sup.stopped
