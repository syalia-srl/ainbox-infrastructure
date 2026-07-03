import json
from contextlib import asynccontextmanager
import httpx
import pytest
from asgi_lifespan import LifespanManager
from ainbox_builder.app import create_app


class _FakeProc:
    def __init__(self, lines, code):
        self._lines = [l.encode() + b"\n" for l in lines]
        self._code = code
        self.stdout = self
    def __aiter__(self):
        async def gen():
            for l in self._lines:
                yield l
        return gen()
    async def wait(self):
        return self._code


async def _ok_spawn(argv, env, cwd):
    return _FakeProc(["step ok"], 0)


@asynccontextmanager
async def _client(app):
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://b") as c:
            yield c


def _app(tmp_path):
    return create_app(repo_root=str(tmp_path), spawn=_ok_spawn)


@pytest.mark.asyncio
async def test_catalog(tmp_path):
    async with _client(_app(tmp_path)) as c:
        r = await c.get("/api/catalog")
        assert r.status_code == 200 and "llm" in r.json()


@pytest.mark.asyncio
async def test_recipe_valid_and_invalid(tmp_path):
    async with _client(_app(tmp_path)) as c:
        ok = await c.post("/api/recipe", json={"selection":
            {"llm": [{"alias": "a", "url": "https://hf/a.gguf"}]}})
        assert ok.status_code == 200 and ok.json()["llama_node"][0]["alias"] == "a"
        bad = await c.post("/api/recipe", json={"selection": {"stt": []}})
        assert bad.status_code == 400


@pytest.mark.asyncio
async def test_build_writes_recipe_and_runs(tmp_path):
    (tmp_path / "recipes").mkdir()
    async with _client(_app(tmp_path)) as c:
        r = await c.post("/api/build", json={
            "name": "t1", "cuda_tag": "12.8.1-devel-ubuntu22.04",
            "registry": "registry.syalia.dev", "push": False,
            "selection": {"llm": [{"alias": "a", "url": "https://hf/a.gguf"}]}})
        assert r.status_code == 200
        bid = r.json()["build_id"]
        written = json.loads((tmp_path / "recipes" / "t1.json").read_text())
        assert written["llama_node"][0]["url"] == "https://hf/a.gguf"
        body = (await c.get(f"/api/build/{bid}/log")).text
        assert "step ok" in body
        st = (await c.get(f"/api/build/{bid}")).json()
        assert st["status"] == "done"


@pytest.mark.asyncio
async def test_build_invalid_recipe_400(tmp_path):
    async with _client(_app(tmp_path)) as c:
        r = await c.post("/api/build", json={
            "name": "t2", "cuda_tag": "x", "registry": "r", "push": False,
            "selection": {"stt": []}})
        assert r.status_code == 400


@pytest.mark.asyncio
async def test_serves_page(tmp_path):
    async with _client(_app(tmp_path)) as c:
        r = await c.get("/")
        assert r.status_code == 200 and "Builder" in r.text
