import asyncio
import pytest
from ainbox_builder.builder import (
    Step, BuildRunner, BuildManager, BuildBusy, LogBuffer)


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


def _fake_spawn(script):
    """script: dict argv[0] -> (lines, code)."""
    async def spawn(argv, env, cwd):
        lines, code = script[argv[0]]
        return _FakeProc(lines, code)
    return spawn


@pytest.mark.asyncio
async def test_runner_streams_and_succeeds():
    log = LogBuffer()
    steps = [Step("build", ["make", "image", "RECIPE=recipes/x.json"])]
    runner = BuildRunner(steps, cwd=".", spawn=_fake_spawn({"make": (["a", "b"], 0)}), log=log)
    got = []
    async def collect():
        async for line in log.stream():
            got.append(line)
    task = asyncio.create_task(collect())
    await runner.run()
    await task
    assert runner.status == "done" and runner.exit_code == 0
    assert [g for g in got if g in ("a", "b")] == ["a", "b"]


@pytest.mark.asyncio
async def test_runner_stops_on_failure():
    log = LogBuffer()
    steps = [Step("build", ["make"]), Step("push", ["docker"])]
    runner = BuildRunner(steps, cwd=".",
                         spawn=_fake_spawn({"make": (["boom"], 2), "docker": (["nope"], 0)}),
                         log=log)
    await runner.run()
    assert runner.status == "failed" and runner.exit_code == 2


@pytest.mark.asyncio
async def test_manager_rejects_concurrent():
    slow = asyncio.Event()
    async def blocking_spawn(argv, env, cwd):
        class P:
            stdout = _FakeProc([], 0)
            async def wait(self_):
                await slow.wait()
                return 0
        return P()
    mgr = BuildManager(cwd=".", spawn=blocking_spawn)
    bid = mgr.start([Step("build", ["make"])])
    with pytest.raises(BuildBusy):
        mgr.start([Step("build", ["make"])])
    slow.set()
    await mgr.get(bid).task
