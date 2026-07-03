"""Assemble + run the build/push shell steps, buffering output for SSE."""
from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass


def parse_docker_root(out) -> str:
    """Trim `docker info -f {{.DockerRootDir}}` output to the path string."""
    if isinstance(out, (bytes, bytearray)):
        out = out.decode(errors="replace")
    return out.strip()


def disk_free_gb(path: str) -> dict:
    """Free/total/used (GB, 1-decimal) of the filesystem holding `path`.

    `used` is `total - free` so the three numbers are self-consistent for a
    readout (ignores root-reserved blocks).
    """
    u = shutil.disk_usage(path)
    gb = 1024 ** 3
    return {"path": path,
            "free_gb": round(u.free / gb, 1),
            "total_gb": round(u.total / gb, 1),
            "used_gb": round((u.total - u.free) / gb, 1)}


async def docker_disk(spawn) -> dict:
    """Where Docker writes images, and how much room is left there."""
    proc = await spawn(["docker", "info", "-f", "{{.DockerRootDir}}"], None, None)
    out = b""
    async for line in proc.stdout:
        out += line
    await proc.wait()
    return disk_free_gb(parse_docker_root(out) or "/")


@dataclass
class Step:
    label: str
    argv: list[str]
    env: dict[str, str] | None = None


def build_command(name: str, cuda_tag: str, registry: str, push: bool) -> list[Step]:
    steps = [Step(label="build",
                  argv=["make", "image", f"RECIPE=recipes/{name}.json"],
                  env={"CUDA_TAG": cuda_tag})]
    if push:
        ref = f"{registry}/ainbox-infra/{name}:latest"
        steps.append(Step(label="tag", argv=["docker", "tag", f"superbot:{name}", ref]))
        steps.append(Step(label="push", argv=["docker", "push", ref]))
    return steps


class LogBuffer:
    def __init__(self):
        self._lines: list[str] = []
        self._event = asyncio.Event()
        self._closed = False

    def append(self, line: str) -> None:
        self._lines.append(line)
        self._event.set()

    def close(self) -> None:
        self._closed = True
        self._event.set()

    async def stream(self):
        i = 0
        while True:
            while i < len(self._lines):
                yield self._lines[i]
                i += 1
            if self._closed:
                return
            self._event.clear()
            await self._event.wait()


async def _default_spawn(argv, env, cwd):
    return await asyncio.create_subprocess_exec(
        *argv, env={**os.environ, **(env or {})}, cwd=cwd,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)


class BuildRunner:
    def __init__(self, steps, cwd, spawn, log: LogBuffer):
        self.steps, self.cwd, self._spawn, self.log = steps, cwd, spawn, log
        self.status = "building"
        self.exit_code = 0
        self.task: asyncio.Task | None = None

    async def run(self):
        try:
            for step in self.steps:
                self.log.append(f"$ [{step.label}] {' '.join(step.argv)}")
                proc = await self._spawn(step.argv, step.env, self.cwd)
                async for raw in proc.stdout:
                    self.log.append(raw.decode(errors="replace").rstrip("\n"))
                code = await proc.wait()
                if code != 0:
                    self.status, self.exit_code = "failed", code
                    self.log.append(f"[{step.label}] exited {code}")
                    return
            self.status, self.exit_code = "done", 0
        finally:
            self.log.close()


class BuildBusy(RuntimeError):
    pass


class BuildManager:
    def __init__(self, cwd, spawn=None):
        self.cwd = cwd
        self._spawn = spawn or _default_spawn
        self._runners: dict[str, BuildRunner] = {}
        self._n = 0

    def _current(self) -> BuildRunner | None:
        for r in self._runners.values():
            if r.status == "building":
                return r
        return None

    def start(self, steps) -> str:
        if self._current():
            raise BuildBusy("a build is already running")
        self._n += 1
        bid = f"b{self._n}"
        runner = BuildRunner(steps, self.cwd, self._spawn, LogBuffer())
        runner.task = asyncio.create_task(runner.run())
        self._runners[bid] = runner
        return bid

    def get(self, bid) -> BuildRunner | None:
        return self._runners.get(bid)
