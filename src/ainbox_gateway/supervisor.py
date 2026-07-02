"""Turn raise-spec nodes into llama-server launch commands + port maps."""
from __future__ import annotations

import subprocess
import time
import urllib.request
from typing import Callable, Protocol

from .pool import Backend, Pool
from .spec import LlmNode, Spec

LLAMA_SERVER_BIN = "/app/llama-server"
MODELS_DIR = "/models"
LORAS_DIR = "/loras"


def assign_ports(spec: Spec, base: int = 9000) -> list[tuple[LlmNode, int]]:
    out: list[tuple[LlmNode, int]] = []
    port = base
    for node in spec.llm:
        for _ in range(node.replicas):
            out.append((node, port))
            port += 1
    return out


def llama_argv(
    node: LlmNode,
    port: int,
    bin: str = LLAMA_SERVER_BIN,
    models_dir: str = MODELS_DIR,
) -> list[str]:
    argv = [
        bin,
        "-m", f"{models_dir}/{node.slug}.gguf",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--alias", node.slug,
        "-c", str(node.n_ctx),
        "-ngl", str(node.n_gpu_layers),
        "--cache-type-k", node.cache_type_k,
        "--cache-type-v", node.cache_type_v,
    ]
    if node.flash_attn:
        argv += ["--flash-attn", "on"]
    if node.loras:
        scaled = ",".join(f"{LORAS_DIR}/{l.file}:{l.scale}" for l in node.loras)
        argv += ["--lora-scaled", scaled]
    return argv


def build_pools(spec: Spec, base: int = 9000) -> dict[str, Pool]:
    by_slug: dict[str, list[Backend]] = {}
    for node, port in assign_ports(spec, base=base):
        by_slug.setdefault(node.slug, []).append(
            Backend(slug=node.slug, base_url=f"http://127.0.0.1:{port}"))
    return {slug: Pool(slug, backends) for slug, backends in by_slug.items()}


class Supervisor(Protocol):
    def start(self, spec: Spec) -> dict[str, Pool]: ...
    def stop(self) -> None: ...


def _http_ready(base_url: str, retries: int = 60, delay: float = 1.0) -> None:
    for _ in range(retries):
        try:
            urllib.request.urlopen(f"{base_url}/v1/models", timeout=2)
            return
        except Exception:
            time.sleep(delay)
    raise RuntimeError(f"backend at {base_url} never became ready")


class LlamaSupervisor:
    """Spawns one llama-server per replica; tears them all down on stop."""

    def __init__(self, spawn: Callable = subprocess.Popen,
                 wait_ready: Callable[[str], None] = _http_ready):
        self._spawn = spawn
        self._wait_ready = wait_ready
        self._procs: list = []

    def start(self, spec: Spec) -> dict[str, Pool]:
        for node, port in assign_ports(spec):
            self._procs.append(self._spawn(llama_argv(node, port)))
            self._wait_ready(f"http://127.0.0.1:{port}")
        return build_pools(spec)

    def stop(self) -> None:
        for p in self._procs:
            p.terminate()
        for p in self._procs:
            p.wait(timeout=10)
        self._procs = []
