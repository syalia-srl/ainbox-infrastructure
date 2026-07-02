"""Turn raise-spec nodes into llama-server launch commands + port maps."""
from __future__ import annotations

from typing import Protocol

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
