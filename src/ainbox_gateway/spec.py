"""Raise-spec: which fixed subset of baked models to bring up, and how."""
from __future__ import annotations

from dataclasses import dataclass, field


class SpecError(ValueError):
    """The raise-spec is structurally invalid."""


@dataclass
class LoraSpec:
    file: str
    alias: str
    scale: float = 1.0


@dataclass
class LlmNode:
    slug: str
    replicas: int = 1
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    flash_attn: bool = False
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    loras: list[LoraSpec] = field(default_factory=list)


@dataclass
class EmbeddingsNode:
    slug: str
    model: str
    device: str = "cuda"


@dataclass
class SttNode:
    slug: str
    model: str
    device: str = "cuda"
    compute_type: str = "float16"


@dataclass
class TtsNode:
    slug: str
    model: str
    device: str = "cuda"
    lang_code: str = "a"
    voice: str = "af_heart"


@dataclass
class ImagesNode:
    slug: str
    model: str
    device: str = "cuda"
    offload: bool = False
    quant: str = "fp8"
    steps: int = 4
    guidance: float = 0.0


@dataclass
class Spec:
    gateway_port: int
    llm: list[LlmNode]
    embeddings: list[EmbeddingsNode] = field(default_factory=list)
    stt: list[SttNode] = field(default_factory=list)
    tts: list[TtsNode] = field(default_factory=list)
    images: list[ImagesNode] = field(default_factory=list)


def _load_node(raw: dict) -> LlmNode:
    if "slug" not in raw:
        raise SpecError("llm node missing required 'slug'")
    loras = [LoraSpec(**l) for l in raw.get("loras", [])]
    return LlmNode(
        slug=raw["slug"],
        replicas=raw.get("replicas", 1),
        n_ctx=raw.get("n_ctx", 4096),
        n_gpu_layers=raw.get("n_gpu_layers", -1),
        flash_attn=raw.get("flash_attn", False),
        cache_type_k=raw.get("cache_type_k", "f16"),
        cache_type_v=raw.get("cache_type_v", "f16"),
        loras=loras,
    )


def _load_embeddings(raw: dict) -> EmbeddingsNode:
    if "slug" not in raw or "model" not in raw:
        raise SpecError("embeddings node needs 'slug' and 'model'")
    return EmbeddingsNode(slug=raw["slug"], model=raw["model"],
                          device=raw.get("device", "cuda"))


def _load_stt(raw: dict) -> SttNode:
    if "slug" not in raw or "model" not in raw:
        raise SpecError("stt node needs 'slug' and 'model'")
    return SttNode(slug=raw["slug"], model=raw["model"],
                   device=raw.get("device", "cuda"),
                   compute_type=raw.get("compute_type", "float16"))


def _load_tts(raw: dict) -> TtsNode:
    if "slug" not in raw or "model" not in raw:
        raise SpecError("tts node needs 'slug' and 'model'")
    return TtsNode(slug=raw["slug"], model=raw["model"],
                   device=raw.get("device", "cuda"),
                   lang_code=raw.get("lang_code", "a"),
                   voice=raw.get("voice", "af_heart"))


def _load_images(raw: dict) -> ImagesNode:
    if "slug" not in raw or "model" not in raw:
        raise SpecError("images node needs 'slug' and 'model'")
    return ImagesNode(slug=raw["slug"], model=raw["model"],
                      device=raw.get("device", "cuda"),
                      offload=raw.get("offload", False),
                      quant=raw.get("quant", "fp8"),
                      steps=raw.get("steps", 4),
                      guidance=raw.get("guidance", 0.0))


def load_spec(data: dict) -> Spec:
    gateway = data.get("gateway")
    if not gateway or "port" not in gateway:
        raise SpecError("spec missing 'gateway.port'")
    raw_llm = data.get("llm") or []
    if not raw_llm:
        raise SpecError("spec must declare at least one 'llm' node")
    return Spec(
        gateway_port=gateway["port"],
        llm=[_load_node(n) for n in raw_llm],
        embeddings=[_load_embeddings(e) for e in data.get("embeddings", [])],
        stt=[_load_stt(s) for s in data.get("stt", [])],
        tts=[_load_tts(t) for t in data.get("tts", [])],
        images=[_load_images(i) for i in data.get("images", [])],
    )
