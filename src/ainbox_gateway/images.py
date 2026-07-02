"""In-process image-generation backends served at /v1/images/generations.

torch/diffusers are imported lazily inside DiffusersFluxGenerator so this
module (and the whole gateway) imports and unit-tests without them installed.
"""
from __future__ import annotations

import io
from typing import Callable, Protocol

from .spec import ImagesNode, Spec


class Generator(Protocol):
    slug: str

    def generate(self, prompt: str, n: int = 1,
                 width: int = 1024, height: int = 1024) -> list[bytes]: ...


def build_generators(
    spec: Spec, factory: Callable[[ImagesNode], "Generator"]
) -> dict[str, "Generator"]:
    return {node.slug: factory(node) for node in spec.images}


class DiffusersFluxGenerator:
    """Real generator over FLUX.1-schnell via diffusers.

    NOTE: not exercised on zion. The concrete fp8/quant loading path and the
    baked checkpoint are pinned in the GPU smoke (docs/smoke-gateway.md).
    """

    def __init__(self, node: ImagesNode):
        import torch  # lazy
        from diffusers import FluxPipeline  # lazy

        self.slug = node.slug
        self._steps = node.steps
        self._guidance = node.guidance
        dtype = torch.bfloat16
        self._pipe = FluxPipeline.from_pretrained(node.model, torch_dtype=dtype)
        if node.offload:
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe = self._pipe.to(node.device)

    def generate(self, prompt: str, n: int = 1,
                 width: int = 1024, height: int = 1024) -> list[bytes]:
        out: list[bytes] = []
        for _ in range(n):
            image = self._pipe(prompt, num_inference_steps=self._steps,
                               guidance_scale=self._guidance,
                               width=width, height=height).images[0]
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            out.append(buf.getvalue())
        return out
