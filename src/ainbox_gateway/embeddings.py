"""In-process embedding backends served at /v1/embeddings.

fastembed is imported lazily inside FastEmbedEmbedder so this module (and the
whole gateway) imports and unit-tests without fastembed installed.
"""
from __future__ import annotations

from typing import Callable, Protocol

from .spec import EmbeddingsNode, Spec


class Embedder(Protocol):
    slug: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def build_embedders(
    spec: Spec, factory: Callable[[EmbeddingsNode], "Embedder"]
) -> dict[str, "Embedder"]:
    return {node.slug: factory(node) for node in spec.embeddings}


class FastEmbedEmbedder:
    """Real embedder over fastembed (ONNX; CUDA when device='cuda')."""

    def __init__(self, node: EmbeddingsNode):
        from fastembed import TextEmbedding  # lazy

        self.slug = node.slug
        providers = ["CUDAExecutionProvider"] if node.device == "cuda" else None
        self._model = TextEmbedding(model_name=node.model, providers=providers)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.embed(texts)]
