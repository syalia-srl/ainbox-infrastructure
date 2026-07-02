from ainbox_gateway.spec import Spec, LlmNode, EmbeddingsNode
from ainbox_gateway.embeddings import build_embedders


class FakeEmbedder:
    def __init__(self, node):
        self.slug = node.slug
        self.model = node.model

    def embed(self, texts):
        return [[float(len(t))] for t in texts]


def test_build_embedders_maps_slug_to_embedder():
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a")],
                embeddings=[EmbeddingsNode(slug="emb", model="MiniLM")])
    embedders = build_embedders(spec, factory=FakeEmbedder)
    assert set(embedders) == {"emb"}
    assert embedders["emb"].embed(["ab", "xyz"]) == [[2.0], [3.0]]


def test_build_embedders_empty_when_no_embeddings():
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a")])
    assert build_embedders(spec, factory=FakeEmbedder) == {}


def test_module_imports_without_fastembed():
    # Importing the module must not require fastembed to be installed.
    import importlib
    import ainbox_gateway.embeddings as m
    importlib.reload(m)
    assert hasattr(m, "FastEmbedEmbedder")
