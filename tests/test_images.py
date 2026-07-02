from ainbox_gateway.spec import Spec, LlmNode, ImagesNode
from ainbox_gateway.images import build_generators


class FakeGen:
    def __init__(self, node):
        self.slug = node.slug

    def generate(self, prompt, n=1, width=1024, height=1024):
        return [f"png:{prompt}:{width}x{height}:{i}".encode() for i in range(n)]


def test_build_generators_maps_slug():
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a")],
                images=[ImagesNode(slug="flux", model="m")])
    g = build_generators(spec, factory=FakeGen)
    assert set(g) == {"flux"}
    assert g["flux"].generate("cat", n=2) == [b"png:cat:1024x1024:0", b"png:cat:1024x1024:1"]


def test_build_generators_empty_when_no_images():
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a")])
    assert build_generators(spec, factory=FakeGen) == {}


def test_module_imports_without_diffusers():
    import importlib
    import ainbox_gateway.images as m
    importlib.reload(m)
    assert hasattr(m, "DiffusersFluxGenerator")
