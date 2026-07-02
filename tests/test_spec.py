import pytest
from ainbox_gateway.spec import load_spec, Spec, LlmNode, LoraSpec, SpecError


def test_minimal_spec():
    spec = load_spec({"gateway": {"port": 8080},
                      "llm": [{"slug": "qwen3.5-2b"}]})
    assert isinstance(spec, Spec)
    assert spec.gateway_port == 8080
    assert spec.llm == [LlmNode(slug="qwen3.5-2b")]


def test_full_node_fields_and_loras():
    spec = load_spec({"gateway": {"port": 9000}, "llm": [{
        "slug": "qwen3.5-9b", "replicas": 2, "n_ctx": 8192,
        "n_gpu_layers": -1, "flash_attn": True,
        "cache_type_k": "q8_0", "cache_type_v": "q8_0",
        "loras": [{"file": "voice.gguf", "alias": "voice", "scale": 0.8}],
    }]})
    node = spec.llm[0]
    assert node.replicas == 2 and node.n_ctx == 8192 and node.flash_attn is True
    assert node.loras == [LoraSpec(file="voice.gguf", alias="voice", scale=0.8)]


def test_missing_gateway_port_raises():
    with pytest.raises(SpecError):
        load_spec({"llm": [{"slug": "x"}]})


def test_node_without_slug_raises():
    with pytest.raises(SpecError):
        load_spec({"gateway": {"port": 8080}, "llm": [{"n_ctx": 4096}]})


def test_empty_llm_raises():
    with pytest.raises(SpecError):
        load_spec({"gateway": {"port": 8080}, "llm": []})
