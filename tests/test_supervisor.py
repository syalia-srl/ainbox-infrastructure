from ainbox_gateway.spec import Spec, LlmNode, LoraSpec
from ainbox_gateway.supervisor import assign_ports, llama_argv


def test_assign_ports_expands_replicas_contiguously():
    spec = Spec(gateway_port=8080, llm=[
        LlmNode(slug="a", replicas=2),
        LlmNode(slug="b", replicas=1),
    ])
    assigned = assign_ports(spec, base=9000)
    assert [(n.slug, p) for n, p in assigned] == [("a", 9000), ("a", 9001), ("b", 9002)]


def test_llama_argv_core_flags():
    argv = llama_argv(LlmNode(slug="qwen3.5-2b", n_ctx=4096, n_gpu_layers=-1), port=9000)
    assert argv[0] == "/app/llama-server"
    assert "-m" in argv and "/models/qwen3.5-2b.gguf" in argv
    assert "--port" in argv and "9000" in argv
    assert "--alias" in argv and "qwen3.5-2b" in argv
    assert argv[argv.index("-c") + 1] == "4096"
    assert argv[argv.index("-ngl") + 1] == "-1"


def test_llama_argv_never_emits_embedding():
    argv = llama_argv(LlmNode(slug="a"), port=9000)
    assert "--embedding" not in argv


def test_llama_argv_flash_attn_and_loras():
    node = LlmNode(slug="a", flash_attn=True,
                   loras=[LoraSpec(file="v.gguf", alias="v", scale=0.8)])
    argv = llama_argv(node, port=9001)
    assert argv[argv.index("--flash-attn") + 1] == "on"
    assert argv[argv.index("--lora-scaled") + 1] == "/loras/v.gguf:0.8"
