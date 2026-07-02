"""Launch the gateway UI with fake backends (no GPU) for local viewing."""
import uvicorn

from ainbox_gateway.app import create_app
from ainbox_gateway.spec import Spec, LlmNode, EmbeddingsNode, SttNode
from ainbox_gateway.supervisor import build_pools


class _FakeSup:
    def start(self, spec): return build_pools(spec)
    def stop(self): pass


class _FakeEmb:
    def __init__(self, n): self.slug = n.slug
    def embed(self, texts): return [[0.0] * 384 for _ in texts]


class _FakeTr:
    def __init__(self, n): self.slug = n.slug
    def transcribe(self, audio, language=None): return "(demo)"


RAW = {
    "gateway": {"port": 8080},
    "llm": [{"slug": "qwen3.5-9b", "replicas": 1, "n_ctx": 8192},
            {"slug": "qwen3.5-2b", "replicas": 2, "n_ctx": 4096}],
    "embeddings": [{"slug": "text-embedding-minilm",
                    "model": "paraphrase-multilingual-MiniLM-L12-v2"}],
    "stt": [{"slug": "whisper-small", "model": "small"}],
}


def main():
    spec = Spec(gateway_port=8080,
                llm=[LlmNode(slug="qwen3.5-9b", n_ctx=8192),
                     LlmNode(slug="qwen3.5-2b", replicas=2, n_ctx=4096)],
                embeddings=[EmbeddingsNode(slug="text-embedding-minilm",
                                           model="paraphrase-multilingual-MiniLM-L12-v2")],
                stt=[SttNode(slug="whisper-small", model="small")])
    app = create_app(spec, _FakeSup(), embedder_factory=_FakeEmb,
                     transcriber_factory=_FakeTr, spec_raw=RAW)
    uvicorn.run(app, host="127.0.0.1", port=8080)


if __name__ == "__main__":
    main()
