import io
import wave

from ainbox_gateway.spec import Spec, LlmNode, TtsNode
from ainbox_gateway.tts import build_synthesizers, _wav_bytes


class FakeSynth:
    def __init__(self, node):
        self.slug = node.slug

    def synthesize(self, text, voice=None):
        return f"wav:{text}:{voice}".encode()


def test_build_synthesizers_maps_slug():
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a")],
                tts=[TtsNode(slug="voice", model="kokoro")])
    s = build_synthesizers(spec, factory=FakeSynth)
    assert set(s) == {"voice"}
    assert s["voice"].synthesize("hi", "ef_dora") == b"wav:hi:ef_dora"


def test_build_synthesizers_empty_when_no_tts():
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a")])
    assert build_synthesizers(spec, factory=FakeSynth) == {}


def test_wav_bytes_is_valid_riff():
    data = _wav_bytes([0.0, 0.5, -0.5, 1.0], rate=24000)
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    with wave.open(io.BytesIO(data)) as w:
        assert w.getframerate() == 24000
        assert w.getnframes() == 4


def test_module_imports_without_kokoro():
    import importlib
    import ainbox_gateway.tts as m
    importlib.reload(m)
    assert hasattr(m, "KokoroSynthesizer")
