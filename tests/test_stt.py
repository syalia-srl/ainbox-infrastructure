from ainbox_gateway.spec import Spec, LlmNode, SttNode
from ainbox_gateway.stt import build_transcribers


class FakeTranscriber:
    def __init__(self, node):
        self.slug = node.slug

    def transcribe(self, audio, language=None):
        return f"transcribed:{len(audio)}:{language}"


def test_build_transcribers_maps_slug():
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a")],
                stt=[SttNode(slug="w", model="small")])
    ts = build_transcribers(spec, factory=FakeTranscriber)
    assert set(ts) == {"w"}
    assert ts["w"].transcribe(b"1234", "es") == "transcribed:4:es"


def test_build_transcribers_empty_when_no_stt():
    spec = Spec(gateway_port=8080, llm=[LlmNode(slug="a")])
    assert build_transcribers(spec, factory=FakeTranscriber) == {}


def test_module_imports_without_faster_whisper():
    import importlib
    import ainbox_gateway.stt as m
    importlib.reload(m)
    assert hasattr(m, "FasterWhisperTranscriber")
