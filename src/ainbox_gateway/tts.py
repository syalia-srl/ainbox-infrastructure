"""In-process text-to-speech backends served at /v1/audio/speech.

kokoro is imported lazily inside KokoroSynthesizer so this module (and the
whole gateway) imports and unit-tests without it installed.
"""
from __future__ import annotations

import io
import wave
from typing import Callable, Protocol

from .spec import TtsNode, Spec


class Synthesizer(Protocol):
    slug: str

    def synthesize(self, text: str, voice: str | None = None) -> bytes: ...


def build_synthesizers(
    spec: Spec, factory: Callable[[TtsNode], "Synthesizer"]
) -> dict[str, "Synthesizer"]:
    return {node.slug: factory(node) for node in spec.tts}


def _wav_bytes(samples: list[float], rate: int = 24000) -> bytes:
    """Encode float samples in [-1, 1] as 16-bit mono PCM WAV."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for s in samples:
            v = int(max(-1.0, min(1.0, float(s))) * 32767)
            frames += int(v).to_bytes(2, "little", signed=True)
        w.writeframes(bytes(frames))
    return buf.getvalue()


class KokoroSynthesizer:
    """Real synthesizer over Kokoro-82M (24 kHz)."""

    def __init__(self, node: TtsNode):
        from kokoro import KPipeline  # lazy

        self.slug = node.slug
        self._default_voice = node.voice
        self._pipeline = KPipeline(lang_code=node.lang_code)

    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        import numpy as np

        chunks = [audio for _, _, audio in
                  self._pipeline(text, voice=voice or self._default_voice)]
        samples = np.concatenate(chunks) if chunks else np.zeros(0)
        return _wav_bytes(samples.tolist(), rate=24000)
