"""In-process speech-to-text backends served at /v1/audio/transcriptions.

faster-whisper is imported lazily inside FasterWhisperTranscriber so this
module (and the whole gateway) imports and unit-tests without it installed.
"""
from __future__ import annotations

import os
import tempfile
from typing import Callable, Protocol

from .spec import SttNode, Spec


class Transcriber(Protocol):
    slug: str

    def transcribe(self, audio: bytes, language: str | None = None) -> str: ...


def build_transcribers(
    spec: Spec, factory: Callable[[SttNode], "Transcriber"]
) -> dict[str, "Transcriber"]:
    return {node.slug: factory(node) for node in spec.stt}


class FasterWhisperTranscriber:
    """Real transcriber over faster-whisper (CTranslate2; CUDA-capable)."""

    def __init__(self, node: SttNode):
        from faster_whisper import WhisperModel  # lazy

        self.slug = node.slug
        self._model = WhisperModel(
            node.model, device=node.device, compute_type=node.compute_type)

    def transcribe(self, audio: bytes, language: str | None = None) -> str:
        fd, path = tempfile.mkstemp(suffix=".audio")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(audio)
            segments, _ = self._model.transcribe(
                path, language=language, vad_filter=True)
            return " ".join(s.text for s in segments).strip()
        finally:
            if os.path.exists(path):
                os.remove(path)
