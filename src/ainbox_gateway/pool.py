"""A backend endpoint and a round-robin pool of same-slug replicas."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from itertools import cycle


@dataclass(frozen=True)
class Backend:
    slug: str
    base_url: str  # e.g. "http://127.0.0.1:9000"; no trailing slash


class Pool:
    def __init__(self, slug: str, backends: list[Backend]):
        if not backends:
            raise ValueError(f"pool '{slug}' has no backends")
        self.slug = slug
        self._backends = list(backends)
        self._cycle = cycle(self._backends)
        self._lock = threading.Lock()

    def next(self) -> Backend:
        with self._lock:
            return next(self._cycle)
