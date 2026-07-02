"""Resolve an OpenAI `model` slug to a backend via its round-robin pool."""
from __future__ import annotations

from .pool import Backend, Pool


class UnknownModel(KeyError):
    """No pool serves the requested model slug."""


class Router:
    def __init__(self, pools: dict[str, Pool]):
        self._pools = pools

    def resolve(self, model: str) -> Backend:
        pool = self._pools.get(model)
        if pool is None:
            raise UnknownModel(model)
        return pool.next()

    def models(self) -> list[str]:
        return sorted(self._pools)
