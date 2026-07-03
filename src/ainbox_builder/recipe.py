"""Pure: turn a UI model selection into a build recipe dict."""
from __future__ import annotations


class RecipeError(ValueError):
    """The selection cannot form a valid recipe."""


def render_recipe(selection: dict) -> dict:
    llm = selection.get("llm") or []
    stt = selection.get("stt") or []
    emb = selection.get("embeddings") or []
    if not (llm or stt or emb):
        raise RecipeError("a recipe needs at least one model")
    return {
        "whisper_nodes": [{"model": n["model"], "alias": n["alias"]}
                          for n in selection.get("stt") or []],
        "embedding_nodes": [{"model": n["model"]}
                            for n in selection.get("embeddings") or []],
        "tts_nodes": [],
        "image_nodes": [],
        "llama_node": [{"url": n["url"], "alias": n["alias"]} for n in llm],
    }


_UNKNOWN_LLM_GB = 6.0


def estimate_image_gb(selection: dict, catalog: dict, base_gb: float,
                      dep_gb: dict | None = None) -> float:
    """Baked-image size: base floor + Σ selected model sizes + present-modality
    backend deps (each counted once, from `dep_gb`).

    Unknown LLMs (custom URLs not in the catalog) count as a coarse guess so the
    estimate stays a conservative warning rather than under-reporting.
    """
    dep_gb = dep_gb or {}
    total = float(base_gb)
    for kind in ("llm", "stt", "embeddings"):
        nodes = selection.get(kind) or []
        for node in nodes:
            entry = catalog.get(kind, {}).get(node.get("alias"))
            if entry and "gb" in entry:
                total += entry["gb"]
            elif kind == "llm":
                total += _UNKNOWN_LLM_GB
        if nodes and kind in dep_gb:
            total += dep_gb[kind]
    return round(total, 1)
