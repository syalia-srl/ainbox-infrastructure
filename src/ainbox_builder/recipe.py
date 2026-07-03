"""Pure: turn a UI model selection into a build recipe dict."""
from __future__ import annotations


class RecipeError(ValueError):
    """The selection cannot form a valid recipe."""


def render_recipe(selection: dict) -> dict:
    llm = selection.get("llm") or []
    if not llm:
        raise RecipeError("a recipe needs at least one LLM")
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


def estimate_image_gb(selection: dict, catalog: dict, base_gb: float) -> float:
    """Rough baked-image size: fixed base overhead + Σ selected model sizes.

    Unknown LLMs (custom URLs not in the catalog) count as a coarse guess so
    the estimate stays a conservative warning rather than under-reporting.
    """
    total = float(base_gb)
    for kind in ("llm", "stt", "embeddings"):
        for node in selection.get(kind) or []:
            entry = catalog.get(kind, {}).get(node.get("alias"))
            if entry and "gb" in entry:
                total += entry["gb"]
            elif kind == "llm":
                total += _UNKNOWN_LLM_GB
    return round(total, 1)
