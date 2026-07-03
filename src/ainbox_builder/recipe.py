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
