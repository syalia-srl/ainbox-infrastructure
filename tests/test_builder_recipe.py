import pytest
from ainbox_builder.recipe import render_recipe, RecipeError


def test_render_recipe_full():
    sel = {
        "llm": [{"alias": "gemma4-e4b", "url": "https://hf/gemma.gguf"},
                {"alias": "qwen3-14b", "url": "https://hf/qwen.gguf"}],
        "stt": [{"alias": "fast_stt", "model": "tiny"}],
        "embeddings": [{"model": "paraphrase-multilingual-MiniLM-L12-v2"}],
    }
    assert render_recipe(sel) == {
        "whisper_nodes": [{"model": "tiny", "alias": "fast_stt"}],
        "embedding_nodes": [{"model": "paraphrase-multilingual-MiniLM-L12-v2"}],
        "tts_nodes": [],
        "image_nodes": [],
        "llama_node": [
            {"url": "https://hf/gemma.gguf", "alias": "gemma4-e4b"},
            {"url": "https://hf/qwen.gguf", "alias": "qwen3-14b"},
        ],
    }


def test_render_recipe_llm_only():
    sel = {"llm": [{"alias": "a", "url": "https://hf/a.gguf"}]}
    out = render_recipe(sel)
    assert out["llama_node"] == [{"url": "https://hf/a.gguf", "alias": "a"}]
    assert out["whisper_nodes"] == [] and out["embedding_nodes"] == []


def test_render_recipe_llm_less_ok_with_stt():
    out = render_recipe({"stt": [{"alias": "x", "model": "tiny"}]})
    assert out["llama_node"] == [] and out["whisper_nodes"] == [{"model": "tiny", "alias": "x"}]


def test_render_recipe_rejects_fully_empty():
    with pytest.raises(RecipeError):
        render_recipe({"llm": [], "stt": []})
