"""Curated model catalog for the builder UI. URLs are HF GGUF resolve links."""

_HF = "https://huggingface.co"

CATALOG = {
    "llm": {
        "gemma4-e4b":   {"url": f"{_HF}/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf?download=true", "size": "5.0 GB"},
        "gemma4-e2b":   {"url": f"{_HF}/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf?download=true", "size": "~3 GB"},
        "qwen3-14b":    {"url": f"{_HF}/unsloth/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q4_K_M.gguf?download=true", "size": "9.0 GB"},
        "qwen3.5-9b":   {"url": f"{_HF}/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf?download=true", "size": "~5.5 GB"},
        "qwen3.5-4b":   {"url": f"{_HF}/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf?download=true", "size": "~2.5 GB"},
        "qwen3.5-2b":   {"url": f"{_HF}/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf?download=true", "size": "~1.5 GB"},
        "qwen3.5-0.8b": {"url": f"{_HF}/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf?download=true", "size": "~0.6 GB"},
    },
    "stt": {
        "whisper-tiny":  {"model": "tiny"},
        "whisper-small": {"model": "small"},
    },
    "embeddings": {
        "minilm": {"model": "paraphrase-multilingual-MiniLM-L12-v2"},
    },
}
