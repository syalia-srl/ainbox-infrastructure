"""Curated model catalog for the builder UI. URLs are HF GGUF resolve links.

`gb` is the on-disk GGUF size (Q4_K_M); it feeds the image-size estimate.
`BASE_IMAGE_GB` is the fixed overhead the Dockerfile bakes regardless of the
recipe: the CUDA-devel base + llama.cpp + the gateway[gpu] deps (torch/onnx).
Calibrate it against a real `docker images` size when convenient.
"""

_HF = "https://huggingface.co"

BASE_IMAGE_GB = 13.0

CATALOG = {
    "llm": {
        "gemma4-e4b":   {"url": f"{_HF}/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf?download=true", "size": "5.0 GB", "gb": 5.0},
        "gemma4-e2b":   {"url": f"{_HF}/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf?download=true", "size": "~3 GB", "gb": 3.0},
        "qwen3-14b":    {"url": f"{_HF}/unsloth/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q4_K_M.gguf?download=true", "size": "9.0 GB", "gb": 9.0},
        "qwen3.5-9b":   {"url": f"{_HF}/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf?download=true", "size": "~5.5 GB", "gb": 5.5},
        "qwen3.5-4b":   {"url": f"{_HF}/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf?download=true", "size": "~2.5 GB", "gb": 2.5},
        "qwen3.5-2b":   {"url": f"{_HF}/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf?download=true", "size": "~1.5 GB", "gb": 1.5},
        "qwen3.5-0.8b": {"url": f"{_HF}/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf?download=true", "size": "~0.6 GB", "gb": 0.6},
    },
    "stt": {
        "whisper-tiny":  {"model": "tiny", "gb": 0.08},
        "whisper-small": {"model": "small", "gb": 0.5},
    },
    "embeddings": {
        "minilm": {"model": "paraphrase-multilingual-MiniLM-L12-v2", "gb": 0.12},
    },
}
