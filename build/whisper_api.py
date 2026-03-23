"""
SuperBot Multi-Model Whisper API Microservice.
Optimized for local inference, air-gapped environments, and dynamic model routing.
"""

import os
import json
import logging
import tempfile
from typing import Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from faster_whisper import WhisperModel

# Setup professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("whisper-api")

app = FastAPI(
    title="SuperBot Multi-Whisper API",
    description="Asynchronous STT service supporting multiple concurrent models.",
    version="2.0.0"
)

# Shared configuration path
CONFIG_FILE = "/app/config/superbot_config.json"

# In-memory storage for loaded models
# Key: model_name (e.g., 'tiny', 'small'), Value: WhisperModel instance
loaded_models: Dict[str, WhisperModel] = {}

@app.on_event("startup")
async def startup_event():
    """
    Initializes and pre-loads all Whisper models defined in the 
    runtime configuration during service startup.
    """
    if not os.path.exists(CONFIG_FILE):
        logger.warning(f"Configuration file not found at {CONFIG_FILE}. No models loaded.")
        return

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            full_config = json.load(f)
            
            # The system now expects 'whisper_nodes' as a list for generality
            whisper_nodes = full_config.get("whisper_nodes", [])
            
            for node in whisper_nodes:
                model_name = node.get("model")
                device = node.get("device", "cpu")  # Dynamic hardware allocation
                compute_type = node.get("compute_type", "int8")
                
                if not model_name:
                    continue

                logger.info(f"Loading model: [{model_name}] on device: [{device}] ({compute_type})")
                
                # local_files_only=True ensures no external calls if models were baked into the image
                loaded_models[model_name] = WhisperModel(
                    model_name,
                    device=device,
                    compute_type=compute_type,
                    local_files_only=True
                )
        
        logger.info(f"Successfully loaded {len(loaded_models)} Whisper models.")
    except Exception as e:
        logger.error(f"Failed to initialize models from config: {e}")

@app.post("/v1/audio/transcriptions/{model_name}", response_model=Dict[str, str])
async def transcribe_audio(model_name: str, file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Routes an audio file to a specific model for transcription.
    
    Args:
        model_name (str): The name of the model to use (must be loaded at startup).
        file (UploadFile): Audio file (.wav, .mp3, .ogg, .m4a).
        
    Returns:
        Dict[str, str]: {"text": "The transcribed content"}
    """
    if model_name not in loaded_models:
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{model_name}' is not loaded. Available models: {list(loaded_models.keys())}"
        )

    # Validate file extension
    if not file.filename.lower().endswith(('.wav', '.mp3', '.ogg', '.m4a')):
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    # Process audio using a temporary file to minimize memory footprint
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, 'wb') as tmp:
            tmp.write(await file.read())

        # Inference using the selected model instance
        segments, _ = loaded_models[model_name].transcribe(tmp_path, beam_size=5)
        text = " ".join([s.text for s in segments])
        
        return {"text": text.strip()}
    except Exception as e:
        logger.error(f"Transcription error for model {model_name}: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error during transcription.")
    finally:
        # Cleanup temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/health")
async def health_check():
    """
    Returns the status of the API and the list of currently active models.
    """
    return {
        "status": "online",
        "active_models": list(loaded_models.keys())
    }