"""
SuperBot Whisper API Microservice.
Optimized for air-gapped environments and CUDA acceleration.
"""

import os
import json
import tempfile
import logging
from typing import Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whisper-api")

app = FastAPI(title="SuperBot Whisper API", version="1.0.0")

CONFIG_FILE = "/app/config/superbot_config.json"

# Default fallback values
MODEL_SIZE = "small"
DEVICE = "cuda"
COMPUTE_TYPE = "int8"
OFFLINE_MODE = True
LISTENING_PORT = 8001 

if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            full_config = json.load(f)
            
            # Sync Ports from Single Source of Truth
            server_ports = full_config.get("server_ports", {})
            LISTENING_PORT = server_ports.get("stt", LISTENING_PORT)
            
            # Load Whisper Node configuration
            w_config = full_config.get("whisper_node", {})            
            MODEL_SIZE = w_config.get("model", MODEL_SIZE)
            DEVICE = w_config.get("device", DEVICE)
            COMPUTE_TYPE = w_config.get("compute_type", COMPUTE_TYPE)
            OFFLINE_MODE = w_config.get("offline_prebaked", True)
            
        logger.info(f"Configuration loaded. Expected port: {LISTENING_PORT}")
    except Exception as e:
        logger.error(f"Failed to parse {CONFIG_FILE}: {e}")
else:
    logger.warning("Config file not found. Using defaults.")

try:
    logger.info(f"Initializing WhisperModel: {MODEL_SIZE} ({DEVICE}/{COMPUTE_TYPE})")
    # local_files_only=True ensures no internet calls at runtime
    whisper_model = WhisperModel(
        MODEL_SIZE, 
        device=DEVICE, 
        compute_type=COMPUTE_TYPE,
        local_files_only=OFFLINE_MODE
    )
    logger.info("WhisperModel loaded successfully.")
except Exception as e:
    logger.critical(f"Critical Failure: {e}")
    whisper_model = None

@app.post("/v1/audio/transcriptions", response_model=Dict[str, str])
async def transcribe_audio(file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Args:
        file (UploadFile): Audio file (.wav, .mp3, .ogg, .m4a).
    Returns:
        Dict[str, str]: {"text": "transcribed content"}
    """
    if whisper_model is None:
        raise HTTPException(status_code=500, detail="STT Engine not initialized.")
        
    if not file.filename.lower().endswith(('.wav', '.mp3', '.ogg', '.m4a')):
        raise HTTPException(status_code=400, detail="Unsupported format.")
    
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, 'wb') as tmp:
            tmp.write(await file.read())

        segments, _ = whisper_model.transcribe(tmp_path, beam_size=5)
        text = " ".join([s.text for s in segments])
        return {"text": text.strip()}
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail="Processing error")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/health")
async def health():
    return {"status": "ready" if whisper_model else "error", "port": LISTENING_PORT}