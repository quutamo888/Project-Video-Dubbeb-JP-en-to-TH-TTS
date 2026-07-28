import os
from dotenv import load_dotenv
import torch
import torch

load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    
    # Device Settings
    USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
    DEVICE = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
    TEMP_DIR = "temp"
    
    # Paths
    OUTPUT_DIR = "output"
    
    # Models
    WHISPER_MODEL_SIZE = "large-v2" # or "medium", "small"
    DEVICE = "cuda" if os.getenv("USE_GPU", "true").lower() == "true" else "cpu"

    # Translation Settings
    # "ollama", "openai", "google", "local-transformer" (NLLB - runs locally)
    TRANSLATION_PROVIDER = os.getenv("TRANSLATION_PROVIDER", "ollama") 
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b") # or mistral, gemma
    
    # Local Transformer Translation (NLLB)
    NLLB_MODEL = os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M")  # ~600MB, supports JP->TH

    # TTS Settings
    TTS_PROVIDER = os.getenv("TTS_PROVIDER", "omnivoice") # "omnivoice", "f5-tts", "mms", "edge-tts"
    EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", "th-TH-PremwadeeNeural") # th-TH-PremwadeeNeural, th-TH-NiwatNeural
    
    # Gender Detection Settings
    # "audio" - Fast, pitch analysis (current default)
    # "visual" - Accurate, face detection from video frames
    # "hybrid" - Visual first, fallback to audio
    GENDER_DETECTION_METHOD = os.getenv("GENDER_DETECTION_METHOD", "audio")
    
    # Audio Gender Model (when using audio detection)
    # "pitch" - Fast, frequency-based (Hz threshold)
    # "ml" - Accurate, ML-based (wav2vec2)
    AUDIO_GENDER_MODEL = os.getenv("AUDIO_GENDER_MODEL", "ml")
    
    @staticmethod
    def validate():
        if Config.TRANSLATION_PROVIDER == "openai" and not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for OpenAI translation")
