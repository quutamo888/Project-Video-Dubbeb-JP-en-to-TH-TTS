import os
from dotenv import load_dotenv
import torch

load_dotenv()

class Config:
    # API Keys & Auth Tokens
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    
    # Device Settings
    USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
    DEVICE = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
    TEMP_DIR = "temp"
    
    # Paths
    OUTPUT_DIR = "output"
    LOGS_DIR = "logs"
    
    # STT Settings
    # "kotoba-whisper" (Japanese SOTA, Ultra Accurate) or "faster-whisper" (Universal)
    STT_ENGINE = os.getenv("STT_ENGINE", "kotoba-whisper")
    KOTOBA_MODEL = os.getenv("KOTOBA_MODEL", "kotoba-tech/kotoba-whisper-v2.2-faster")
    WHISPER_MODEL_SIZE = "large-v2" # or "medium", "small"

    # Translation Settings
    # "local-ctranslate2" (Fastest GPU, offline), "local-qwen" (Context-Aware LLM, offline), "local-transformer", "google"
    TRANSLATION_PROVIDER = os.getenv("TRANSLATION_PROVIDER", "local-ctranslate2")
    CT2_NLLB_MODEL = os.getenv("CT2_NLLB_MODEL", "JustFrederik/nllb-200-distilled-600M-ct2-float16")
    NLLB_MODEL = os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M")
    QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b") # or mistral, gemma

    # TTS Settings
    TTS_PROVIDER = os.getenv("TTS_PROVIDER", "omnivoice") # "omnivoice", "mms", "edge-tts"
    OMNIVOICE_MODEL = os.getenv("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
    EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", "th-TH-PremwadeeNeural") # th-TH-PremwadeeNeural, th-TH-NiwatNeural
    OMNIVOICE_USE_CLONE = os.getenv("OMNIVOICE_USE_CLONE", "true").lower() == "true"
    OMNIVOICE_CLONE_MODE = os.getenv("OMNIVOICE_CLONE_MODE", "timbre") # "timbre" (Authentic Thai accent), "full", "disabled"
    OMNIVOICE_REUSE_PROMPT = os.getenv("OMNIVOICE_REUSE_PROMPT", "true").lower() == "true" # Consistent speaker voice
    
    # Gender Detection Settings
    # "audio" - Fast, pitch analysis (current default)
    # "visual" - Accurate, face detection from video frames
    # "hybrid" - Visual first, fallback to audio
    GENDER_DETECTION_METHOD = os.getenv("GENDER_DETECTION_METHOD", "audio")

    # Audio Gender Model (when using audio detection)
    # "ml-robust" - audeering/wav2vec2-large-robust-12-ft-age-gender (SOTA multilingual, robust)
    # "ml-librispeech" - alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech (Legacy)
    # "pitch" - Fast, frequency-based (Hz threshold)
    AUDIO_GENDER_MODEL = os.getenv("AUDIO_GENDER_MODEL", "ml-robust")
    ROBUST_GENDER_MODEL = os.getenv("ROBUST_GENDER_MODEL", "audeering/wav2vec2-large-robust-12-ft-age-gender")
    LEGACY_GENDER_MODEL = os.getenv("LEGACY_GENDER_MODEL", "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech")

    # Speaker Clustering & Consistent Gender
    # Groups segments by speaker embedding, performs majority voting with linguistic hints to prevent flipping
    SPEAKER_CLUSTERING_GENDER = os.getenv("SPEAKER_CLUSTERING_GENDER", "true").lower() == "true"

    # Output Video Audio Settings
    # true: Preserve original soundtrack (Track 1) and add Thai dubbed (Track 2)
    # false: Replace audio with Thai dubbed only
    DUAL_AUDIO_TRACKS = os.getenv("DUAL_AUDIO_TRACKS", "true").lower() == "true"

    # Multitask Settings
    ENABLE_MULTITASK = os.getenv("ENABLE_MULTITASK", "false").lower() == "true"
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
    
    @staticmethod
    def validate():
        if Config.TRANSLATION_PROVIDER == "openai" and not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for OpenAI translation")
