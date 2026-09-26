import re
import os
import logging
import torch
from config import Config

logger = logging.getLogger(__name__)

# Cache for models
_whisper_model = None
_kotoba_model = None

def transcribe_kotoba_whisper(audio_path, model_ref=None, device="auto", log_callback=None):
    """
    Transcribes Japanese audio using Kotoba-Whisper (SOTA Japanese ASR via faster-whisper CTranslate2).
    """
    def log(msg):
        logger.info(msg)
        if log_callback:
            log_callback(msg)

    from faster_whisper import WhisperModel

    if device == "auto":
        device_str = "cuda" if torch.cuda.is_available() and Config.USE_GPU else "cpu"
    else:
        device_str = device

    model_path = model_ref or getattr(Config, "KOTOBA_MODEL", "kotoba-tech/kotoba-whisper-v2.2-faster")
    compute_type = "float16" if device_str == "cuda" else "int8"
    log(f"Loading Kotoba-Whisper model '{model_path}' on {device_str} ({compute_type})...")

    model = WhisperModel(model_path, device=device_str, compute_type=compute_type)

    log(f"Transcribing {audio_path} with Kotoba-Whisper (Language: ja)...")
    transcribe_kwargs = {
        "language": "ja",
        "beam_size": 5,
        "vad_filter": True,
        "vad_parameters": dict(min_silence_duration_ms=500),
    }

    segments, info = model.transcribe(audio_path, **transcribe_kwargs)

    result = []
    for i, segment in enumerate(segments):
        text = segment.text.strip()
        if text:
            start_sec = round(float(segment.start), 3)
            end_sec = round(float(segment.end), 3)
            log(f"   📝 [Transcribe {i+1}] [{start_sec:.2f}s -> {end_sec:.2f}s]: \"{text}\"")
            result.append({
                "start": start_sec,
                "end": end_sec,
                "text": text,
                "emotion": "neutral",
                "event": None
            })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log(f"Kotoba-Whisper transcription complete. Found {len(result)} segments.")
    return result, "ja"

def transcribe_faster_whisper(audio_path, model_size="large-v2", device="auto", source_lang="auto", log_callback=None):
    """
    Transcribes audio using faster-whisper.
    """
    def log(msg):
        logger.info(msg)
        if log_callback:
            log_callback(msg)

    from faster_whisper import WhisperModel

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() and Config.USE_GPU else "cpu"

    log(f"Loading Whisper model '{model_size}' on {device}...")

    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    log(f"Transcribing {audio_path} (Source Lang: {source_lang})...")
    transcribe_kwargs = {"beam_size": 5, "vad_filter": True}
    if source_lang and source_lang != "auto":
        transcribe_kwargs["language"] = source_lang

    segments, info = model.transcribe(audio_path, **transcribe_kwargs)

    result = []
    for i, segment in enumerate(segments):
        text = segment.text.strip()
        if text:
            log(f"   📝 [Transcribe {i+1}] [{segment.start:.2f}s -> {segment.end:.2f}s]: \"{text}\"")
        result.append({
            "start": segment.start,
            "end": segment.end,
            "text": text,
            "emotion": "neutral",
            "event": None
        })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log(f"Whisper transcription complete. Detected language: {info.language}")
    return result, info.language

def transcribe_audio(audio_path, model_size="large-v2", device="auto", source_lang="auto", engine="kotoba-whisper", log_callback=None):
    """
    Unified entry point for transcribing audio.
    Supports Kotoba-Whisper (Japanese SOTA via faster-whisper) and faster-whisper (universal large-v2).
    """
    def log(msg):
        logger.info(msg)
        if log_callback:
            log_callback(msg)

    engine_choice = (engine or getattr(Config, "STT_ENGINE", "kotoba-whisper")).lower()

    if "kotoba" in engine_choice:
        # Kotoba-Whisper is specialized for Japanese
        # If source_lang is specified as non-Japanese (e.g. en, zh, ko, th), fallback to universal faster-whisper
        if source_lang and source_lang.lower() not in ["auto", "ja", "jpn", "japanese"]:
            log(f"ℹ️ Kotoba-Whisper is specialized for Japanese. Source language is '{source_lang}'. Switching to faster-whisper ({model_size})...")
            return transcribe_faster_whisper(audio_path, model_size=model_size, device=device, source_lang=source_lang, log_callback=log_callback)

        try:
            return transcribe_kotoba_whisper(audio_path, device=device, log_callback=log_callback)
        except Exception as e:
            log(f"⚠️ Kotoba-Whisper failed: {e}. Falling back to faster-whisper...")
            return transcribe_faster_whisper(audio_path, model_size=model_size, device=device, source_lang=source_lang, log_callback=log_callback)
    else:
        return transcribe_faster_whisper(audio_path, model_size=model_size, device=device, source_lang=source_lang, log_callback=log_callback)
