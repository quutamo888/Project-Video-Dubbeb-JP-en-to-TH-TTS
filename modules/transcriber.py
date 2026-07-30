from faster_whisper import WhisperModel
import logging
import os
import torch

logger = logging.getLogger(__name__)

def transcribe_audio(audio_path, model_size="large-v3", device="auto", source_lang="auto", log_callback=None):
    """
    Transcribes audio using faster-whisper.
    
    Args:
        audio_path (str): Path to the audio file.
        model_size (str): Whisper model size (e.g., "medium", "large-v3").
        device (str): "cuda" or "cpu".
        source_lang (str): Source audio language code or "auto".
        log_callback (function): Function to send log messages to UI.
        
    Returns:
        list: List of segments (dict with 'start', 'end', 'text').
    """
    def log(msg):
        logger.info(msg)
        if log_callback:
            log_callback(msg)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    log(f"Loading Whisper model '{model_size}' on {device}...")
    
    try:
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
                "text": text
            })
            
        log(f"Transcription complete. Detected language: {info.language}")
        return result, info.language
        
    except Exception as e:
        log(f"Transcription failed: {str(e)}")
        raise
