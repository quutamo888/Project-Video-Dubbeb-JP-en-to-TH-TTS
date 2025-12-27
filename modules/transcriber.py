from faster_whisper import WhisperModel
import logging
import os
import torch

logger = logging.getLogger(__name__)

def transcribe_audio(audio_path, model_size="large-v3", device="auto"):
    """
    Transcribes audio using faster-whisper.
    
    Args:
        audio_path (str): Path to the audio file.
        model_size (str): Whisper model size (e.g., "medium", "large-v3").
        device (str): "cuda" or "cpu".
        
    Returns:
        list: List of segments (dict with 'start', 'end', 'text').
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    logger.info(f"Loading Whisper model '{model_size}' on {device}...")
    
    try:
        # compute_type="float16" for GPU, "int8" for CPU usually better for speed/compat
        compute_type = "float16" if device == "cuda" else "int8"
        
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        
        logger.info(f"Transcribing {audio_path}...")
        segments, info = model.transcribe(audio_path, beam_size=5, vad_filter=True)
        
        result = []
        for segment in segments:
            # logger.debug(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            result.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
            
        logger.info(f"Transcription complete. Detected language: {info.language}")
        return result, info.language
        
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise
