import asyncio
import os
import logging
import torch
import scipy.io.wavfile
import numpy as np
from config import Config
try:
    from transformers import VitsModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

import librosa

logger = logging.getLogger(__name__)

class VoiceGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() and Config.USE_GPU else "cpu"
        self.models = {} # Store 'male' and 'female' models
        self.tokenizers = {}
        self.sample_rate = 16000 
    
    def get_model_and_tokenizer(self, gender):
        # Map gender to model ID
        if gender == 'male':
            model_id = "VIZINTZOR/MMS-TTS-THAI-MALEV1"
        else:
            model_id = "VIZINTZOR/MMS-TTS-THAI-FEMALEV1"
            
        # Lazy load
        if model_id not in self.models:
            if HAS_TRANSFORMERS:
                try:
                    logger.info(f"Loading TTS model: {model_id}...")
                    self.models[model_id] = VitsModel.from_pretrained(model_id).to(self.device)
                    self.tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id)
                except Exception as e:
                    logger.error(f"Failed to load {model_id}: {e}")
                    return None, None
            else:
                return None, None
                
        return self.models.get(model_id), self.tokenizers.get(model_id)

    def _clean_text(self, text):
        import re
        # Remove polite particles
        text = re.sub(r'(ครับ|ค่ะ|คะ|นะครับ|นะค่ะ|จ๊ะ|จ้ะ|นะคะ)', '', text)
        return text.strip()

    def generate(self, segments, original_audio_path, output_dir, male_pitch=0, female_pitch=0, stop_event=None):
        """
        Generates audio using VIZINTZOR Dual Models (Local).
        args:
            male_pitch (float): Semitones to shift male voice (default 0).
            female_pitch (float): Semitones to shift female voice (default 0).
        """
        generated_files = []
        
        for i, seg in enumerate(segments):
            if stop_event and stop_event.is_set():
                logger.info("Voice generation stopped by user.")
                break

            text = seg.get('translated_text', '')
            gender = seg.get('gender', 'unknown')
            
            if not text:
                continue
            
            # Clean polite particles
            text = self._clean_text(text)
            if not text: # skipped if text became empty
                continue
                
            output_filename = os.path.join(output_dir, f"seg_{i:04d}.wav")
            
            # Resume capability
            if os.path.exists(output_filename):
                generated_files.append(output_filename)
                continue
                
            logger.info(f"Generating segment {i} ({gender}): {text[:30]}...")
            
            try:
                model, tokenizer = self.get_model_and_tokenizer(gender)
                
                if model is None:
                    raise RuntimeError(f"Model for {gender} not loaded")

                inputs = tokenizer(text, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    output = model(**inputs).waveform
                
                audio_np = output.cpu().numpy().squeeze()
                
                # Apply Pitch Shift if requested
                pitch_shift = 0
                if gender == 'male':
                    pitch_shift = male_pitch
                else:
                    pitch_shift = female_pitch
                    
                if pitch_shift != 0:
                     # Use model's native sampling rate for higher quality shift
                     sr = model.config.sampling_rate if hasattr(model, 'config') else self.sample_rate
                     try:
                         # Use librosa for pitch shifting (CPU based, might be slow but high quality)
                         audio_np = librosa.effects.pitch_shift(audio_np, sr=sr, n_steps=pitch_shift)
                     except Exception as e:
                         logger.warning(f"Pitch shift failed: {e}")

                # Check for silence/empty
                if audio_np is None or len(audio_np) == 0:
                     logger.warning(f"Segment {i} produced empty audio.")
                     continue

                max_val = np.max(np.abs(audio_np))
                if max_val == 0:
                    logger.warning(f"Segment {i} is silent. writing silent file.")
                    audio_int16 = np.zeros(len(audio_np), dtype=np.int16)
                    # Default SR if silent
                    sr = self.sample_rate 
                else:
                    audio_norm = audio_np / max_val
                    audio_int16 = (audio_norm * 32767).astype(np.int16)
                    # Use model's native sampling rate if available
                    sr = model.config.sampling_rate if hasattr(model, 'config') else self.sample_rate
                    
                    # Hallucination Check for Audio (Duration)
                    target_duration = seg.get('end', 0) - seg.get('start', 0)
                    if target_duration > 0:
                        gen_duration = len(audio_int16) / sr
                        # If generated audio is > 4x target (and > 2s diff), it's likely a loop/hallucination
                        if gen_duration > target_duration * 4 and (gen_duration - target_duration) > 2.0:
                             logger.warning(f"TTS Hallucination detected for seg {i} (Gen: {gen_duration:.1f}s vs Target: {target_duration:.1f}s). Skipping/Silencing.")
                             # Fallback to silence
                             audio_int16 = np.zeros(int(target_duration * sr), dtype=np.int16)
                
                scipy.io.wavfile.write(output_filename, sr, audio_int16)
                generated_files.append(output_filename)
                
            except Exception as e:
                logger.error(f"Failed to generate segment {i}: {e}")
                continue
                
        return generated_files

_generator = VoiceGenerator()

def generate_voice(segments, original_audio_path, output_dir, male_pitch=0, female_pitch=0, stop_event=None):
    return _generator.generate(segments, original_audio_path, output_dir, male_pitch, female_pitch, stop_event)
