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

        return self.models.get(model_id), self.tokenizers.get(model_id)

    def _clean_text(self, text):
        import re
        # Remove polite particles
        text = re.sub(r'(ครับ|ค่ะ|คะ|นะครับ|นะค่ะ|จ๊ะ|จ้ะ|นะคะ)', '', text)
        return text.strip()

    def generate(self, segments, original_audio_path, output_dir, male_pitch=0, female_pitch=0, stop_event=None):
        """
        Generates audio using configured provider (F5-TTS or MMS).
        """
        provider = Config.TTS_PROVIDER
        
        if provider == "f5-tts":
            return self.generate_f5(segments, original_audio_path, output_dir, stop_event)
        
        # Fallback to MMS (Existing Logic)
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
            if not text: continue
                
            output_filename = os.path.join(output_dir, f"seg_{i:04d}.wav")
            if os.path.exists(output_filename):
                generated_files.append(output_filename)
                continue
                
            logger.info(f"Generating segment {i} ({gender}): {text[:30]}...")
            
            try:
                model, tokenizer = self.get_model_and_tokenizer(gender)
                if model is None: raise RuntimeError(f"Model for {gender} not loaded")

                inputs = tokenizer(text, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    output = model(**inputs).waveform
                
                audio_np = output.cpu().numpy().squeeze()
                
                # Pitch Shift
                pitch_shift = male_pitch if gender == 'male' else female_pitch
                if pitch_shift != 0:
                     sr = model.config.sampling_rate if hasattr(model, 'config') else self.sample_rate
                     try:
                         audio_np = librosa.effects.pitch_shift(audio_np, sr=sr, n_steps=pitch_shift)
                     except Exception as e:
                         pass

                # Save
                if len(audio_np) > 0:
                    sr = model.config.sampling_rate if hasattr(model, 'config') else self.sample_rate
                    # Check Hallucination
                    target_duration = seg.get('end', 0) - seg.get('start', 0)
                    gen_duration = len(audio_np) / sr
                    if target_duration > 0 and gen_duration > target_duration * 4 and (gen_duration - target_duration) > 2.0:
                         logger.warning(f"MMS Hallucination detected seg {i}. Silencing.")
                         audio_int16 = np.zeros(int(target_duration * sr), dtype=np.int16)
                    else:
                        max_val = np.max(np.abs(audio_np))
                        if max_val == 0:
                            audio_int16 = np.zeros(len(audio_np), dtype=np.int16)
                        else:
                            audio_int16 = (audio_np / max_val * 32767).astype(np.int16)
                            
                    scipy.io.wavfile.write(output_filename, sr, audio_int16)
                    generated_files.append(output_filename)
            except Exception as e:
                logger.error(f"Failed to generate segment {i}: {e}")
                continue
                
        return generated_files

    def generate_f5(self, segments, original_audio_path, output_dir, stop_event=None):
        logger.info("Initializing F5-TTS...")
        try:
            from f5_tts_th.tts import TTS
            import soundfile as sf
            from pydub import AudioSegment
        except ImportError:
            logger.error("F5-TTS not installed. Please install 'f5-tts-th' and 'soundfile'.")
            return []

        # Load F5 Model (v1 recommended in docs)
        if not hasattr(self, 'f5_model'):
            self.f5_model = TTS(model="v1")
        
        generated_files = []
        
        # Load full audio for slicing
        full_audio = AudioSegment.from_file(original_audio_path)
        
        for i, seg in enumerate(segments):
            if stop_event and stop_event.is_set():
                logger.info("F5-TTS stopped by user.")
                break
                
            text = seg.get('translated_text', '')
            original_text = seg.get('text', '') # Original transcript
            
            if not text: continue
            
            text = self._clean_text(text)
            if not text: continue
            
            output_filename = os.path.join(output_dir, f"seg_{i:04d}.wav")
            if os.path.exists(output_filename):
                generated_files.append(output_filename)
                continue
            
            # Extract Reference Audio
            start_ms = int(seg['start'] * 1000)
            end_ms = int(seg['end'] * 1000)
            
            # Ensure at least 1-2 sec for reference? F5 might need decent length.
            # If segment is too short, extend slightly (careful of noise)
            if end_ms - start_ms < 500:
                end_ms = start_ms + 1000
                
            ref_audio_seg = full_audio[start_ms:end_ms]
            ref_audio_path = os.path.join(output_dir, f"ref_{i:04d}.wav")
            ref_audio_seg.export(ref_audio_path, format="wav")
            
            logger.info(f"F5-TTS Generating seg {i}: {text[:30]}...")
            
            try:
                # Infer
                # Note: ref_text is the text of the REFERENCE audio.
                # If we use original audio, we should use original text.
                # But F5-TTS-THAI might only support Thai text. 
                # Let's try passing "." if original text causes issues, but ideally we pass original text.
                # If the library crashes on non-Thai ref_text, we might need a fixed Thai reference.
                # For now, let's assume it can handle foreign characters or ignore them?
                # Actually, providing incorrect ref_text usually degrades quality (prosody mismatch).
                # Strategy: Pass original text. If it fails, fallback strategy needed (maybe fixed ref).
                
                wav = self.f5_model.infer(
                    ref_audio=ref_audio_path,
                    ref_text=original_text if original_text else ".",
                    gen_text=text,
                    step=32,
                    cfg=2.0,
                    speed=1.0
                )
                
                # Write output
                sf.write(output_filename, wav, 24000)
                generated_files.append(output_filename)
                
                # Cleanup ref
                if os.path.exists(ref_audio_path):
                    os.remove(ref_audio_path)
                    
            except Exception as e:
                logger.error(f"F5-TTS Failed seg {i}: {e}")
                # Fallback? Or just skip
                continue
                
        return generated_files

_generator = VoiceGenerator()

def generate_voice(segments, original_audio_path, output_dir, male_pitch=0, female_pitch=0, stop_event=None):
    return _generator.generate(segments, original_audio_path, output_dir, male_pitch, female_pitch, stop_event)
