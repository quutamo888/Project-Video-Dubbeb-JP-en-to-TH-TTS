import asyncio
import os
import logging
import torch
import torchaudio
import soundfile as sf
import scipy.io.wavfile
import numpy as np
from config import Config
try:
    from transformers import VitsModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

import librosa

# Monkey-patch torchaudio.load to use soundfile/librosa instead of torchcodec
# This avoids torchcodec dependency issues on Windows with PyTorch nightly
_original_torchaudio_load = torchaudio.load

def _patched_torchaudio_load(filepath, *args, **kwargs):
    """Patched torchaudio.load that uses soundfile instead of torchcodec."""
    try:
        # Try soundfile first
        waveform, sample_rate = sf.read(filepath, dtype='float32')
        # Convert to torch tensor and ensure correct shape (channels, samples)
        waveform = torch.from_numpy(waveform)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension
        elif waveform.dim() == 2:
            waveform = waveform.T  # Transpose to (channels, samples)
        return waveform, sample_rate
    except Exception as e:
        # Fallback to librosa
        try:
            waveform, sample_rate = librosa.load(filepath, sr=None, mono=False)
            waveform = torch.from_numpy(waveform)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            return waveform, sample_rate
        except Exception as e2:
            # Last resort: try original torchaudio
            return _original_torchaudio_load(filepath, *args, **kwargs)

# Apply the patch
torchaudio.load = _patched_torchaudio_load

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
        Generates audio using configured provider (F5-TTS, MMS, or Edge-TTS).
        """
        provider = Config.TTS_PROVIDER
        
        if provider == "f5-tts":
            return self.generate_f5(segments, original_audio_path, output_dir, stop_event)
        elif provider == "edge-tts":
            return self.generate_edge_tts(segments, output_dir, male_pitch, female_pitch, stop_event)
        elif provider == "omnivoice":
            return self.generate_omnivoice(segments, original_audio_path, output_dir, stop_event)
        
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

    def generate_omnivoice(self, segments, original_audio_path, output_dir, stop_event=None):
        logger.info("Initializing OmniVoice...")
        try:
            from omnivoice import OmniVoice
            import soundfile as sf
            from pydub import AudioSegment
        except ImportError:
            logger.error("OmniVoice not installed. Please install 'omnivoice'.")
            return []

        if not hasattr(self, 'omnivoice_model'):
            device = "cuda:0" if torch.cuda.is_available() and Config.USE_GPU else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() and Config.USE_GPU else torch.float32
            logger.info(f"Loading OmniVoice model onto {device} with {dtype}...")
            self.omnivoice_model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=dtype)
        
        generated_files = []
        
        has_ref = original_audio_path and os.path.exists(original_audio_path)
        if has_ref:
            full_audio = AudioSegment.from_file(original_audio_path)
        
        for i, seg in enumerate(segments):
            if stop_event and stop_event.is_set():
                logger.info("OmniVoice stopped by user.")
                break
                
            text = seg.get('translated_text', '')
            original_text = seg.get('text', '')
            
            if not text:
                continue
            
            text = self._clean_text(text)
            if not text:
                continue
            
            output_filename = os.path.join(output_dir, f"seg_{i:04d}.wav")
            if os.path.exists(output_filename):
                generated_files.append(output_filename)
                continue
                
            logger.info(f"OmniVoice Generating seg {i}: {text[:30]}...")
            
            ref_audio_path = None
            try:
                if has_ref:
                    start_ms = int(seg['start'] * 1000)
                    end_ms = int(seg['end'] * 1000)
                    if end_ms - start_ms < 1000:
                        end_ms = start_ms + 1000
                    
                    ref_audio_seg = full_audio[start_ms:end_ms]
                    ref_audio_path = os.path.join(output_dir, f"ref_omni_{i:04d}.wav")
                    ref_audio_seg.export(ref_audio_path, format="wav")
                    
                    audio_list = self.omnivoice_model.generate(
                        text=text,
                        ref_audio=ref_audio_path,
                        ref_text=original_text if original_text else ".",
                        speed=1.0,
                        num_step=32
                    )
                else:
                    audio_list = self.omnivoice_model.generate(
                        text=text,
                        speed=1.0,
                        num_step=32
                    )
                
                final_audio = np.asarray(audio_list[0])
                sf.write(output_filename, final_audio, 24000)
                generated_files.append(output_filename)
            except Exception as e:
                logger.error(f"OmniVoice Failed seg {i}: {e}")
            finally:
                if ref_audio_path and os.path.exists(ref_audio_path):
                    try:
                        os.remove(ref_audio_path)
                    except Exception:
                        pass
                        
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

    def generate_edge_tts(self, segments, output_dir, male_pitch=0, female_pitch=0, stop_event=None):
        """
        Generates audio using Microsoft Edge TTS (cloud-based, free).
        Provides high-quality Thai voices without local model downloads.
        """
        logger.info("Initializing Edge TTS...")
        
        try:
            import edge_tts
            import asyncio
        except ImportError:
            logger.error("Edge TTS not installed. Please install 'edge-tts'.")
            return []
        
        generated_files = []
        
        # Voice mapping based on gender
        # Thai voices from Microsoft Azure
        voices = {
            "male": "th-TH-NiwatNeural",      # Thai male voice
            "female": "th-TH-PremwadeeNeural" # Thai female voice (default)
        }
        
        async def generate_segment_async(i, seg):
            """Async function to generate a single segment."""
            text = seg.get('translated_text', '')
            gender = seg.get('gender', 'Female').lower()
            
            if not text:
                return None
            
            text = self._clean_text(text)
            if not text:
                return None
            
            output_filename = os.path.join(output_dir, f"seg_{i:04d}.mp3")
            if os.path.exists(output_filename):
                return output_filename
            
            voice = voices.get(gender, voices["female"])
            logger.info(f"Edge-TTS seg {i} ({gender}): {text[:30]}... using {voice}")
            
            try:
                # Create TTS communicator
                communicate = edge_tts.Communicate(text, voice)
                
                # Generate and save
                await communicate.save(output_filename)
                
                # Convert MP3 to WAV for consistency
                output_wav = output_filename.replace(".mp3", ".wav")
                from pydub import AudioSegment
                audio = AudioSegment.from_mp3(output_filename)
                
                # Apply pitch shift if needed
                pitch_shift = male_pitch if gender == "male" else female_pitch
                if pitch_shift != 0:
                    try:
                        # Pitch shift using pydub (changes speed too, so we adjust)
                        # For real pitch shift without speed change, we'd need librosa
                        # But for simplicity, we'll use frame_rate manipulation
                        new_sample_rate = int(audio.frame_rate * (2.0 ** (pitch_shift / 12.0)))
                        audio = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sample_rate})
                        audio = audio.set_frame_rate(44100)  # Normalize back
                    except Exception as e:
                        logger.warning(f"Pitch shift failed for seg {i}: {e}")
                
                audio.export(output_wav, format="wav")
                
                # Remove temp MP3
                if os.path.exists(output_filename):
                    os.remove(output_filename)
                
                return output_wav
                
            except Exception as e:
                logger.error(f"Edge-TTS failed for seg {i}: {e}")
                return None
        
        # Process all segments
        async def generate_all():
            tasks = []
            for i, seg in enumerate(segments):
                if stop_event and stop_event.is_set():
                    logger.info("Edge-TTS stopped by user.")
                    break
                tasks.append(generate_segment_async(i, seg))
            
            results = await asyncio.gather(*tasks)
            return [r for r in results if r is not None]
        
        # Run async loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        generated_files = loop.run_until_complete(generate_all())
        
        logger.info(f"Edge-TTS generated {len(generated_files)} audio files")
        return generated_files

_generator = VoiceGenerator()

def generate_voice(segments, original_audio_path, output_dir, male_pitch=0, female_pitch=0, stop_event=None):
    return _generator.generate(segments, original_audio_path, output_dir, male_pitch, female_pitch, stop_event)
