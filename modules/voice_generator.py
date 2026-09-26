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

try:
    from omnivoice import OmniVoice
    OMNIVOICE_AVAILABLE = True
except ImportError:
    OMNIVOICE_AVAILABLE = False

# Monkey-patch torchaudio.load to use soundfile/librosa instead of torchcodec
# This avoids torchcodec dependency issues on Windows with PyTorch nightly
def _patched_torchaudio_load(filepath, *args, **kwargs):
    """Patched torchaudio.load that uses soundfile/librosa instead of torchcodec."""
    try:
        # Try soundfile first
        waveform, sample_rate = sf.read(filepath, dtype='float32')
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
            raise RuntimeError(f"Failed to load audio file '{filepath}': sf_err={e}, librosa_err={e2}")

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

    def _clean_text(self, text, tts_lang="th"):
        import re
        if tts_lang == "th":
            # Remove polite particles for Thai
            text = re.sub(r'(ครับ|ค่ะ|คะ|นะครับ|นะค่ะ|จ๊ะ|จ้ะ|นะคะ)', '', text)
        return text.strip()

    def generate(self, segments, original_audio_path, output_dir, male_pitch=0, female_pitch=0, stop_event=None, tts_lang="th", log_callback=None, enable_multitask=False, max_workers=4, use_clone=True, clone_mode="full", reuse_speaker_voice=True):
        """
        Generates audio using configured provider (OmniVoice, MMS, or Edge-TTS).
        """
        import glob
        # Clean up stale segment files from previous runs to prevent cross-video cache bugs
        for f in glob.glob(os.path.join(output_dir, "seg_*.*")):
            try:
                os.remove(f)
            except Exception:
                pass

        provider = Config.TTS_PROVIDER
        
        if provider == "edge-tts":
            return self.generate_edge_tts(segments, output_dir, male_pitch, female_pitch, stop_event=stop_event, tts_lang=tts_lang, log_callback=log_callback, enable_multitask=enable_multitask, max_workers=max_workers)
        elif provider == "omnivoice":
            return self.generate_omnivoice(segments, original_audio_path, output_dir, male_pitch=male_pitch, female_pitch=female_pitch, stop_event=stop_event, tts_lang=tts_lang, log_callback=log_callback, enable_multitask=enable_multitask, max_workers=max_workers, use_clone=use_clone, clone_mode=clone_mode, reuse_speaker_voice=reuse_speaker_voice)
        
        # Fallback to MMS (Existing Logic)
        import concurrent.futures

        def log(msg):
            logger.info(msg)
            if log_callback: log_callback(msg)

        generated_files = []

        def process_mms_segment(item):
            i, seg = item
            if stop_event and stop_event.is_set():
                return None

            text = seg.get('translated_text', '')
            gender = seg.get('gender', 'unknown')
            
            if not text:
                return None
            
            text = self._clean_text(text, tts_lang=tts_lang)
            if not text:
                return None
                
            output_filename = os.path.join(output_dir, f"seg_{i:04d}.wav")
            if os.path.exists(output_filename):
                return output_filename
                
            log(f"   🎤 [MMS {i+1}/{len(segments)}] Generating audio ({gender}, {tts_lang}): \"{text[:30]}\"")
            
            try:
                if not hasattr(self, 'omni_lock'):
                    import threading
                    self.omni_lock = threading.Lock()

                with self.omni_lock:
                    model, tokenizer = self.get_model_and_tokenizer(gender)
                    if model is None: raise RuntimeError(f"Model for {gender} not loaded")

                    inputs = tokenizer(text, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        output = model(**inputs).waveform
                
                audio_np = output.cpu().numpy().squeeze()
                
                pitch_shift = male_pitch if gender == 'male' else female_pitch
                if pitch_shift != 0:
                     sr = model.config.sampling_rate if hasattr(model, 'config') else self.sample_rate
                     try:
                         audio_np = librosa.effects.pitch_shift(audio_np, sr=sr, n_steps=pitch_shift)
                     except Exception as e:
                         pass

                if len(audio_np) > 0:
                    sr = model.config.sampling_rate if hasattr(model, 'config') else self.sample_rate
                    target_duration = seg.get('end', 0) - seg.get('start', 0)
                    gen_duration = len(audio_np) / sr
                    if target_duration > 0 and gen_duration > target_duration * 4 and (gen_duration - target_duration) > 2.0:
                         log(f"   ⚠️ MMS Hallucination detected seg {i+1}. Silencing.")
                         audio_int16 = np.zeros(int(target_duration * sr), dtype=np.int16)
                    else:
                        max_val = np.max(np.abs(audio_np))
                        if max_val == 0:
                            audio_int16 = np.zeros(len(audio_np), dtype=np.int16)
                        else:
                            audio_int16 = (audio_np / max_val * 32767).astype(np.int16)
                            
                    scipy.io.wavfile.write(output_filename, sr, audio_int16)
                    return output_filename
            except Exception as e:
                log(f"   ❌ MMS Failed seg {i+1}: {e}")
                return None

        items = list(enumerate(segments))
        generated_files = []
        for item in items:
            i, seg = item
            if stop_event and stop_event.is_set():
                log("MMS stopped by user.")
                break
            r = process_mms_segment(item)
            generated_files.append(r)
            seg['audio_file'] = r

        return generated_files

    def generate_omnivoice(self, segments, original_audio_path, output_dir, male_pitch=0, female_pitch=0, stop_event=None, tts_lang="th", log_callback=None, enable_multitask=False, max_workers=4, use_clone=True, clone_mode="full", reuse_speaker_voice=True):
        import concurrent.futures
        
        def log(msg):
            logger.info(msg)
            if log_callback: log_callback(msg)

        mode_str = f"Multitask ({max_workers} threads)" if (enable_multitask and max_workers > 1) else "Sequential"
        clone_str = f"Full Clone (Voice + Accent)" if clone_mode == "full" else ("Timbre Only (Voice Timbre, No Accent)" if clone_mode == "timbre" else "Disabled")
        reuse_str = ", Consistent Speaker Profile" if (reuse_speaker_voice and clone_mode == "full") else ""
        log(f"Initializing OmniVoice [{mode_str}, {clone_str}{reuse_str}]...")

        if not OMNIVOICE_AVAILABLE:
            log("❌ OmniVoice not installed. Please install 'omnivoice'.")
            return []

        if not hasattr(self, 'omnivoice_model'):
            device = "cuda:0" if torch.cuda.is_available() and Config.USE_GPU else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() and Config.USE_GPU else torch.float32
            target_model = getattr(Config, 'OMNIVOICE_MODEL', 'k2-fsa/OmniVoice')
            token = getattr(Config, 'HF_TOKEN', None) or None
            if "Ex0TiiC" in target_model and not token:
                log("⚠️ 'Ex0TiiC' model is gated and requires HF_TOKEN in .env. Using open 'k2-fsa/OmniVoice'...")
                target_model = "k2-fsa/OmniVoice"
            log(f"Loading OmniVoice model '{target_model}' onto {device} with {dtype}...")
            
            if torch.cuda.is_available() and Config.USE_GPU:
                try:
                    torch.backends.cuda.enable_flash_sdp(True)
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                except Exception:
                    pass

            try:
                self.omnivoice_model = OmniVoice.from_pretrained(target_model, device_map=device, dtype=dtype, token=token)
                log(f"✅ Successfully loaded OmniVoice model: {target_model}")
            except Exception as e:
                log(f"⚠️ Failed to load '{target_model}': {e}")
                if target_model != "k2-fsa/OmniVoice":
                    log(f"🔄 Falling back to standard 'k2-fsa/OmniVoice'...")
                    self.omnivoice_model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=dtype)
                    log(f"✅ Loaded fallback 'k2-fsa/OmniVoice'")
                else:
                    raise e
        
        from pydub import AudioSegment
        is_clone_active = use_clone and clone_mode != "disabled"
        has_ref = is_clone_active and original_audio_path and os.path.exists(original_audio_path)
        full_audio = AudioSegment.from_file(original_audio_path) if has_ref else None

        if not hasattr(self, 'omni_lock'):
            import threading
            self.omni_lock = threading.Lock()

        # Pre-build reusable VoiceClonePrompts per gender if consistent speaker voice is enabled
        speaker_prompts = {}
        if is_clone_active and clone_mode == "full" and has_ref and reuse_speaker_voice:
            log("   👤 Building Consistent Speaker Voice Prompts per gender...")
            for g in ['female', 'male']:
                g_segs = [s for s in segments if s.get('gender', 'female').lower() == g and s.get('text')]
                if g_segs:
                    best_seg = max(g_segs, key=lambda s: (3 <= (s.get('end', 0) - s.get('start', 0)) <= 10, s.get('end', 0) - s.get('start', 0)))
                    s_ms = int(best_seg['start'] * 1000)
                    e_ms = int(best_seg['end'] * 1000)
                    if e_ms - s_ms < 1000:
                        e_ms = s_ms + 1000
                    
                    ref_audio_seg = full_audio[s_ms:e_ms]
                    ref_path = os.path.join(output_dir, f"ref_speaker_{g}.wav")
                    ref_audio_seg.export(ref_path, format="wav")
                    
                    try:
                        ref_txt = best_seg.get('text', '.')
                        with self.omni_lock:
                            prompt = self.omnivoice_model.create_voice_clone_prompt(
                                ref_audio=ref_path,
                                ref_text=ref_txt if ref_txt else ".",
                                preprocess_prompt=True
                            )
                        speaker_prompts[g] = prompt
                        log(f"      ✅ Built consistent {g.capitalize()} speaker voice profile")
                    except Exception as pe:
                        log(f"      ⚠️ Could not build prompt for {g}: {pe}")
                    finally:
                        if os.path.exists(ref_path):
                            try: os.remove(ref_path)
                            except Exception: pass

        def process_omni_segment(item):
            i, seg = item
            if stop_event and stop_event.is_set():
                return None
                
            text = seg.get('translated_text', '')
            original_text = seg.get('text', '')
            gender = seg.get('gender', 'female').lower()
            emotion = seg.get('emotion', 'neutral').lower()
            event = seg.get('event')

            if not text:
                return None

            text = self._clean_text(text, tts_lang=tts_lang)
            if not text:
                return None

            # Integrate event into inline control syntax if supported (e.g. [laughter])
            if event == "laughter" and "[laughter]" not in text:
                text = f"{text} [laughter]"

            output_filename = os.path.join(output_dir, f"seg_{i:04d}.wav")
            if os.path.exists(output_filename):
                return output_filename

            emotion_info = f", emotion: {emotion.upper()}" if emotion != "neutral" else ""
            log(f"   🎤 [OmniVoice {i+1}/{len(segments)}] Generating audio ({gender}, {tts_lang}{emotion_info}): \"{text}\"")

            ref_audio_path = None
            try:
                # Map detected emotion to OmniVoice instruct attributes & speed
                gender_token = "female" if gender == "female" else "male"
                if emotion == "happy":
                    voice_instruct = f"{gender_token}, high pitch"
                    base_speed = 1.05
                elif emotion == "angry":
                    voice_instruct = f"{gender_token}, high pitch"
                    base_speed = 1.10
                elif emotion == "sad":
                    voice_instruct = f"{gender_token}, low pitch"
                    base_speed = 0.92
                elif emotion == "fearful":
                    voice_instruct = f"{gender_token}, whisper"
                    base_speed = 1.02
                elif emotion in ["disgusted", "surprised"]:
                    voice_instruct = f"{gender_token}, high pitch"
                    base_speed = 1.05
                else:
                    voice_instruct = gender_token
                    base_speed = 1.0

                norm_lang = tts_lang.split("-")[0].lower() if tts_lang else "th"

                with self.omni_lock:
                    with torch.inference_mode():
                        if clone_mode == "timbre":
                            audio_list = self.omnivoice_model.generate(
                                text=text,
                                language=norm_lang,
                                instruct=voice_instruct,
                                speed=base_speed,
                                num_step=32
                            )
                        elif has_ref and full_audio:
                            if reuse_speaker_voice and gender in speaker_prompts:
                                audio_list = self.omnivoice_model.generate(
                                    text=text,
                                    language=norm_lang,
                                    voice_clone_prompt=speaker_prompts[gender],
                                    speed=base_speed,
                                    num_step=32
                                )
                            else:
                                start_ms = int(seg['start'] * 1000)
                                end_ms = int(seg['end'] * 1000)
                                if end_ms - start_ms < 1000:
                                    end_ms = start_ms + 1000

                                ref_audio_seg = full_audio[start_ms:end_ms]
                                ref_audio_path = os.path.join(output_dir, f"ref_omni_{i:04d}.wav")
                                ref_audio_seg.export(ref_audio_path, format="wav")

                                audio_list = self.omnivoice_model.generate(
                                    text=text,
                                    language=norm_lang,
                                    ref_audio=ref_audio_path,
                                    ref_text=original_text if original_text else ".",
                                    speed=base_speed,
                                    num_step=32
                                )
                        else:
                            audio_list = self.omnivoice_model.generate(
                                text=text,
                                language=norm_lang,
                                instruct=voice_instruct,
                                speed=base_speed,
                                num_step=32
                            )
                
                final_audio = np.asarray(audio_list[0])

                # Differentiate Male and Female voices when standard TTS (or pitch adjustment requested)
                pitch_shift = male_pitch if gender == 'male' else female_pitch
                if not use_clone and gender == 'male' and pitch_shift == 0:
                    pitch_shift = -3.5  # Pitch down for male voice differentiation

                if pitch_shift != 0:
                    try:
                        final_audio = librosa.effects.pitch_shift(final_audio, sr=24000, n_steps=pitch_shift)
                    except Exception as pe:
                        log(f"   ⚠️ Pitch shift failed for seg {i+1}: {pe}")

                sf.write(output_filename, final_audio, 24000)
                return output_filename
            except Exception as e:
                log(f"   ❌ OmniVoice Failed seg {i+1}: {e}")
                return None
            finally:
                if ref_audio_path and os.path.exists(ref_audio_path):
                    try:
                        os.remove(ref_audio_path)
                    except Exception:
                        pass

        items = list(enumerate(segments))
        generated_files = []
        for item in items:
            i, seg = item
            if stop_event and stop_event.is_set():
                log("OmniVoice stopped by user.")
                break
            r = process_omni_segment(item)
            generated_files.append(r)
            seg['audio_file'] = r
        return generated_files

    def generate_edge_tts(self, segments, output_dir, male_pitch=0, female_pitch=0, stop_event=None, tts_lang="th", log_callback=None, enable_multitask=False, max_workers=4):
        def log(msg):
            logger.info(msg)
            if log_callback: log_callback(msg)

        mode_str = f"Multitask ({max_workers} tasks)" if (enable_multitask and max_workers > 1) else "Sequential"
        log(f"Initializing Edge TTS [{mode_str}]...")
        
        try:
            import edge_tts
            import asyncio
        except ImportError:
            log("❌ Edge TTS not installed. Please install 'edge-tts'.")
            return []
        
        if tts_lang == "en":
            voices = {
                "male": "en-US-AndrewNeural",
                "female": "en-US-AvaNeural"
            }
        else:
            voices = {
                "male": "th-TH-NiwatNeural",
                "female": "th-TH-PremwadeeNeural"
            }
        
        semaphore = asyncio.Semaphore(max_workers if (enable_multitask and max_workers > 0) else 4)

        async def generate_segment_async(i, seg):
            async with semaphore:
                text = seg.get('translated_text', '')
                gender = seg.get('gender', 'Female').lower()
                
                if not text:
                    return None
                
                text = self._clean_text(text, tts_lang=tts_lang)
                if not text:
                    return None
                
                output_filename = os.path.join(output_dir, f"seg_{i:04d}.mp3")
                if os.path.exists(output_filename):
                    return output_filename
                
                voice = voices.get(gender, voices["female"])
                log(f"   🎤 [Edge-TTS {i+1}/{len(segments)}] Generating audio ({gender}, {tts_lang}): \"{text}\" ({voice})")
                
                try:
                    communicate = edge_tts.Communicate(text, voice)
                    await communicate.save(output_filename)
                    
                    output_wav = output_filename.replace(".mp3", ".wav")
                    from pydub import AudioSegment
                    audio = AudioSegment.from_mp3(output_filename)
                    
                    pitch_shift = male_pitch if gender == "male" else female_pitch
                    if pitch_shift != 0:
                        try:
                            new_sample_rate = int(audio.frame_rate * (2.0 ** (pitch_shift / 12.0)))
                            audio = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sample_rate})
                            audio = audio.set_frame_rate(44100)
                        except Exception as e:
                            log(f"   ⚠️ Pitch shift failed for seg {i+1}: {e}")
                    
                    audio.export(output_wav, format="wav")
                    
                    if os.path.exists(output_filename):
                        os.remove(output_filename)
                    
                    return output_wav
                    
                except Exception as e:
                    log(f"   ❌ Edge-TTS failed for seg {i+1}: {e}")
                    return None
        
        async def generate_all():
            tasks = []
            for i, seg in enumerate(segments):
                if stop_event and stop_event.is_set():
                    log("Edge-TTS stopped by user.")
                    break
                tasks.append(generate_segment_async(i, seg))

            results = await asyncio.gather(*tasks)
            for seg, r in zip(segments, results):
                seg['audio_file'] = r
            return list(results)

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        generated_files = loop.run_until_complete(generate_all())

        successful_count = sum(1 for f in generated_files if f)
        log(f"Edge-TTS generated {successful_count}/{len(segments)} audio files")
        return generated_files

_generator = VoiceGenerator()

def generate_voice(segments, original_audio_path, output_dir, male_pitch=0, female_pitch=0, stop_event=None, tts_lang="th", log_callback=None, enable_multitask=False, max_workers=4, use_clone=True, clone_mode="full", reuse_speaker_voice=True):
    return _generator.generate(segments, original_audio_path, output_dir, male_pitch, female_pitch, stop_event, tts_lang=tts_lang, log_callback=log_callback, enable_multitask=enable_multitask, max_workers=max_workers, use_clone=use_clone, clone_mode=clone_mode, reuse_speaker_voice=reuse_speaker_voice)
