import argparse
import os
import sys
import logging

# Prevent CUDA memory fragmentation on Windows
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Set torchaudio backend BEFORE importing modules that use it
# This avoids torchcodec issues on Windows with PyTorch nightly
try:
    import torchaudio
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass

from config import Config
from utils import setup_logging, ensure_dirs, cleanup_temp

from modules.audio_extractor import extract_audio
from modules.transcriber import transcribe_audio
from modules.translator import translate_text
from modules.voice_generator import generate_voice
from modules.video_merger import merge_video
from modules.gender_classifier import classifier

from modules.vocal_isolator import separate_vocals

def run_pipeline(input_file, output_file, language="th",
                 progress_callback=None, log_callback=None,
                 male_pitch=0.0, female_pitch=0.0,
                 translation_temperature=0.3, stop_event=None,
                 source_lang="auto", target_lang=None, tts_lang=None,
                 enable_multitask=False, max_workers=4, use_clone=True, clone_mode="full", reuse_speaker_voice=True,
                 stt_engine=None, dual_audio=None):
    """
    Executes the full dubbing pipeline.
    """
    if target_lang is None:
        target_lang = language
    if tts_lang is None:
        tts_lang = target_lang
    if stt_engine is None:
        stt_engine = getattr(Config, "STT_ENGINE", "kotoba-whisper")
    if dual_audio is None:
        dual_audio = getattr(Config, "DUAL_AUDIO_TRACKS", True)

    setup_logging(Config.LOGS_DIR)
    logger = logging.getLogger("pipeline")

    def log(msg):
        print(msg)
        logger.info(msg)
        if log_callback:
            log_callback(msg)

    def progress(val, msg=""):
        if progress_callback:
            progress_callback(val, msg)

    def check_stop():
        if stop_event and stop_event.is_set():
            log("Process stopped by user.")
            raise InterruptedError("Stopped by user")

    try:
        check_stop()
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file '{input_file}' not found.")

        ensure_dirs([Config.TEMP_DIR, Config.OUTPUT_DIR, Config.LOGS_DIR])
        
        log("="*50)
        log(f"🎬 Starting AI Video Dubbing Pipeline")
        log(f"📂 Input: {input_file}")
        log(f"📂 Output: {output_file}")
        log(f"🎙️ STT Engine: {stt_engine.upper()}")
        log(f"🌐 Source Audio Lang: {source_lang}")
        log(f"🌐 Target Subtitle Lang: {target_lang}")
        log(f"🌐 Target TTS Voice Lang: {tts_lang}")
        log(f"🎤 Voice Clone Mode: {clone_mode.upper()} ({'Full Voice+Accent' if clone_mode=='full' else ('Tone Only, No Foreign Accent' if clone_mode in ['timbre', 'tone'] else 'Disabled')})")
        log(f"👤 Consistent Speaker Voice: {'Enabled' if reuse_speaker_voice else 'Disabled (Dynamic Per-Sentence)'}")
        log(f"⚡ Multitask: {'Enabled (' + str(max_workers) + ' tasks)' if enable_multitask else 'Disabled (Sequential)'}")
        log("="*50)
        
        # 1. Extract Audio
        log("")
        log("📢 [Step 1/6] Extracting Audio from Video...")
        progress(0.05, "Extracting Audio...")
        
        check_stop()
        audio_path = extract_audio(input_file, Config.TEMP_DIR)
        log(f"   ✅ Audio extracted to: {audio_path}")
        
        check_stop()
        # 1.5 Separate Vocals (Demucs)
        log("")
        log("🎵 [Step 2/6] Separating Background Music (Demucs)...")
        log("   🔄 Using htdemucs model for vocal isolation...")
        progress(0.10, "Separating Vocals...")
        
        bg_track_path = separate_vocals(audio_path, Config.TEMP_DIR)
        if bg_track_path:
            log(f"   ✅ Background track saved to: {bg_track_path}")
        else:
            log("   ⚠️ Vocal separation failed, continuing without background track")
        
        check_stop()
        # 2. Transcribe Audio
        log("")
        log("📝 [Step 3/6] Transcribing Audio to Text...")
        log(f"   🔄 Using STT Engine: {stt_engine}")
        progress(0.15, "Transcribing...")

        segments, detected_source_lang = transcribe_audio(
            audio_path,
            model_size=Config.WHISPER_MODEL_SIZE,
            source_lang=source_lang,
            engine=stt_engine,
            log_callback=log
        )
        final_source_lang = source_lang if source_lang != "auto" else detected_source_lang
        log(f"   ✅ Source Language: {final_source_lang} (Detected: {detected_source_lang})")
        log(f"   ✅ Found {len(segments)} speech segments")

        out_base = os.path.splitext(output_file)[0]
        try:
            from utils import export_subtitles_to_srt
            orig_srt = f"{out_base}_source_{final_source_lang}.srt"
            export_subtitles_to_srt(segments, orig_srt, text_key="text")
            log(f"   💾 Original subtitle (.srt) saved: {orig_srt}")
        except Exception as e:
            log(f"   ⚠️ Could not export source .srt: {e}")

        check_stop()
        # 3. Detect Gender
        log("")
        log("👤 [Step 4/6] Analyzing Voice Genders...")
        log(f"   🔄 Method: {Config.GENDER_DETECTION_METHOD}")
        log(f"   🔄 Processing {len(segments)} segments...")
        progress(0.30, "Analyzing Gender...")
        
        # Choose classifier based on config
        if Config.GENDER_DETECTION_METHOD in ["visual", "hybrid"]:
            try:
                from modules.visual_gender_classifier import visual_classifier
                use_visual = True
                log(f"   🎥 Using visual gender detection (face analysis)")
                visual_classifier.load_video(input_file)
            except Exception as e:
                log(f"WARNING: Visual classifier failed to initialize: {e}. Falling back to audio.")
                use_visual = False
        else:
            use_visual = False
        
        if use_visual and Config.GENDER_DETECTION_METHOD == "visual":
            male_count = 0
            female_count = 0
            for i, seg in enumerate(segments):
                if stop_event and stop_event.is_set(): check_stop()
                gender = visual_classifier.detect_gender(input_file, seg['start'], seg['end'])
                if gender == 'unknown':
                    gender = 'female'
                seg['gender'] = gender
                if gender.lower() == "male":
                    male_count += 1
                else:
                    female_count += 1
                if i % 10 == 0:
                    progress(0.30 + (0.1 * (i/len(segments))), f"Analyzing seg {i+1}/{len(segments)}")
        else:
            use_clustering = getattr(Config, "SPEAKER_CLUSTERING_GENDER", True)
            if hasattr(classifier, 'analyze_all_segments'):
                segments = classifier.analyze_all_segments(
                    audio_path,
                    segments,
                    use_clustering=use_clustering,
                    lang=source_lang,
                    log_callback=log
                )
            else:
                for i, seg in enumerate(segments):
                    if stop_event and stop_event.is_set(): check_stop()
                    seg['gender'] = classifier.detect_gender(audio_path, seg['start'], seg['end'])

            male_count = sum(1 for s in segments if s.get('gender') == 'male')
            female_count = len(segments) - male_count

        if use_visual:
            visual_classifier.cleanup()
        if hasattr(classifier, 'unload'):
            classifier.unload()

        log(f"   ✅ Gender analysis complete: {male_count} Male, {female_count} Female")
        
        check_stop()
        # 4. Translate Text
        log("")
        log(f"🌐 [Step 5/6] Translating Text to {target_lang.upper()}...")
        
        if Config.TRANSLATION_PROVIDER == "local-ctranslate2":
            model_info = f"local-ctranslate2 ({Config.CT2_NLLB_MODEL})"
        elif Config.TRANSLATION_PROVIDER == "ollama":
            model_info = f"ollama ({Config.OLLAMA_MODEL})"
        elif Config.TRANSLATION_PROVIDER == "local-transformer":
            model_info = f"local-transformer ({Config.NLLB_MODEL})"
        else:
            model_info = Config.TRANSLATION_PROVIDER
        
        log(f"   🔄 Using {model_info}")
        if Config.TRANSLATION_PROVIDER == "ollama":
            log(f"   🔄 Temperature: {translation_temperature}")
        progress(0.45, "Translating...")
        
        translated_segments = translate_text(
            segments, 
            target_lang=target_lang, 
            source_lang=final_source_lang,
            stop_event=stop_event, 
            temperature=translation_temperature,
            log_callback=log,
            enable_multitask=enable_multitask,
            max_workers=max_workers
        )
        log(f"   ✅ Translation complete for {len(translated_segments)} segments")

        try:
            from utils import export_subtitles_to_srt
            trans_srt = f"{out_base}_{target_lang}.srt"
            export_subtitles_to_srt(translated_segments, trans_srt, text_key="translated_text")
            log(f"   💾 Translated subtitle (.srt) saved: {trans_srt}")
        except Exception as e:
            log(f"   ⚠️ Could not export translated .srt: {e}")

        if Config.TRANSLATION_PROVIDER == "local-ctranslate2":
            log(f"   ✅ CTranslate2 model unloaded - VRAM released")
        elif Config.TRANSLATION_PROVIDER == "ollama":
            log(f"   ✅ Ollama model unloaded - VRAM released")
        elif Config.TRANSLATION_PROVIDER == "local-transformer":
            log(f"   ✅ NLLB model unloaded - VRAM released")
        
        check_stop()
        # 5. Generate Speech
        log("")
        log("🎤 [Step 6/6] Generating Dubbed Voice...")
        log(f"   🔄 Using TTS Provider: {Config.TTS_PROVIDER} (Lang: {tts_lang})")
        log(f"   🔄 Male pitch: {male_pitch}, Female pitch: {female_pitch}")
        progress(0.60, "Generating Voice...")
        
        audio_segments = generate_voice(
            translated_segments, 
            original_audio_path=audio_path, 
            output_dir=Config.TEMP_DIR,
            male_pitch=male_pitch,
            female_pitch=female_pitch,
            stop_event=stop_event,
            tts_lang=tts_lang,
            log_callback=log,
            enable_multitask=enable_multitask,
            max_workers=max_workers,
            use_clone=use_clone,
            clone_mode=clone_mode,
            reuse_speaker_voice=reuse_speaker_voice
        )
        successful_clips = sum(1 for f in audio_segments if f)
        log(f"   ✅ Voice generation complete: {successful_clips}/{len(translated_segments)} audio clips")
        
        check_stop()
        # 6. Merge Video
        log("")
        log("🎬 [Final] Merging Video with New Audio...")
        progress(0.85, "Merging Video...")
        
        merge_video(
            input_file,
            audio_segments,
            translated_segments,
            output_file,
            background_audio_path=bg_track_path,
            original_audio_path=audio_path,
            dual_audio=dual_audio,
            target_lang=target_lang
        )
        
        check_stop()
        log("")
        log("="*50)
        log(f"🎉 SUCCESS! Dubbing Complete!")
        log(f"📂 Output saved to: {output_file}")
        log("="*50)
        progress(1.0, "Complete!")
        return True
        
    except InterruptedError:
        log("Stopped.")
        return False
    except Exception as e:
        log(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        cleanup_temp(Config.TEMP_DIR)

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="AI Video Dubbing Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input video file path")
    parser.add_argument("--output", "-o", help="Output video file path")
    parser.add_argument("--language", "-l", default="th", help="Target language code (default: th)")
    
    args = parser.parse_args()
    
    # Determine output path
    if not args.output:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_dubbed_{args.language}{ext}"

    run_pipeline(args.input, args.output, args.language)

if __name__ == "__main__":
    main()
