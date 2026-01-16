import argparse
import os
import sys

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
                 translation_temperature=0.3, stop_event=None):
    """
    Executes the full dubbing pipeline.
    """
    def log(msg):
        print(msg)
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
            
        ensure_dirs([Config.TEMP_DIR, Config.OUTPUT_DIR])
        
        log("="*50)
        log(f"🎬 Starting AI Video Dubbing Pipeline")
        log(f"📂 Input: {input_file}")
        log(f"📂 Output: {output_file}")
        log(f"🌐 Target Language: {language}")
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
        log(f"   🔄 Using Whisper model: {Config.WHISPER_MODEL_SIZE}")
        progress(0.15, "Transcribing...")
        
        segments, source_lang = transcribe_audio(audio_path, model_size=Config.WHISPER_MODEL_SIZE)
        log(f"   ✅ Detected Source Language: {source_lang}")
        log(f"   ✅ Found {len(segments)} speech segments")
        
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
                # Load video for visual classifier
                visual_classifier.load_video(input_file)
            except Exception as e:
                # Assuming 'logger' is defined elsewhere or needs to be imported/defined
                # For now, using 'log' as a fallback
                log(f"WARNING: Visual classifier failed to initialize: {e}. Falling back to audio.")
                use_visual = False
        else:
            use_visual = False
        
        # Load audio classifier if needed
        if not use_visual or Config.GENDER_DETECTION_METHOD == "hybrid":
            if hasattr(classifier, 'load_audio'):
                classifier.load_audio(audio_path)
            
        male_count = 0
        female_count = 0
        for i, seg in enumerate(segments):
            if stop_event and stop_event.is_set(): check_stop()
            
            gender = 'unknown'
            
            # Try visual detection first (if configured)
            if use_visual:
                gender = visual_classifier.detect_gender(input_file, seg['start'], seg['end'])
            
            # Fallback to audio if visual failed or hybrid mode
            if gender == 'unknown' and Config.GENDER_DETECTION_METHOD in ["audio", "hybrid"]:
                gender = classifier.detect_gender(audio_path, seg['start'], seg['end'])
            
            # Default to female if still unknown
            if gender == 'unknown':
                gender = 'female'
            
            seg['gender'] = gender
            if gender.lower() == "male":
                male_count += 1
            else:
                female_count += 1
            if i % 10 == 0:
                progress(0.30 + (0.1 * (i/len(segments))), f"Analyzing seg {i+1}/{len(segments)}")
        
        # Cleanup visual classifier
        if use_visual:
            visual_classifier.cleanup()
            
        log(f"   ✅ Gender analysis complete: {male_count} Male, {female_count} Female")
        
        check_stop()
        # 4. Translate Text
        log("")
        log(f"🌐 [Step 5/6] Translating Text to {language.upper()}...")
        
        # Show correct model name based on provider
        if Config.TRANSLATION_PROVIDER == "ollama":
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
            target_lang=language, 
            source_lang=source_lang,
            stop_event=stop_event, 
            temperature=translation_temperature
        )
        log(f"   ✅ Translation complete for {len(translated_segments)} segments")
        if Config.TRANSLATION_PROVIDER == "ollama":
            log(f"   ✅ Ollama model unloaded - VRAM released")
        elif Config.TRANSLATION_PROVIDER == "local-transformer":
            log(f"   ✅ NLLB model unloaded - VRAM released")
        
        check_stop()
        # 5. Generate Speech
        log("")
        log("🎤 [Step 6/6] Generating Dubbed Voice...")
        log(f"   🔄 Using TTS Provider: {Config.TTS_PROVIDER}")
        log(f"   🔄 Male pitch: {male_pitch}, Female pitch: {female_pitch}")
        progress(0.60, "Generating Voice...")
        
        audio_segments = generate_voice(
            translated_segments, 
            original_audio_path=audio_path, 
            output_dir=Config.TEMP_DIR,
            male_pitch=male_pitch,
            female_pitch=female_pitch,
            stop_event=stop_event
        )
        log(f"   ✅ Voice generation complete: {len(audio_segments)} audio clips")
        
        check_stop()
        # 6. Merge Video
        log("")
        log("🎬 [Final] Merging Video with New Audio...")
        progress(0.85, "Merging Video...")
        
        merge_video(input_file, audio_segments, translated_segments, output_file, background_audio_path=bg_track_path)
        
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
