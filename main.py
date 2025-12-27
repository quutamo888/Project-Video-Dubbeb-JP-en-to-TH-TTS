import argparse
import os
import sys
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
                 male_pitch=0.0, female_pitch=0.0, stop_event=None):
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
        
        log(f"Processing {input_file}...")
        progress(0.05, "Extracting Audio...")
        
        check_stop()
        # 1. Extract Audio
        audio_path = extract_audio(input_file, Config.TEMP_DIR)
        
        check_stop()
        # 1.5 Separate Vocals (Demucs)
        # This can reduce hallucinations in translation too if we transcribe clean voice, 
        # but primarily for background preservation.
        log("Separating background music (Demucs)...")
        progress(0.10, "Separating Vocals...")
        
        bg_track_path = separate_vocals(audio_path, Config.TEMP_DIR)
        
        check_stop()
        # 2. Transcribe Audio
        # NOTE: Transcribing the ISOLATED vocals usually yields better results!
        # But Demucs vocals path is: <temp>/htdemucs/<name>/vocals.wav (need to find it)
        # For now, let's transcribe original to be safe, or transcribe vocals if found.
        
        log("Transcribing audio...")
        progress(0.15, "Transcribing...")
        segments = transcribe_audio(audio_path, model_size=Config.WHISPER_MODEL_SIZE)
        
        check_stop()
        # 3. Detect Gender
        log("Analyzing voice genders...")
        progress(0.30, "Analyzing Gender...")
        # Load audio once for classifier if optimized
        if hasattr(classifier, 'load_audio'):
            classifier.load_audio(audio_path)
            
        for i, seg in enumerate(segments):
            if stop_event and stop_event.is_set(): check_stop()
            gender = classifier.detect_gender(audio_path, seg['start'], seg['end'])
            seg['gender'] = gender
            if i % 10 == 0: # Minor progress updates
                progress(0.30 + (0.1 * (i/len(segments))), f"Analyzing seg {i}/{len(segments)}")
        
        check_stop()
        # 4. Translate Text
        log(f"Translating to {language}...")
        progress(0.45, "Translating...")
        translated_segments = translate_text(segments, target_lang=language, stop_event=stop_event)
        
        check_stop()
        # 5. Generate Speech
        log("Cloning voice...")
        progress(0.60, "Generating Voice...")
        audio_segments = generate_voice(
            translated_segments, 
            original_audio_path=audio_path, 
            output_dir=Config.TEMP_DIR,
            male_pitch=male_pitch,
            female_pitch=female_pitch,
            stop_event=stop_event
        )
        
        check_stop()
        # 6. Merge Video
        log("Merging video...")
        progress(0.85, "Merging Video...")
        merge_video(input_file, audio_segments, translated_segments, output_file, background_audio_path=bg_track_path)
        
        check_stop()
        log(f"Done! Output saved to: {output_file}")
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
        # cleanup_temp(Config.TEMP_DIR)
        pass

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
