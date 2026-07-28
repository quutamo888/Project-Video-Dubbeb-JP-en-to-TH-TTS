import os
import subprocess
import logging
from config import Config

logger = logging.getLogger(__name__)

def separate_vocals(audio_path, output_dir):
    """
    Separates vocals from the audio using Demucs.
    Returns the path to the background (no_vocals) audio file.
    """
    import sys
    try:
        logger.info(f"Separating vocals for: {audio_path}")
        
        # Use sys.executable to run demucs to ensure we use the virtualenv's installation
        cmd = [
            sys.executable,
            "-m",
            "demucs",
            "--two-stems=vocals",
            "-n", "htdemucs", 
            "-o", output_dir,
            "--mp3",
        ]
        
        if Config.USE_GPU:
            cmd.extend(["-d", "cuda"])
        else:
            cmd.extend(["-d", "cpu"])
            
        cmd.append(audio_path)
        
        logger.info(f"Running Demucs: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode != 0:
             raise RuntimeError(f"Demucs separation failed: {result.stderr}")

        # Determine output path
        # Demucs structure: output_dir/htdemucs/audio_filename/no_vocals.mp3
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        stem_path = os.path.join(output_dir, "htdemucs", filename, "no_vocals.mp3")
        
        if os.path.exists(stem_path):
            logger.info(f"Vocal separation complete. Background track: {stem_path}")
            return stem_path
        else:
            logger.error(f"Expected output file not found: {stem_path}")
            return None

    except Exception as e:
        logger.error(f"Error running vocal isolation: {e}")
        return None
