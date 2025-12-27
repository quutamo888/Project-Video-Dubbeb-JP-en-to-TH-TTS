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
    try:
        logger.info(f"Separating vocals for: {audio_path}")
        
        # Demucs outputs to: <output_dir>/htdemucs/<filename>/...
        # We use the htdemucs model (default, fast, good quality)
        
        # Check if demucs is installed
        try:
            subprocess.run(["demucs", "--help"], capture_output=True, check=True)
        except (output, FileNotFoundError):
             # Try python -m demucs
             pass

        # Construct command
        # -n htdemucs: High quality hybrid transformer
        # --two-stems=vocals: Only separate into 'vocals' and 'no_vocals' (faster)
        cmd = [
            "demucs",
            "--two-stems=vocals",
            "-n", "htdemucs", 
            "-o", output_dir,
        ]
        
        if Config.USE_GPU:
            cmd.extend(["-d", "cuda"])
        else:
            cmd.extend(["-d", "cpu"])
            
        cmd.append(audio_path)
        
        # Run Demucs
        # Use shell=True for windows if needed, but list is safer usually. 
        # On windows, sometimes "demucs" isn't in path if just pip installed, might need "python -m demucs"
        
        logger.info(f"Running Demucs: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode != 0:
            logger.error(f"Demucs failed: {result.stderr}")
            # Try fallback to python -m demucs
            cmd[0] = "python"
            cmd.insert(1, "-m")
            cmd.insert(2, "demucs")
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode != 0:
                 raise RuntimeError(f"Demucs separation failed: {result.stderr}")

        # Determine output path
        # Demucs structure: output_dir/htdemucs/audio_filename/no_vocals.wav
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        
        # Demucs might normalize filename? usually strictly follows input basename
        stem_path = os.path.join(output_dir, "htdemucs", filename, "no_vocals.wav")
        
        if os.path.exists(stem_path):
            logger.info(f"Vocal separation complete. Background track: {stem_path}")
            return stem_path
        else:
            logger.error(f"Expected output file not found: {stem_path}")
            return None

    except Exception as e:
        logger.error(f"Error checking/running vocal isolation: {e}")
        return None
