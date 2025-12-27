import os
import subprocess
import logging

logger = logging.getLogger(__name__)

def extract_audio(video_path, output_dir):
    """
    Extracts audio from a video file using ffmpeg.
    
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the extracted audio.
        
    Returns:
        str: Path to the extracted audio file.
    """
    filename = os.path.basename(video_path)
    name, _ = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}.wav")
    
    if os.path.exists(output_path):
        logger.info(f"Audio already extracted: {output_path}")
        return output_path

    logger.info(f"Extracting audio from {video_path}...")
    
    # Use ffmpeg to extract audio
    # -y: Overwrite output file
    # -vn: Disable video
    # -acodec pcm_s16le: 16-bit PCM audio (WAV)
    # -ar 44100: 44.1kHz sample rate
    # -ac 2: Stereo
    command = [
        "ffmpeg",
        "-i", video_path,
        "-y",
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        output_path
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Audio extracted successfully: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed: {e.stderr.decode()}")
        raise RuntimeError("Failed to extract audio using FFmpeg") from e
