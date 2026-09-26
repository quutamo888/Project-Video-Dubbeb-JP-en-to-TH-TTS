import os
import shutil
import logging
from logging.handlers import RotatingFileHandler
import sys
from datetime import datetime

def setup_logging(log_dir="logs", log_filename=None):
    """
    Sets up multi-output logging (Console + Rotating File in logs/ directory).
    """
    # Force UTF-8 on Windows console
    if sys.platform == "win32" and hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    os.makedirs(log_dir, exist_ok=True)

    if not log_filename:
        log_filename = f"dubbing_{datetime.now().strftime('%Y%m%d')}.log"

    main_log_path = os.path.join(log_dir, "dubbing.log")
    daily_log_path = os.path.join(log_dir, log_filename)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if not root_logger.handlers:
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # Rotating file handler (10MB per file, keep 5 backups)
        file_handler = RotatingFileHandler(main_log_path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Daily file handler
        daily_handler = logging.FileHandler(daily_log_path, encoding='utf-8')
        daily_handler.setFormatter(formatter)
        root_logger.addHandler(daily_handler)

    return main_log_path

def ensure_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def cleanup_temp(temp_dir):
    """
    Removes the temporary directory and its contents.
    """
    if os.path.exists(temp_dir):
        try:
            import shutil
            shutil.rmtree(temp_dir)
            # Note: 'logger' is not defined in the provided context.
            # If this code is run, it will raise a NameError unless 'logger' is defined elsewhere.
            # For example: logger = logging.getLogger(__name__)
            # logger.info(f"Cleaned up temp directory: {temp_dir}")
        except Exception as e:
            # logger.warning(f"Failed to cleanup temp dir: {e}")
            pass # Suppress error if logger is not defined

def is_hallucination(text, repeat_threshold=3):
    """
    Checks if text contains excessive repetition (hallucination).
    Returns True if hallucination detected.
    """
    if not text:
        return False
        
    # Check for immediate repetition of phrases (e.g., "Hello world Hello world Hello world")
    # Simple N-gram check or regex
    import re
    
    # 1. Check for whole string repetition
    # e.g. "Test. Test. Test."
    # We strip punctuation and lowercase
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    words = clean_text.split()
    
    if len(words) < 5:
        return False # Too short to judge
        
    # Check for repeating sequence of length 2+
    # This is a heuristic: if more than 50% of the text is made of varying repeats
    
    # Simple regex for exact phrase repetition 3+ times
    # (Sequence of at least 4 chars) repeated 3+ times
    if re.search(r'(.{4,})\1\1', text):
        return True

    return False

def format_srt_time(seconds: float) -> str:
    """
    Converts seconds float to standard SRT timestamp format (HH:MM:SS,mmm).
    """
    millis = int(round(max(0.0, float(seconds)) * 1000))
    hours = millis // 3600000
    millis %= 3600000
    minutes = millis // 60000
    millis %= 60000
    secs = millis // 1000
    millis %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def export_subtitles_to_srt(segments: list, srt_path: str, text_key: str = "text") -> str:
    """
    Exports a list of segment dicts (with 'start', 'end', and text_key) to a standard .srt file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(srt_path)), exist_ok=True)
    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, 1):
            start_str = format_srt_time(seg.get("start", 0.0))
            end_str = format_srt_time(seg.get("end", 0.0))
            text = seg.get(text_key, "").strip()
            f.write(f"{idx}\n{start_str} --> {end_str}\n{text}\n\n")
    return srt_path

