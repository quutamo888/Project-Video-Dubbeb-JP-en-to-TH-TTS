import os
import shutil
import logging

import sys

def setup_logging():
    # Force UTF-8 on Windows console
    if sys.platform == "win32" and hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("dubbing.log", encoding='utf-8')
        ]
    )

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
