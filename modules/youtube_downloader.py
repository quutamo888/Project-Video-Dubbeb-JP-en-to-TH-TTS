import os
import re
import sys
import subprocess
import logging

logger = logging.getLogger(__name__)

def ensure_yt_dlp(log_callback=None):
    """
    Ensures yt-dlp is available in the Python environment.
    If missing, automatically installs it into the active virtual environment.
    """
    try:
        import yt_dlp
        return yt_dlp
    except ImportError:
        msg = "ℹ️ yt-dlp is not yet installed. Auto-installing into virtual environment..."
        logger.info(msg)
        if log_callback:
            log_callback(msg)

        installed = False
        # 1. Try uv pip install
        try:
            subprocess.run(["uv", "pip", "install", "yt-dlp"], check=True)
            installed = True
        except Exception:
            pass

        # 2. Try sys.executable -m pip install
        if not installed:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "yt-dlp"], check=True)
                installed = True
            except Exception as e:
                raise ImportError(
                    f"yt-dlp auto-installation failed: {e}. "
                    f"Please run in terminal: uv pip install yt-dlp"
                )

        import yt_dlp
        logger.info("✅ yt-dlp successfully installed!")
        if log_callback:
            log_callback("✅ yt-dlp successfully installed!")
        return yt_dlp

def sanitize_filename(name):
    """Removes invalid characters for filesystem paths."""
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()

def get_video_info(url):
    """
    Extracts video metadata (title, duration, available resolutions) without downloading.
    """
    yt_dlp = ensure_yt_dlp()

    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info.get('title', 'video')
        duration = info.get('duration', 0)

        # Collect available resolutions
        formats = info.get('formats', [])
        heights = set()
        for f in formats:
            h = f.get('height')
            if h and isinstance(h, int) and h >= 144:
                heights.add(h)
        sorted_heights = sorted(list(heights), reverse=True)

        return {
            'title': title,
            'duration': duration,
            'available_resolutions': sorted_heights
        }

def download_youtube_video(url, resolution="1080p", output_dir="temp", progress_callback=None, log_callback=None):
    """
    Downloads a YouTube video at the requested resolution, merging video and audio into MP4.

    Args:
        url (str): YouTube URL
        resolution (str): e.g. "1080p", "720p", "480p", "360p", or "best"
        output_dir (str): Directory where the downloaded video will be saved
        progress_callback (callable): Optional callback(percent: float 0-1, msg: str)
        log_callback (callable): Optional callback(msg: str)

    Returns:
        str: Absolute path to the downloaded MP4 file
    """
    os.makedirs(output_dir, exist_ok=True)

    def log(msg):
        logger.info(msg)
        if log_callback:
            log_callback(msg)

    yt_dlp = ensure_yt_dlp(log_callback=log)

    def progress_hook(d):
        if d['status'] == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
            downloaded = d.get('downloaded_bytes', 0)
            if total > 0:
                percent = downloaded / total
                eta = d.get('eta', 0)
                speed = d.get('speed', 0)
                speed_str = f"{speed / 1024 / 1024:.1f} MB/s" if speed else ""
                msg = f"Downloading: {percent*100:.1f}% ({speed_str}, ETA: {eta}s)"
                if progress_callback:
                    progress_callback(percent * 0.95, msg)
            else:
                if progress_callback:
                    progress_callback(0.5, "Downloading...")
        elif d['status'] == 'finished':
            if progress_callback:
                progress_callback(0.95, "Merging video and audio...")
            log("   ✅ Video stream downloaded, processing container...")

    # Format selector based on resolution
    res_str = str(resolution).lower().replace("p", "").strip()
    if res_str.isdigit():
        h = int(res_str)
        fmt = f"bestvideo[height<={h}]+bestaudio/best[height<={h}]/best"
    else:
        fmt = "bestvideo+bestaudio/best"

    out_template = os.path.join(output_dir, "%(title).100s [%(id)s].%(ext)s")

    ydl_opts = {
        'format': fmt,
        'outtmpl': out_template,
        'merge_output_format': 'mp4',
        'progress_hooks': [progress_hook],
        'quiet': True,
        'no_warnings': True,
    }

    log(f"🎬 Downloading YouTube video: {url}")
    log(f"   Desired Resolution: {resolution}")
    log(f"   Target Directory: {output_dir}")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        base, _ = os.path.splitext(filename)
        final_mp4 = f"{base}.mp4"

        if not os.path.exists(final_mp4) and os.path.exists(filename):
            final_mp4 = filename

    if progress_callback:
        progress_callback(1.0, "Download Complete!")

    log(f"   🎉 Download completed: {final_mp4}")
    return os.path.abspath(final_mp4)
