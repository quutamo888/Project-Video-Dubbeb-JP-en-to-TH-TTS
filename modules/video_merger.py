import os
import logging
import subprocess
from pydub import AudioSegment
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip

logger = logging.getLogger(__name__)

def has_audio_stream(video_path):
    """Checks if the video file contains an audio stream using ffprobe or ffmpeg."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return "audio" in result.stdout.lower()
    except Exception:
        try:
            cmd = ["ffmpeg", "-i", video_path]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return "Audio:" in result.stderr
        except Exception:
            return True

def mux_video_ffmpeg(video_path, dubbed_audio_path, output_path, dual_audio=True, target_lang="th"):
    """
    Fast and lossless muxing using FFmpeg stream copy (-c:v copy).
    - If dual_audio=True and video has audio:
        Track 1: Original audio (preserved)
        Track 2: Dubbed audio (set as default playback track)
    - If dual_audio=False (or source video has no audio):
        Track 1: Dubbed audio
    """
    source_has_audio = has_audio_stream(video_path)
    tgt_title = "Thai Dubbed (OmniVoice)" if str(target_lang).lower() in ["th", "tha"] else f"{str(target_lang).upper()} Dubbed"

    if dual_audio and source_has_audio:
        logger.info("🎬 Direct FFmpeg Muxing: Dual Audio Tracks (Track 1: Original, Track 2: Dubbed [Default])...")
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", dubbed_audio_path,
            "-map", "0:v:0",
            "-map", "0:a:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a:0", "copy",
            "-c:a:1", "aac",
            "-b:a:1", "192k",
            "-metadata:s:a:0", "title=Original Audio",
            "-metadata:s:a:1", f"title={tgt_title}",
            "-disposition:a:0", "none",
            "-disposition:a:1", "default",
            "-shortest",
            output_path
        ]
    else:
        logger.info("🎬 Direct FFmpeg Muxing: Single Audio Track (Dubbed)...")
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", dubbed_audio_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-metadata:s:a:0", f"title={tgt_title}",
            "-disposition:a:0", "default",
            "-shortest",
            output_path
        ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"✅ Fast FFmpeg muxing completed successfully: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode('utf-8', errors='replace') if e.stderr else str(e)
        logger.warning(f"Fast FFmpeg stream copy failed: {err_msg}. Falling back to MoviePy...")
        return False

def speed_change(sound, speed=1.0):
    # Manually override the frame_rate to get the right duration
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
        "frame_rate": int(sound.frame_rate * speed)
    })
    # Convert the frame rate back to standard to keep the pitch same? 
    # Actually pydub's simple speed change changes pitch. 
    # For complex time stretching without pitch shift, we need audiostretchy or complex algorithm.
    # For simplicity in this v1, checking if pydub allows frame_rate hack which changes pitch.
    # To KEEP pitch, we might need ffmpeg (atempo filter).
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)

def build_smart_spliced_track(video_duration_ms, segments, background_audio_path=None, original_audio_path=None):
    """
    Constructs a base audio track using Smart Splicing:
    - During speech segments: Uses isolated background audio (no_vocals from Demucs) reduced by 3dB so TTS is clear without ghost original voice.
    - During non-speech intervals (between sentences, laughs, reactions, sfx): Uses original audio with smooth crossfade transitions to preserve laughter, impacts, and ambient atmosphere.
    """
    total_ms = int(video_duration_ms)

    original = None
    if original_audio_path and os.path.exists(original_audio_path):
        try:
            original = AudioSegment.from_file(original_audio_path)
            if len(original) < total_ms:
                original += AudioSegment.silent(duration=total_ms - len(original))
            else:
                original = original[:total_ms]
        except Exception as e:
            logger.warning(f"Could not load original audio for smart splicing: {e}")
            original = None

    background = None
    if background_audio_path and os.path.exists(background_audio_path):
        try:
            background = AudioSegment.from_file(background_audio_path)
            if len(background) < total_ms:
                background += AudioSegment.silent(duration=total_ms - len(background))
            else:
                background = background[:total_ms]
            background = background - 3  # gentle 3dB attenuation for speech clarity
        except Exception as e:
            logger.warning(f"Could not load background audio: {e}")
            background = None

    # Fallbacks if one or both are missing
    if background is None and original is None:
        return AudioSegment.silent(duration=total_ms)
    if background is None:
        return original
    if original is None:
        return background

    # 1. Identify speech time spans (with padding to prevent clipping start/end of speech)
    raw_spans = []
    for seg in segments:
        s = max(0, int(seg['start'] * 1000) - 80)
        e = min(total_ms, int(seg['end'] * 1000) + 80)
        # Check if actual generated TTS audio is longer than segment end
        audio_file = seg.get('audio_file')
        if audio_file and os.path.exists(audio_file):
            try:
                tts_dur = len(AudioSegment.from_file(audio_file))
                e = max(e, min(total_ms, int(seg['start'] * 1000) + tts_dur + 80))
            except Exception:
                pass
        if e > s:
            raw_spans.append((s, e))

    # Merge overlapping speech spans
    raw_spans.sort(key=lambda x: x[0])
    speech_spans = []
    for s, e in raw_spans:
        if speech_spans and s <= speech_spans[-1][1]:
            speech_spans[-1] = (speech_spans[-1][0], max(speech_spans[-1][1], e))
        else:
            speech_spans.append((s, e))

    # 2. Build non-speech spans (gaps between speech)
    non_speech_spans = []
    curr = 0
    for s, e in speech_spans:
        if s > curr:
            non_speech_spans.append((curr, s))
        curr = max(curr, e)
    if curr < total_ms:
        non_speech_spans.append((curr, total_ms))

    # 3. Create full-timeline track
    base_track = background
    pad = 40  # 40ms crossfade overlap
    preserved_count = 0

    for gap_start, gap_end in non_speech_spans:
        gap_dur = gap_end - gap_start
        if gap_dur < 150:
            # Skip micro-gaps shorter than 150ms to maintain stability
            continue

        s_pad = max(0, gap_start - pad)
        e_pad = min(total_ms, gap_end + pad)
        slice_dur = e_pad - s_pad

        orig_slice = original[s_pad:e_pad]
        fade_time = min(pad * 2, slice_dur // 3)
        if fade_time > 0:
            orig_slice = orig_slice.fade_in(fade_time).fade_out(fade_time)

        # Splice original audio slice into non-speech interval
        base_track = base_track[:s_pad] + orig_slice + base_track[e_pad:]
        preserved_count += 1

    logger.info(f"✅ Smart Splicing active: preserved {preserved_count} ambient audio intervals")
    return base_track

def merge_video(video_path, audio_files, segments, output_path, background_audio_path=None, original_audio_path=None, dual_audio=None, target_lang="th"):
    """
    Combines the generated audio files into a single track synchronized with the video.
    Optionally overlays onto a background track.
    Fast direct FFmpeg stream muxing is used to support Dual Audio Tracks (Original + Dubbed)
    and instant lossless export without re-encoding.

    Args:
        video_path (str): Path to original video.
        audio_files (list): List of paths to generated audio files.
        segments (list): List of segment dicts (start, end).
        output_path (str): Final output video path.
        background_audio_path (str): Path to background/no_vocals audio.
        original_audio_path (str): Path to original extracted audio for smart ambient preservation.
        dual_audio (bool): If True, preserves original audio as Track 1 and sets dubbed audio as Track 2 (default).
        target_lang (str): Target language for track metadata (e.g. 'th').
    """
    from config import Config
    if dual_audio is None:
        dual_audio = getattr(Config, "DUAL_AUDIO_TRACKS", True)

    logger.info(f"Merging audio and video (Dual Audio: {dual_audio})...")

    # Create an empty audio track with the duration of the video
    video = VideoFileClip(video_path)
    video_duration = video.duration
    video_duration_ms = int(video_duration * 1000)

    # Base Audio with Smart Splicing
    combined_audio = build_smart_spliced_track(
        video_duration_ms=video_duration_ms,
        segments=segments,
        background_audio_path=background_audio_path,
        original_audio_path=original_audio_path
    )

    for i, seg in enumerate(segments):
        # 1. Prefer audio_file directly bound to segment
        audio_file = seg.get('audio_file')
        # 2. Fallback to index in audio_files list
        if not audio_file and audio_files and i < len(audio_files):
            audio_file = audio_files[i]

        if not audio_file or not os.path.exists(audio_file):
            continue

        start_time = seg['start'] * 1000 # to ms
        end_time = seg['end'] * 1000
        target_duration = end_time - start_time

        try:
            audio_segment = AudioSegment.from_file(audio_file)
        except Exception as e:
            logger.warning(f"Could not load audio file {audio_file}: {e}. Skipping.")
            continue

        current_duration = len(audio_segment)

        # Check available space before next segment to prevent overlapping next speaker
        next_start_time = (segments[i + 1]['start'] * 1000) if i + 1 < len(segments) else (video_duration * 1000)
        available_window = max(target_duration, next_start_time - start_time)

        # If audio overruns next speaker's start time by >300ms, adjust playback rate gently (up to 1.25x)
        if current_duration > available_window + 300 and available_window > 500:
            speed_ratio = min(current_duration / available_window, 1.25)
            try:
                audio_segment = speed_change(audio_segment, speed=speed_ratio)
            except Exception as se:
                logger.debug(f"Speed adjustment skipped for seg {i+1}: {se}")

        # Placement: overlay puts the audio at the exact specified start_time
        combined_audio = combined_audio.overlay(audio_segment, position=start_time)

    # Export full audio to temp file
    base_name, _ = os.path.splitext(output_path)
    temp_audio_path = f"{base_name}_temp_audio.mp3"
    combined_audio.export(temp_audio_path, format="mp3")

    # Close video clip to release handle before FFmpeg touches it
    video.close()

    # 1. Try Fast Direct FFmpeg Muxing (Lossless stream copy + Multi-Audio tracks)
    mux_success = mux_video_ffmpeg(
        video_path=video_path,
        dubbed_audio_path=temp_audio_path,
        output_path=output_path,
        dual_audio=dual_audio,
        target_lang=target_lang
    )

    if mux_success:
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass
        logger.info(f"Video saved successfully via FFmpeg to {output_path}")
        return output_path

    # 2. Fallback to MoviePy re-encoding if direct FFmpeg mux failed
    logger.info("Muxing final video using MoviePy fallback...")
    video = VideoFileClip(video_path)
    new_audioclip = AudioFileClip(temp_audio_path)
    final_video = video.with_audio(new_audioclip)

    video_codec = "h264_nvenc" if Config.USE_GPU else "libx264"
    preset = "p4" if Config.USE_GPU else "ultrafast"

    logger.info(f"Writing video using codec: {video_codec}...")
    try:
        final_video.write_videofile(
            output_path,
            codec=video_codec,
            audio_codec="aac",
            preset=preset,
            threads=4 if not Config.USE_GPU else None
        )
    except Exception as e:
        logger.error(f"Failed with {video_codec}, falling back to CPU (libx264)...")
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", preset="ultrafast")

    video.close()
    new_audioclip.close()

    if os.path.exists(temp_audio_path):
        try:
            os.remove(temp_audio_path)
        except Exception:
            pass

    logger.info(f"Video saved to {output_path}")
    return output_path
