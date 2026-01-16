import os
import logging
from pydub import AudioSegment
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip

logger = logging.getLogger(__name__)

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

def merge_video(video_path, audio_files, segments, output_path, background_audio_path=None):
    """
    Combines the generated audio files into a single track synchronized with the video.
    Optionally overlays onto a background track.
    
    Args:
        video_path (str): Path to original video.
        audio_files (list): List of paths to generated audio files.
        segments (list): List of segment dicts (start, end).
        output_path (str): Final output video path.
        background_audio_path (str): Path to background/no_vocals audio.
    """
    logger.info("Merging audio and video...")
    
    # Create an empty audio track with the duration of the video
    video = VideoFileClip(video_path)
    video_duration = video.duration
    
    # Base Audio
    if background_audio_path and os.path.exists(background_audio_path):
        logger.info(f"Using background track: {background_audio_path}")
        try:
            background = AudioSegment.from_file(background_audio_path)
            # Ensure background matches video duration
            if len(background) > video_duration * 1000:
                background = background[:int(video_duration * 1000)]
            else:
                # Pad silence if short
                background += AudioSegment.silent(duration=int(video_duration * 1000) - len(background))
            
            # Lower background volume slightly to let vocals shine
            combined_audio = background - 3 # reduce by 3dB
        except Exception as e:
            logger.error(f"Failed to load background: {e}")
            combined_audio = AudioSegment.silent(duration=int(video_duration * 1000))
    else:
        combined_audio = AudioSegment.silent(duration=int(video_duration * 1000))
    
    for i, (audio_file, seg) in enumerate(zip(audio_files, segments)):
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
        
        # Placement
        # Note: overlay puts the audio at the specified position
        combined_audio = combined_audio.overlay(audio_segment, position=start_time)
        
    # Export full audio to temp file
    base_name, _ = os.path.splitext(output_path)
    temp_audio_path = f"{base_name}_temp_audio.mp3"
    combined_audio.export(temp_audio_path, format="mp3")
    
    # Combine with video using moviepy
    logger.info("Muxing final video...")
    
    # We load audio clip
    new_audioclip = AudioFileClip(temp_audio_path)
    
    # If using background, we can assume the duration is correct.
    # When setting audio, ensure we don't accidentally cut the video or audio short
    final_video = video.with_audio(new_audioclip)
    
    # Configure codec based on hardware
    # libx264 is CPU, h264_nvenc is NVIDIA GPU
    from config import Config
    video_codec = "h264_nvenc" if Config.USE_GPU else "libx264"
    
    # Add preset for speed if using NVENC, or CPU preset
    preset = "p4" if Config.USE_GPU else "ultrafast" 
    
    logger.info(f"Writing video using codec: {video_codec}...")
    
    try:
        final_video.write_videofile(
            output_path, 
            codec=video_codec, 
            audio_codec="aac",
            preset=preset, 
            threads=4 if not Config.USE_GPU else None # Threads only matter for CPU
        )
    except Exception as e:
        logger.error(f"Failed with {video_codec}, falling back to CPU (libx264)...")
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", preset="ultrafast")
    
    # Cleanup
    if os.path.exists(temp_audio_path):
        try:
            os.remove(temp_audio_path)
        except:
            pass
        
    logger.info(f"Video saved to {output_path}")
    return output_path
