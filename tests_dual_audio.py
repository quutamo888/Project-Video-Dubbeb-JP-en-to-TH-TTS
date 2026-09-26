import os
import sys
from config import Config

def test_config():
    assert hasattr(Config, "DUAL_AUDIO_TRACKS"), "Config missing DUAL_AUDIO_TRACKS"
    assert Config.DUAL_AUDIO_TRACKS is True, f"Expected DUAL_AUDIO_TRACKS=True, got {Config.DUAL_AUDIO_TRACKS}"
    print("✅ Config check passed: DUAL_AUDIO_TRACKS is True")

def test_merger_imports():
    from modules.video_merger import has_audio_stream, mux_video_ffmpeg, merge_video
    assert callable(has_audio_stream)
    assert callable(mux_video_ffmpeg)
    assert callable(merge_video)
    print("✅ Video merger functions successfully imported")

def test_ffmpeg_dual_audio_command_generation():
    from modules import video_merger
    executed_commands = []

    def mock_has_audio_stream(video_path):
        return True

    def mock_subprocess_run(cmd, check=True, stdout=None, stderr=None):
        executed_commands.append(cmd)
        class DummyResult:
            stdout = ""
            stderr = b""
        return DummyResult()

    orig_has_audio = video_merger.has_audio_stream
    orig_run = video_merger.subprocess.run

    video_merger.has_audio_stream = mock_has_audio_stream
    video_merger.subprocess.run = mock_subprocess_run

    try:
        ok = video_merger.mux_video_ffmpeg(
            video_path="dummy_input.mp4",
            dubbed_audio_path="dummy_audio.mp3",
            output_path="dummy_output.mp4",
            dual_audio=True,
            target_lang="th"
        )
        assert ok is True
        assert len(executed_commands) == 1
        cmd = executed_commands[0]

        # Check map arguments
        assert "-map" in cmd
        assert "0:v:0" in cmd
        assert "0:a:0" in cmd
        assert "1:a:0" in cmd

        # Check codecs: lossless video copy, original audio copy, dubbed aac
        c_v_idx = cmd.index("-c:v")
        assert cmd[c_v_idx + 1] == "copy"
        ca0_idx = cmd.index("-c:a:0")
        assert cmd[ca0_idx + 1] == "copy"
        ca1_idx = cmd.index("-c:a:1")
        assert cmd[ca1_idx + 1] == "aac"

        # Check titles & dispositions
        assert "title=Original Audio" in cmd
        assert "title=Thai Dubbed (OmniVoice)" in cmd
        disp0_idx = cmd.index("-disposition:a:0")
        assert cmd[disp0_idx + 1] == "none"
        disp1_idx = cmd.index("-disposition:a:1")
        assert cmd[disp1_idx + 1] == "default"

        print("✅ Dual Audio FFmpeg command structure verified!")
    finally:
        video_merger.has_audio_stream = orig_has_audio
        video_merger.subprocess.run = orig_run

if __name__ == "__main__":
    test_config()
    test_merger_imports()
    test_ffmpeg_dual_audio_command_generation()
    print("🎉 All Dual Audio unit tests passed successfully!")
