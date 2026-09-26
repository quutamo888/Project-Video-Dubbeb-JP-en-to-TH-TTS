import os
import sys
from config import Config
from utils import format_srt_time, export_subtitles_to_srt

def test_config():
    assert Config.STT_ENGINE == "kotoba-whisper", f"Expected 'kotoba-whisper', got {Config.STT_ENGINE}"
    assert "kotoba" in Config.KOTOBA_MODEL, f"Expected Kotoba model reference, got {Config.KOTOBA_MODEL}"
    print("✅ Config check passed: STT_ENGINE is kotoba-whisper")

def test_srt_export():
    import tempfile
    test_segments = [
        {"start": 0.25, "end": 2.15, "text": "こんにちは、元気ですか。"},
        {"start": 3.00, "end": 5.80, "text": "はい、元気です。よろしくお願いします。"}
    ]
    tmp_path = tempfile.mktemp(suffix=".srt")
    export_subtitles_to_srt(test_segments, tmp_path, text_key="text")
    assert os.path.exists(tmp_path)
    with open(tmp_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "00:00:00,250 --> 00:00:02,150" in content
    assert "こんにちは、元気ですか。" in content
    assert "00:00:03,000 --> 00:00:05,800" in content
    os.remove(tmp_path)
    print("✅ Subtitle SRT export check passed")

def test_transcriber_imports():
    from modules.transcriber import transcribe_kotoba_whisper, transcribe_faster_whisper, transcribe_audio
    print("✅ Transcriber functions successfully imported")

if __name__ == "__main__":
    test_config()
    test_srt_export()
    test_transcriber_imports()
    print("🎉 All Kotoba-Whisper unit tests passed!")
