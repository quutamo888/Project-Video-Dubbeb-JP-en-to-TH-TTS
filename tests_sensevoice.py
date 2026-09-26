import sys
from modules.transcriber import extract_sensevoice_tags, detect_sensevoice_language

def run_checks():
    # 1. Test Happy + Laughter
    raw1 = "<|ja|><|HAPPY|><|LAUGHTER|><|withitn|>ありがとうございます"
    clean1, emo1, ev1 = extract_sensevoice_tags(raw1)
    assert clean1 == "ありがとうございます", f"Expected 'ありがとうございます', got '{clean1}'"
    assert emo1 == "happy", f"Expected 'happy', got '{emo1}'"
    assert ev1 == "laughter", f"Expected 'laughter', got '{ev1}'"
    assert detect_sensevoice_language(raw1) == "ja"

    # 2. Test Sad
    raw2 = "<|en|><|SAD|><|Speech|><|woitn|>I am very sorry."
    clean2, emo2, ev2 = extract_sensevoice_tags(raw2)
    assert clean2 == "I am very sorry.", f"Expected 'I am very sorry.', got '{clean2}'"
    assert emo2 == "sad", f"Expected 'sad', got '{emo2}'"
    assert ev2 is None, f"Expected None, got '{ev2}'"
    assert detect_sensevoice_language(raw2) == "en"

    # 3. Test Angry
    raw3 = "<|zh|><|ANGRY|><|Speech|><|withitn|>你到底想怎么样"
    clean3, emo3, ev3 = extract_sensevoice_tags(raw3)
    assert clean3 == "你到底想怎么样", f"Expected '你到底想怎么样', got '{clean3}'"
    assert emo3 == "angry", f"Expected 'angry', got '{emo3}'"
    assert detect_sensevoice_language(raw3) == "zh"

    # 4. Test Plain text
    raw4 = "Just plain text without tags"
    clean4, emo4, ev4 = extract_sensevoice_tags(raw4)
    assert clean4 == "Just plain text without tags"
    assert emo4 == "neutral"
    assert ev4 is None

    # 5. Test _group_timestamps_to_segments
    from modules.transcriber import _group_timestamps_to_segments
    timestamps = [[500, 1200], [1300, 2000], [2100, 3000]]
    words = ["今日は", "いい天気", "ですね。"]
    grouped = _group_timestamps_to_segments(timestamps, words, "<|ja|><|HAPPY|>今日はいい天気ですね。")
    assert len(grouped) == 1
    assert grouped[0]["start"] == 0.5
    assert grouped[0]["end"] == 3.0
    assert grouped[0]["emotion"] == "happy"
    assert "今日はいい天気ですね。" in grouped[0]["text"]

    # 6. Test SRT Formatting & Export
    from utils import format_srt_time, export_subtitles_to_srt
    import tempfile, os
    assert format_srt_time(0.0) == "00:00:00,000"
    assert format_srt_time(65.123) == "00:01:05,123"
    assert format_srt_time(3661.05) == "01:01:01,050"

    tmp_srt = tempfile.mktemp(suffix=".srt")
    export_subtitles_to_srt([
        {"start": 0.5, "end": 2.1, "text": "Hello world"},
        {"start": 3.0, "end": 5.4, "text": "Goodbye world"}
    ], tmp_srt, text_key="text")
    assert os.path.exists(tmp_srt)
    with open(tmp_srt, "r", encoding="utf-8") as f:
        content = f.read()
    assert "00:00:00,500 --> 00:00:02,100" in content
    assert "Hello world" in content
    assert "00:00:03,000 --> 00:00:05,400" in content
    assert "Goodbye world" in content
    os.remove(tmp_srt)

    # 7. Test Hallucination Deduplication
    import re
    from utils import is_hallucination
    hallucinated = "ไม่ครับ ไม่เลยครับ ไม่ครับ ไม่เลยครับ ไม่ครับ ไม่เลยครับ"
    assert is_hallucination(hallucinated) is True
    deduped = re.sub(r'(.{4,})\1+', r'\1', hallucinated)
    assert len(deduped) < len(hallucinated)

    print("✅ All SenseVoice tag extraction, timestamp grouping, SRT export, and hallucination assertions passed!")

if __name__ == "__main__":
    run_checks()
