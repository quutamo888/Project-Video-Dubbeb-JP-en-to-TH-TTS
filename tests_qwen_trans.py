import os
import sys
import tempfile
from config import Config

def test_config():
    assert hasattr(Config, "QWEN_MODEL"), "Config missing QWEN_MODEL"
    assert "Qwen" in Config.QWEN_MODEL, f"Unexpected QWEN_MODEL: {Config.QWEN_MODEL}"
    print(f"✅ Config check passed: QWEN_MODEL is {Config.QWEN_MODEL}")

def test_translator_imports():
    from modules.translator import (
        get_qwen_model_and_tokenizer,
        unload_qwen_model,
        translate_with_qwen,
        translate_srt_file
    )
    assert callable(get_qwen_model_and_tokenizer)
    assert callable(unload_qwen_model)
    assert callable(translate_with_qwen)
    assert callable(translate_srt_file)
    print("✅ Translator Qwen functions successfully imported")

def test_srt_parsing_and_dispatch():
    # Test that translate_srt_file correctly dispatches engine selection
    from modules import translator
    sample_srt = """1
00:00:01,000 --> 00:00:03,000
おはようございます。

2
00:00:03,500 --> 00:00:06,000
今日もいい天気ですね。
"""
    tmp_in = tempfile.mktemp(suffix=".srt")
    tmp_out = tempfile.mktemp(suffix=".srt")
    with open(tmp_in, "w", encoding="utf-8") as f:
        f.write(sample_srt)

    called_qwen = []
    original_translate_with_qwen = translator.translate_with_qwen
    def mock_translate_with_qwen(text, context_history=None, source_lang="ja", target_lang="th", gender="female"):
        called_qwen.append((text, len(context_history) if context_history else 0))
        if "おはよう" in text:
            return "สวัสดีตอนเช้าครับ"
        return "วันนี้อากาศดีจังเลยนะครับ"

    translator.translate_with_qwen = mock_translate_with_qwen
    try:
        out = translator.translate_srt_file(tmp_in, tmp_out, source_lang="ja", target_lang="th", engine="qwen")
        assert os.path.exists(out)
        with open(out, "r", encoding="utf-8") as f:
            content = f.read()
        assert "สวัสดีตอนเช้าครับ" in content
        assert "วันนี้อากาศดีจังเลยนะครับ" in content
        assert len(called_qwen) == 2
        # Second call should have context_history of length 1
        assert called_qwen[1][1] == 1
        print("✅ translate_srt_file with engine='qwen' and rolling context verified!")
    finally:
        translator.translate_with_qwen = original_translate_with_qwen
        if os.path.exists(tmp_in): os.remove(tmp_in)
        if os.path.exists(tmp_out): os.remove(tmp_out)

def test_translate_text_qwen_sequential():
    from modules import translator
    from config import Config

    orig_provider = Config.TRANSLATION_PROVIDER
    Config.TRANSLATION_PROVIDER = "local-qwen"

    segments = [
        {"start": 0.0, "end": 1.0, "text": "こんにちは", "gender": "male"},
        {"start": 1.2, "end": 2.5, "text": "元気ですか", "gender": "male"}
    ]

    called = []
    def mock_qwen(text, context_history=None, **kwargs):
        called.append(text)
        return "ทดสอบ"

    orig_qwen = translator.translate_with_qwen
    translator.translate_with_qwen = mock_qwen
    try:
        # Even with enable_multitask=True, local-qwen must run sequentially
        res = translator.translate_text(segments, enable_multitask=True, max_workers=4)
        assert len(res) == 2
        assert len(called) == 2
        print("✅ translate_text safely runs Qwen sequentially even with multitask enabled!")
    finally:
        translator.translate_with_qwen = orig_qwen
        Config.TRANSLATION_PROVIDER = orig_provider

if __name__ == "__main__":
    test_config()
    test_translator_imports()
    test_srt_parsing_and_dispatch()
    test_translate_text_qwen_sequential()
    print("🎉 All Qwen translation unit tests passed successfully!")
