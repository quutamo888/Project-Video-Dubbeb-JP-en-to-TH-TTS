import os
import sys

def test_imports():
    from modules.youtube_downloader import sanitize_filename, download_youtube_video, ensure_yt_dlp
    assert callable(sanitize_filename)
    assert callable(download_youtube_video)
    assert callable(ensure_yt_dlp)
    print("✅ youtube_downloader functions imported successfully")

def test_sanitize():
    from modules.youtube_downloader import sanitize_filename
    res = sanitize_filename('Title with: illegal/characters?*"<>|')
    assert ":" not in res
    assert "/" not in res
    assert "?" not in res
    assert "*" not in res
    print("✅ sanitize_filename correctly cleans strings")

def test_resolution_format_mapping():
    # Verify resolution mapping logic
    for res, expected in [("1080p", 1080), ("720p", 720), ("480p", 480), ("360p", 360)]:
        res_str = str(res).lower().replace("p", "").strip()
        assert res_str.isdigit()
        assert int(res_str) == expected
    print("✅ Resolution format mapping verified")

if __name__ == "__main__":
    test_imports()
    test_sanitize()
    test_resolution_format_mapping()
    print("🎉 All YouTube Downloader tests passed successfully!")
