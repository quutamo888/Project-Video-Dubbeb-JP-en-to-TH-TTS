import numpy as np
import os
import sys

def test_linguistic_gender_hints():
    from modules.gender_classifier import GenderClassifier

    # 1. Male Japanese sentences
    m1 = "僕はオープンに言います"
    m_score, f_score = GenderClassifier.detect_linguistic_gender_hints(m1, "ja")
    assert m_score > 0, f"Expected positive male score for '{m1}'"
    assert f_score == 0, f"Expected zero female score for '{m1}'"

    m2 = "俺が絶対に勝つぜ！"
    m_score, f_score = GenderClassifier.detect_linguistic_gender_hints(m2, "ja")
    assert m_score >= 4.0, f"Expected strong male score for pronoun + particle in '{m2}'"

    m3 = "行くぞ！"
    m_score, f_score = GenderClassifier.detect_linguistic_gender_hints(m3, "ja")
    assert m_score > 0

    # 2. Female Japanese sentences
    f1 = "あたしは何も知らないわよ"
    m_score, f_score = GenderClassifier.detect_linguistic_gender_hints(f1, "ja")
    assert f_score >= 4.0, f"Expected strong female score for '{f1}'"
    assert m_score == 0

    f2 = "これでいいのかしら？"
    m_score, f_score = GenderClassifier.detect_linguistic_gender_hints(f2, "ja")
    assert f_score > 0

    # 3. Neutral Japanese sentence
    n1 = "今日はいい天気ですね。"
    m_score, f_score = GenderClassifier.detect_linguistic_gender_hints(n1, "ja")
    assert m_score == 0 and f_score == 0, f"Expected neutral score for '{n1}'"

    # 4. English gender sentences
    e_m = "I am a guy from Tokyo"
    m_score, f_score = GenderClassifier.detect_linguistic_gender_hints(e_m, "en")
    assert m_score > 0

    e_f = "I'm a girl from Osaka"
    m_score, f_score = GenderClassifier.detect_linguistic_gender_hints(e_f, "en")
    assert f_score > 0

    print("✅ test_linguistic_gender_hints passed!")

def test_acoustic_embedding_and_clustering():
    from modules.gender_classifier import GenderClassifier

    clf = GenderClassifier()
    sr = 16000
    clf.sr = sr

    # Create synthetic audio signals (low pitch male 120Hz vs high pitch female 250Hz)
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    male_wave = (0.5 * np.sin(2 * np.pi * 120 * t)).astype(np.float32)
    fem_wave = (0.5 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)

    emb_male1 = clf.extract_speaker_embedding(male_wave)
    emb_male2 = clf.extract_speaker_embedding(male_wave + 0.02 * np.random.randn(len(male_wave)).astype(np.float32))
    emb_fem = clf.extract_speaker_embedding(fem_wave)

    # Embedding shape and unit norm
    assert len(emb_male1) > 0
    assert abs(np.linalg.norm(emb_male1) - 1.0) < 1e-4
    assert abs(np.linalg.norm(emb_fem) - 1.0) < 1e-4

    # Cosine distance between same speaker audio should be much smaller than different speaker audio
    dist_same = 1.0 - float(np.dot(emb_male1, emb_male2))
    dist_diff = 1.0 - float(np.dot(emb_male1, emb_fem))

    assert dist_same < dist_diff, f"Expected dist_same ({dist_same}) < dist_diff ({dist_diff})"
    print("✅ test_acoustic_embedding_and_clustering passed!")

def test_majority_voting_consistency():
    from modules.gender_classifier import GenderClassifier

    clf = GenderClassifier()

    # Mock audio signals and segments
    sr = 16000
    clf.sr = sr
    clf.y = np.zeros(sr * 10, dtype=np.float32)
    clf.current_audio_path = "mock.wav"

    segments = [
        {"start": 0.0, "end": 2.0, "text": "僕がやります"},
        {"start": 2.5, "end": 4.0, "text": "はい、わかりました"},
        {"start": 4.5, "end": 6.0, "text": "行くぞ！"},
    ]

    # Mock cluster_speakers to group all 3 segments into SPEAKER_00
    clf.cluster_speakers = lambda segs, path: ({0: "SPEAKER_00", 1: "SPEAKER_00", 2: "SPEAKER_00"},
                                               [{'id': "SPEAKER_00", 'centroid': np.zeros(64), 'count': 3, 'indices': [0, 1, 2]}])

    # Even if 1 segment returned female pitch, linguistic markers and majority should lock entire speaker to male
    clf.detect_gender = lambda path, s, e: "female" if s == 2.5 else "male"

    results = clf.analyze_all_segments("mock.wav", segments, use_clustering=True, lang="ja")

    # All segments for SPEAKER_00 must be uniformly male
    for seg in results:
        assert seg["speaker"] == "SPEAKER_00"
        assert seg["gender"] == "male", f"Expected 'male', got {seg['gender']}"

    print("✅ test_majority_voting_consistency passed (Zero gender flipping)!")

def test_config_defaults():
    from config import Config
    assert hasattr(Config, "AUDIO_GENDER_MODEL")
    assert hasattr(Config, "ROBUST_GENDER_MODEL")
    assert hasattr(Config, "SPEAKER_CLUSTERING_GENDER")
    assert Config.AUDIO_GENDER_MODEL in ["ml-robust", "ml-librispeech", "pitch"]
    assert Config.SPEAKER_CLUSTERING_GENDER is True
    print("✅ test_config_defaults passed!")

if __name__ == "__main__":
    test_linguistic_gender_hints()
    test_acoustic_embedding_and_clustering()
    test_majority_voting_consistency()
    test_config_defaults()
    print("🎉 All Gender Classifier tests passed successfully!")
