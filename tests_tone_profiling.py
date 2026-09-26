import numpy as np
from modules.voice_generator import VoiceGenerator

def test_extract_acoustic_tone_instruct():
    sr = 16000

    # 1. Low pitch male voice (115 Hz)
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    male_wave = (0.5 * np.sin(2 * np.pi * 115 * t) + 0.2 * np.sin(2 * np.pi * 230 * t)).astype(np.float32)
    instruct_m, f0_m = VoiceGenerator.extract_acoustic_tone_instruct(male_wave, sr=sr, gender="male")
    assert "male" in instruct_m
    assert "low pitch" in instruct_m
    assert 105 <= f0_m <= 125, f"Expected F0 ~ 115, got {f0_m}"
    print(f"✅ Male Tone Profile: {instruct_m} (F0={f0_m:.1f}Hz)")

    # 2. High pitch female voice (240 Hz)
    fem_wave = (0.5 * np.sin(2 * np.pi * 240 * t) + 0.2 * np.sin(2 * np.pi * 480 * t)).astype(np.float32)
    instruct_f, f0_f = VoiceGenerator.extract_acoustic_tone_instruct(fem_wave, sr=sr, gender="female")
    assert "female" in instruct_f
    assert "high pitch" in instruct_f
    assert 230 <= f0_f <= 250, f"Expected F0 ~ 240, got {f0_f}"
    print(f"✅ Female Tone Profile: {instruct_f} (F0={f0_f:.1f}Hz)")

    # 3. Child voice (330 Hz)
    child_wave = (0.5 * np.sin(2 * np.pi * 330 * t) + 0.2 * np.sin(2 * np.pi * 660 * t)).astype(np.float32)
    instruct_c, f0_c = VoiceGenerator.extract_acoustic_tone_instruct(child_wave, sr=sr, gender="female")
    assert "child" in instruct_c
    assert "very high pitch" in instruct_c
    assert 320 <= f0_c <= 345, f"Expected F0 ~ 330, got {f0_c}"
    print(f"✅ Child Tone Profile: {instruct_c} (F0={f0_c:.1f}Hz)")

    # 4. Very low pitch elderly male voice (90 Hz)
    elder_wave = (0.5 * np.sin(2 * np.pi * 90 * t) + 0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    instruct_e, f0_e = VoiceGenerator.extract_acoustic_tone_instruct(elder_wave, sr=sr, gender="male")
    assert "male" in instruct_e
    assert "elderly" in instruct_e
    assert "very low pitch" in instruct_e
    assert 80 <= f0_e <= 100, f"Expected F0 ~ 90, got {f0_e}"
    print(f"✅ Elderly Male Tone Profile: {instruct_e} (F0={f0_e:.1f}Hz)")

if __name__ == "__main__":
    test_extract_acoustic_tone_instruct()
    print("🎉 All Acoustic Tone Profiling tests passed successfully!")
