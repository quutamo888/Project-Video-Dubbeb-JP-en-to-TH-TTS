# AI Video Dubbing Pro (JP/EN -> Thai) 🎥🇹🇭

A comprehensive AI-powered video dubbing tool designed for high-quality, **uncensored** Thai localization. It features voice cloning, bridge translation for Japanese content, and a modern GUI with real-time system monitoring.

## 🚀 Key Features

*   **Advanced TTS**: Uses **F5-TTS** (State-of-the-Art) for natural Thai speech and **Voice Cloning** to match the original speaker's tone.
*   **Smart Translation**:
    *   **JP -> EN -> TH Bridge**: Automatically detects Japanese and translates via English for better context.
    *   **Uncensored & Natural**: Tuned for "Spoken Thai" (ภาษาพูด) and supports explicit language without filtering.
    *   **Thai Validator**: Automatically fixes failed translations (English output) by falling back to Google Translate.
*   **Vocal Isolation**: Uses **Demucs** to separate voice from background music/SFX, preserving the original audio atmosphere.
*   **Modern GUI**:
    *   **Real-time Monitor**: View CPU, GPU, and Disk usage while rendering.
    *   **Translation Creativity**: Adjust "Temperature" to control how creative or literal the translation should be.
    *   **Tone Adjustment**: Finetune pitch for Male/Female voices.
*   **Performance**: Supports NVIDIA GPU acceleration (CUDA) for transcoding and inference.

## 🛠️ Prerequisites

*   **Python 3.10+**
*   **FFmpeg**: Must be installed and added to system PATH.
*   **NVIDIA GPU** (Recommended): For fast rendering and F5-TTS/Whisper inference.
*   **Ollama**: (Optional but Recommended) For high-quality local translation.
    *   Install Ollama and pull a model (e.g., `ollama pull llama3` or `openthaigpt`).
    *   Configure URL/Model in `config.py` if needed.

## 📦 Installation

This project uses `uv` for fast package management.

1.  **Clone/Download** this repository.
2.  **Install Dependencies**:
    ```powershell
    uv sync
    ```
    *(This creates a virtual environment in `.venv` and installs all required packages)*

## ▶️ How to Run

### 1. GUI Mode (Recommended)
The easiest way to use the tool.

```powershell
.\.venv\Scripts\python gui.py
```

*   **Input Video**: Browse to your source file (`.mp4`, `.mkv`, etc.).
*   **Output Video**: Choose where to save the dubbed version.
*   **Settings**:
    *   **Translation Temperature**: 0.3 (Standard) to 0.7 (Creative).
    *   **Pitch**: Adjust if voices sound too deep or too high.
*   **Start**: Click "START DUBBING".

### 2. CLI Mode (Advanced)
For automation or headless usage.

```powershell
.\.venv\Scripts\python main.py --input "path/to/video.mp4" --language th
```

## 📂 Project Structure

*   `gui.py`: Main Graphical User Interface.
*   `main.py`: Core pipeline orchestration.
*   `modules/`:
    *   `transcriber.py`: Whisper STT logic.
    *   `translator.py`: Ollama/Google translation logic with Bridge support.
    *   `voice_generator.py`: F5-TTS and voice cloning logic.
    *   `vocal_isolator.py`: Demucs background separation.
    *   `video_merger.py`: MoviePy/FFmpeg video assembly.
*   `temp/`: Temporary processing artifacts (Auto-cleaned after success).
*   `output/`: Final dubbed videos.

## ⚠️ Troubleshooting

*   **Ollama Error**: Ensure Ollama is running (`ollama serve`). If you don't have Ollama, the system will fallback to Google Translate automatically.
*   **Audio/Video Sync**: If lipsync is off, check if the source video has variable frame rate (VFR). Convert to CFR if needed.
*   **GPU Not Used**: Ensure you have installed CUDA toolkit and your PyTorch version matches.

---
*Created by Antigravity*
