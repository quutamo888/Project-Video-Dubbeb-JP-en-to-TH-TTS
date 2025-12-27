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

You can choose between **uv** (Recommended) or standard **pip**.

### Option 1: Using `uv` (Fast & Recommended)
This method automatically manages the virtual environment.

1.  **Sync Dependencies**:
    ```powershell
    uv sync
    ```
    *(Creates `.venv` automatically)*

2.  **Run**:
    ```powershell
    .\.venv\Scripts\python gui.py
    ```

### Option 2: Using standard `pip`
For those who prefer traditional setup.

1.  **Create Virtual Environment** (Optional but recommended):
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\activate
    ```

2.  **Install Dependencies**:
    ```powershell
    pip install -r requirements.txt
    ```

3.  **Run**:
    ```powershell
    python gui.py
    ```
    *(Or `.\.venv\Scripts\python gui.py` if using venv)*

## ▶️ How to Use

### 1. GUI Mode (Easiest)
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

## 🙏 Acknowledgements

Special thanks to these amazing open-source projects that make this tool possible:

*   **[F5-TTS-THAI](https://github.com/VYNCX/F5-TTS-THAI)**: The core engine for high-quality Thai voice cloning.
*   **[Faster Whisper](https://github.com/SYSTRAN/faster-whisper)**: For lightning-fast and accurate speech transcription.
*   **[Ollama](https://ollama.com/)**: Enabling local LLM inference for uncensored translation.
*   **Gemma 2 / Llama 3 Models**: Powering the contextual understanding and translation capabilities.
*   **Demucs**: For state-of-the-art music source separation.

---
*Created by Antigravity*
