# AI Video Dubbing Pro (JP/EN -> Thai) 🎥🇹🇭

A comprehensive AI-powered video dubbing tool designed for high-quality, **uncensored** Thai localization. It features voice cloning, bridge translation for Japanese content, and a modern GUI with real-time system monitoring.

## 📺 Demo & UI Preview

### GUI Interface
![AI Video Dubbing Pro GUI](Review/UI.png)

### Video Demo
[![Watch the Video Demo](https://img.youtube.com/vi/xdQI-p60SMg/0.jpg)](https://youtu.be/xdQI-p60SMg)


## 🚀 Key Features

*   **Advanced TTS**: Uses **OmniVoice** (State-of-the-Art) as default for natural Thai speech and **Voice Cloning** to match original speaker's tone, with Edge-TTS and MMS as options.
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

## 🆕 Recent Updates & Bug Fixes / การปรับปรุงแก้ไขล่าสุด

*   **👤 Consistent Speaker Voice Profile (Reuse Voice Prompt) / ล็อคเสียงตัวละครเดิมตลอดทั้งคลิป**:
    *   เพิ่มตัวเลือก GUI `Consistent Speaker Voice (Reuse Voice Prompt)`
    *   ระบบจะทำการสร้าง `VoiceClonePrompt` สำหรับเสียงผู้หญิง และเสียงผู้ชาย เพียงครั้งเดียวจากท่อนเสียงที่ชัดเจนที่สุด
    *   นำ Prompt เสียงเดิมมาใช้ซ้ำทุกประโยค ช่วยให้ตัวละครชาย/หญิง มีโทนเสียงเดียวกันตลอดทั้งวิดีโอ (100% Voice Consistency) ไม่เปลี่ยนไปมาทุกประโยค

*   **🎤 OmniVoice Voice Cloning Modes (3 รูปแบบการเจนเสียง)**:
    *   เพิ่มตัวเลือก GUI `OmniVoice Clone Mode`:
        *   **Full Clone (Voice + Accent)**: เลียนแบบทั้งน้ำเสียงตัวละคร (Timbre) และสำนวน/จังหวะการพูดเดิม (Accent/Prosody)
        *   **Timbre Only (Voice Only, No Accent)**: เลียนแบบเฉพาะโทนเสียงชาย/หญิง (Timbre) พูดภาษาไทยด้วยจังหวะธรรมชาติ ไร้สำนวนต่างชาติ (ทำงานเร็วขึ้น 100 เท่า ป้องกันการค้าง)
        *   **Disabled (Standard TTS)**: ใช้เสียงสังเคราะห์มาตรฐานพร้อมระบบปรับ Pitch ชาย/หญิง

*   **⚡ Multitask & Parallel Processing (ประมวลผลขนาน Step 5 & 6)**:
    *   เพิ่มตัวเลือก GUI **Enable Multitask** และกำหนดจำนวน **Max Worker Tasks** (1-16)
    *   แปลภาษา (`modules/translator.py`) และสร้างเสียง Edge-TTS ขนานกันผ่าน ThreadPoolExecutor & Async Semaphores
    *   ปรับปรุงระบบ Thread-lock สำหรับ OmniVoice CUDA GPU ป้องกัน GPU ค้าง (GPU 100% Freeze Fix)

*   **🌐 Flexible Language Selection Dropdowns / เลือกระบุภาษาได้อย่างยืดหยุ่น**:
    *   **Source Audio Language**: เลือก `Auto Detect`, `Japanese (ja)`, `English (en)`, หรือ `Thai (th)`
    *   **Output Subtitle Language**: เลือกภาษาซับไตเติลเป้าหมาย (`Thai`, `English`)
    *   **OmniVoice / TTS Target Voice**: เลือกภาษาของเสียงพากย์ TTS (`Thai`, `English`)

*   **📝 Real-time Sentence-by-Sentence Console Logs / แสดง Log ละเอียดแบบประโยคต่อประโยค**:
    *   **Step 3/6 (Transcribe)**: แสดงเวลาและประโยคภาษาต้นทางที่ถอดได้ในขณะนั้น
    *   **Step 5/6 (Translate)**: แสดงประโยคต้นทาง และผลลัพธ์ประโยคแปลภาษาไทย/อังกฤษแบบเรียลไทม์
    *   **Step 6/6 (TTS Generation)**: แสดงประโยคและเพศตัวละครที่กำลังสังเคราะห์เสียงพากย์

*   **🔧 Windows & PyTorch Compatibility Fixes / แก้ไข Bug ระบบบน Windows**:
    *   แก้ไขข้อผิดพลาด `Could not load libtorchcodec` บน Windows PyTorch nightly โดยใช้ `soundfile`/`librosa` แทน
    *   แก้ไข `NameError: OMNIVOICE_AVAILABLE`
    *   ลบโมเดล F5-TTS ที่ไม่ได้ใช้งานออกจากระบบและคู่มือ

## 🛠️ Prerequisites

*   **Python 3.10+**
*   **FFmpeg**: Must be installed and added to system PATH.
*   **NVIDIA GPU** (Recommended): For fast rendering and Whisper/OmniVoice inference.
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
    *   `voice_generator.py`: OmniVoice, Edge-TTS, and MMS voice synthesis logic.
    *   `vocal_isolator.py`: Demucs background separation.
    *   `video_merger.py`: MoviePy/FFmpeg video assembly.
*   `temp/`: Temporary processing artifacts (Auto-cleaned after success).
*   `output/`: Final dubbed videos.

## 🙏 Acknowledgements

Special thanks to these amazing open-source projects that make this tool possible:

*   **[OmniVoice](https://github.com/k2-fsa/OmniVoice)**: The default core engine for high-quality Thai voice cloning and TTS.
*   **[Faster Whisper](https://github.com/SYSTRAN/faster-whisper)**: For lightning-fast and accurate speech transcription.
*   **[Ollama](https://ollama.com/)**: Enabling local LLM inference for uncensored translation.
*   **Gemma 2 / Llama 3 Models**: Powering the contextual understanding and translation capabilities.
*   **Demucs**: For state-of-the-art music source separation.

