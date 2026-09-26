# CONTEXT & ADR: Local JP->TH Translation & OmniVoice Thai Optimization

## 1. Glossary & Domain Definitions
- **CTranslate2 NLLB Engine**: ระบบแปลภาษา C++ CUDA แบบออฟไลน์ 100% ไม่ต้องต่อเน็ต ไม่ต้องรัน Ollama รันบน Tensor Cores ของ RTX 5070 โดยตรง
- **Direct Route (jpn_Jpan -> tha_Thai)**: การแปลตรงระหว่างภาษาญี่ปุ่นสู่ภาษาไทยโดยไม่ผ่านภาษาอังกฤษเป็นตัวกลาง ลดความคลาดเคลื่อนและลดเวลาประมวลผลครึ่งหนึ่ง
- **Standalone Subtitle Translator**: หน้าต่าง/แท็บเครื่องมือใน GUI สำหรับรับไฟล์ `.srt` ภาษาญี่ปุ่น แปลงเป็น `.th.srt` โดยรักษา Timecode เดิมไว้ครบถ้วน
- **Thai Prosody Enforcement**: เทคนิคการสังเคราะห์เสียง OmniVoice ที่เลียนแบบเฉพาะเนื้อเสียง (Timbre) และตัดจังหวะ/สำเนียงภาษาญี่ปุ่นออก เพื่อให้พูดภาษาไทยด้วยสำเนียงไทยแท้
- **Gated HuggingFace Model**: โมเดลที่ต้องกดยอมรับเงื่อนไขบน HuggingFace และใช้ `HF_TOKEN` ในการดาวน์โหลด (เช่น `Ex0TiiC/OmniVoice-TH-NonverbalTTS-v1`)
- **SenseVoice (Alibaba FunASR)**: โมเดลถอดเสียงความเร็วสูงพิเศษ (10-15x เร็วกว่า Whisper) รองรับ JA, EN, ZH, KO และตรวจจับอารมณ์/Audio Events อัตโนมัติ
- **Emotion-Conditioned TTS**: การส่งแท็กอารมณ์ (Happy, Sad, Angry, Neutral) ที่ได้จาก SenseVoice ไปควบคุมการสังเคราะห์เสียงของ OmniVoice ให้ตรงกับอารมณ์ต้นฉบับ

---

## 2. Architecture Decision Records (ADRs)

### ADR-001: Local JP->TH Translation Engine via CTranslate2
- **Context**: ผู้ใช้ต้องการแปลภาษาในโปรแกรมโดยตรง ไม่พึ่ง Ollama หรือ API ภายนอก และใช้ GPU เต็มประสิทธิภาพ
- **Decision**: ใช้ **CTranslate2** ร่วมกับโมเดล **NLLB-200-distilled-600M** หรือ **1.3B** (FP16 บน CUDA)
- **Status**: Accepted
- **Consequences**:
  - ประมวลผลเร็วขึ้น 5-10 เท่าเมื่อเทียบกับ HuggingFace PyTorch มาตรฐาน
  - ใช้ VRAM ต่ำ (~1-2 GB) ปลอดภัยไม่ชนกับ VRAM ของ OmniVoice
  - ไม่เกิดภาพหลอน (Zero Hallucination) โครงสร้างประโยคคงที่

### ADR-002: Standalone Subtitle Translation GUI & Pipeline Integration
- **Context**: ผู้ใช้ต้องการแปลซับไตเติลภาษาญี่ปุ่นเป็นไทยจากโปรแกรมโดยตรง
- **Decision**:
  1. เพิ่มแท็บ/เมนู **"แปลซับไตเติล (.srt)"** ใน [gui.py](file:///f:/project/Project-Video-Dubbeb-JP-en-to-TH-TTS/gui.py) ให้เลือกไฟล์ `.srt`, แปลด้วย CTranslate2 และบันทึกเป็น `.th.srt`
  2. เพิ่มตัวเลือก `local-ctranslate2` ใน [config.py](file:///f:/project/Project-Video-Dubbeb-JP-en-to-TH-TTS/config.py) เพื่อให้ Pipeline หลัก (Step 5) สามารถสลับมาใช้ CTranslate2 ได้ทันที
- **Status**: Accepted

### ADR-003: OmniVoice Model Update & Thai Accent Enforcement
- **Context**: ผู้ใช้ต้องการอัปเดตโมเดลเป็นโมเดลเสียงไทยเฉพาะ (`Ex0TiiC/OmniVoice-TH-NonverbalTTS-v1`), เร่งความเร็ว Engine, และล็อกสำเนียงไทยแท้
- **Decision**:
  1. เพิ่มค่าคอนฟิก `OMNIVOICE_MODEL` และ `HUGGINGFACE_TOKEN` ใน [config.py](file:///f:/project/Project-Video-Dubbeb-JP-en-to-TH-TTS/config.py) (รองรับทั้ง `Ex0TiiC`, `wannaphong`, `k2-fsa`)
  2. การเร่งความเร็วบน Windows: ใช้ `torch.inference_mode()` + `torch.float16` พร้อม PyTorch SDPA (Scaled Dot-Product Attention) เนื่องจาก `torch.compile` บน Windows ขาด Triton compiler
  3. ล็อกสำเนียงไทย: ปรับแต่งให้ OmniVoice ทำงานในโหมด `timbre` และส่ง `language='th'` (หรือ `norm_lang`) ให้โมเดลโดยตรง โดยค่า `instruct` ต้องใช้ token ที่อยู่ใน whitelist (`female` / `male`) ตามข้อกำหนดของ OmniVoice `_resolve_instruct` ห้ามใส่คำบรรยายภาษาไทยใน instruct
- **Status**: Accepted

### ADR-004: SenseVoice Integration (Superseded by ADR-005)
- **Status**: Deprecated / Superseded

### ADR-005: Kotoba-Whisper Integration for SOTA Japanese Subtitle Transcription
- **Context**: ผู้ใช้ต้องการถอดเสียงภาษาญี่ปุ่นที่มีความแม่นยำสูงสุดในระดับ SOTA โดยไม่เกิด Hallucination รวมประโยคยาวเหมือน SenseVoice และให้มี timestamp ตรงตามช่วงเวลาพูดจริงในวิดีโอ
- **Decision**:
  1. ใช้ **Kotoba-Whisper** (`kotoba-tech/kotoba-whisper-v2.2-faster`) ผ่านเอนจิน `faster-whisper` CTranslate2
  2. ตั้งค่า `STT_ENGINE = "kotoba-whisper"` เป็นค่าเริ่มต้นใน [config.py](file:///f:/project/Project-Video-Dubbeb-JP-en-to-TH-TTS/config.py)
  3. เพิ่มตัวเลือกใน [gui.py](file:///f:/project/Project-Video-Dubbeb-JP-en-to-TH-TTS/gui.py): `Kotoba-Whisper (Japanese SOTA, Ultra Accurate)` และ `faster-whisper (Large-v2, Universal)`
  4. ทำงานร่วมกับ CTranslate2 บน GPU (`float16`) หรือ CPU (`int8`) พร้อม VAD filtering (`min_silence_duration_ms=500`) เพื่อแบ่งประโยคย่อยอย่างเป็นธรรมชาติ
  5. หากเลือกภาษาต้นทางที่ไม่ใช่ภาษาญี่ปุ่น (เช่น อังกฤษ, จีน, เกาหลี, ไทย) ระบบจะสลับไปใช้ `faster-whisper (large-v2)` อัตโนมัติ
- **Status**: Accepted

### ADR-006: Context-Aware Offline Subtitle Translation with Qwen2.5-3B
- **Context**: การแปลซับไตเติลภาษาญี่ปุ่นเป็นไทยแบบรายบรรทัด (Sentence-level MT เช่น NLLB) ขาดบริบทของบทสนทนาก่อนหน้า ทำให้สรรพนาม (ฉัน/ผม/เธอ) ผิดเพี้ยน และระดับความสุภาพไม่ต่อเนื่อง
- **Decision**:
  1. เพิ่มตัวเลือกแปลด้วย LLM ออฟไลน์ผ่าน GPU CUDA: **`Qwen/Qwen2.5-3B-Instruct`** (หรือรุ่นที่ระบุใน `Config.QWEN_MODEL`)
  2. ใช้เทคนิค **Sliding Window Context Prompting**: ส่งประวัติบทสนทนา 2-3 บรรทัดก่อนหน้าเข้าไปใน System Prompt เพื่อให้ LLM รับรู้เพศผู้พูด สรรพนาม และอารมณ์ของฉาก
  3. เพิ่มตัวเลือก Engine ในหน้าต่าง **Standalone Subtitle Translator (`SubtitleTranslatorWindow`)**:
     - `Qwen2.5-3B (Context-Aware LLM, Best Thai Quality, Offline GPU)`
     - `CTranslate2 NLLB-200 (Fastest GPU, Offline)`
  4. เพิ่มตัวเลือก `local-qwen (Context-Aware LLM, High Quality GPU)` ในแท็บหลักสำหรับกระบวนการ Dubbing Step 5
  5. คืน VRAM อัตโนมัติ (`unload_qwen_model()`) หลังเสร็จสิ้นการแปล
- **Status**: Accepted

### ADR-007: Dual Audio Streams with Stream-Copy FFmpeg Muxing
- **Context**: ผู้ใช้ต้องการเก็บเสียงต้นฉบับไว้ในไฟล์วิดีโอผลลัพธ์ ไม่ให้ถูกแทนที่หรือลบทิ้ง แต่ให้เพิ่มเสียงพากย์ไทย (OmniVoice + Background) เป็นแทร็กเสียงที่ 2 (Dual Audio Tracks)
- **Decision**:
  1. ใช้ **Direct FFmpeg Stream Copy (`-c:v copy`)** แทนการ Re-encode วิดีโอผ่าน MoviePy ลดเวลาเรนเดอร์จากหลายนาทีเหลือ 2-3 วินาที และรักษาคุณภาพภาพ 100%
  2. จัดโครงสร้าง Multi-Audio Streams:
     - **Track 1**: เสียงต้นฉบับ (`-metadata:s:a:0 title="Original Audio"` `-disposition:a:0 none`)
     - **Track 2**: เสียงพากย์ไทย (`-metadata:s:a:1 title="Thai Dubbed (OmniVoice)"` `-disposition:a:1 default`)
     - กำหนดให้ Track 2 เป็น Default Audio Track เมื่อเปิดเล่นวิดีโอ
  3. เพิ่มตัวเลือก Checkbox `Dual Audio (Keep Original + Add Dubbed)` ใน GUI และคอนฟิก `Config.DUAL_AUDIO_TRACKS` (เปิดใช้งานเป็นค่าเริ่มต้น)
  4. หากไฟล์วิดีโอต้นทางไม่มีแทร็กเสียง หรือเกิดปัญหา Container stream copy ระบบจะปรับเป็น Single-track หรือ Fallback ไปยัง MoviePy อัตโนมัติ
- **Status**: Accepted

### ADR-008: Integrated YouTube Video Downloader with Quality Selection
- **Context**: ผู้ใช้ต้องการดูดวิดีโอจาก YouTube URL โดยตรง พร้อมระบุความละเอียดที่ต้องการ และส่งไฟล์เข้า Input Pipeline ของระบบพากย์ทันที
- **Decision**:
  1. เพิ่มโมดูล `modules/youtube_downloader.py` ทำงานผ่าน `yt-dlp` ดึงสตรีมภาพและเสียงคุณภาพสูงสุดตามความละเอียดที่เลือก (1080p, 720p, 480p, 360p, Best) และรวมเป็น MP4 ผ่าน FFmpeg
  2. เพิ่มหน้าต่าง `YouTubeDownloaderWindow` ใน [gui.py](gui.py) พร้อมปุ่มเปิด 2 จุด:
     - ปุ่ม **`⬇️ YouTube`** ข้างช่อง Input Video
     - ปุ่ม **`⬇️ YouTube DL`** ในแผง Utilities
  3. ฟังก์ชัน Auto-populate: เมื่อดาวน์โหลดเสร็จ จะกรอกชื่อไฟล์ลงใน `Input Video` และตั้งชื่อไฟล์ `Output Video` ในหน้าหลักให้อัตโนมัติ
- **Status**: Accepted

### ADR-009: Dual-Mode Advanced Gender Detection (Robust ML & Speaker Clustering with Majority Voting)
- **Context**: การตรวจจับเพศเสียงเดิมใช้โมเดล `alefiury/wav2vec2...librispeech` (เทรนจาก Audiobooks ภาษาอังกฤษ) มักประเมินเสียงญี่ปุ่น/อนิเมะ/โทนเสียงสูงผิดพลาด และการประเมินทีละประโยคแบบอิสระทำให้ตัวละครเดิมสลับเพศไปมา (Gender Flipping) ส่งผลให้เสียงพากย์ TTS และสรรพนามภาษาไทยเปลี่ยนกลับไปกลับมา
- **Decision**:
  1. **Option 1: Robust SOTA ML Model (`audeering/wav2vec2-large-robust-12-ft-age-gender`)**:
     - รองรับการประเมินเสียงหลากหลายภาษา ทนทานต่อ Noise และอารมณ์/การตะโกนของตัวละคร
     - ทำ fallback ไปยัง `ml-librispeech` หรือ `pitch (YIN)` หากดาวน์โหลดไม่ได้
  2. **Option 2: Speaker Clustering & Duration-Weighted Majority Voting with Linguistic Hints**:
     - สกัด Acoustic Timbre Embedding (MFCCs, spectral contrast, centroid, F0 statistics) แต่ละประโยค
     - รวมกลุ่มประโยคด้วย Cosine Distance คลัสเตอร์เป็น Speaker IDs (`SPEAKER_00`, `SPEAKER_01`, ...)
     - ตรวจจับเบาะแสทางภาษา (Linguistic Pronoun Hints): สรรพนามภาษาญี่ปุ่นบ่งบอกเพศชัดเจน (ชาย: `僕`, `俺`, `ぜ`, `ぞ` / หญิง: `あたし`, `かしら`, `わよ`)
     - ทำ Majority Voting ร่วมกันระหว่างความยาวเสียง (Duration Weight) + ผลลัพธ์ ML Audio + เบาะแสทางภาษา
     - ล็อกเพศเดียวให้กับทุกประโยคของ Speaker นั้นทั้งคลิป (Zero Gender Flipping)
  3. เพิ่มตัวเลือกใน GUI:
     - Dropdown `Audio Model`: `ml-robust (audeering, Multi-lingual AI)`, `ml-librispeech (wav2vec2, Legacy)`, `pitch (Hz Frequency, Fast)`
     - Checkbox: `Lock Gender per Speaker (Clustering)` (Default: Enabled)
- **Status**: Accepted




