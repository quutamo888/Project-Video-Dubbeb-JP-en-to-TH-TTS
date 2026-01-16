# 🎯 New Features Guide

## 1. Local Translation (NLLB) - แปลในเครื่องโดยไม่ต้องใช้ API

### ข้อดี:
- ✅ รันใน GPU ของคุณ ไม่ต้องเชื่อมต่อ Ollama
- ✅ รวดเร็ว แม่นยำ สำหรับ JP → TH
- ✅ ไม่ต้องติดตั้ง LLM ขนาดใหญ่ (ใช้ model แค่ ~600MB)
- ✅ ฟรี ไม่มีค่าใช้จ่าย API

### วิธีใช้:

#### Option 1: แก้ไขใน `config.py`
```python
TRANSLATION_PROVIDER = "local-transformer"
```

#### Option 2: ตั้งค่า Environment Variable
```bash
set TRANSLATION_PROVIDER=local-transformer
```

#### Option 3: แก้ใน `.env` file
```
TRANSLATION_PROVIDER=local-transformer
```

### Model ที่ใช้:
- **Default**: `facebook/nllb-200-distilled-600M` (~600MB)
- รองรับภาษา: Japanese, Thai, English, Chinese, Korean และอีก 195+ ภาษา

### การทำงาน:
1. โหลด NLLB model (download ครั้งแรก)
2. แปลข้อความทีละ segment
3. Auto unload model หลังแปลเสร็จ → คืน VRAM ให้ TTS

---

## 2. Edge TTS - เสียงพากย์ไทยคุณภาพสูงจาก Microsoft

### ข้อดี:
- ✅ เสียงไทยธรรมชาติ สมจริง (Microsoft Azure quality)
- ✅ ไม่ต้อง download model
- ✅ ฟรี ไม่มีค่าใช้จ่าย
- ✅ รองรับเสียงชายและหญิง
- ✅ รวดเร็ว (cloud-based)

### วิธีใช้:

#### Option 1: แก้ไขใน `config.py`
```python
TTS_PROVIDER = "edge-tts"
```

#### Option 2: ตั้งค่า Environment Variable
```bash
set TTS_PROVIDER=edge-tts
```

#### Option 3: แก้ใน `.env` file
```
TTS_PROVIDER=edge-tts
```

### เสียงที่ใช้:
- **ชาย**: `th-TH-NiwatNeural` (นิวัฒน์)
- **หญิง**: `th-TH-PremwadeeNeural` (เปรมวดี) - default

### การทำงาน:
1. ส่งข้อความไปยัง Microsoft Azure TTS
2. รับไฟล์เสียง MP3
3. แปลงเป็น WAV พร้อม apply pitch adjustment
4. รวมเข้ากับวิดีโอ

---

## 📊 เปรียบเทียบ Options

| Feature | F5-TTS | Edge-TTS | MMS-TTS |
|---------|--------|----------|---------|
| คุณภาพเสียง | 🌟🌟🌟🌟🌟 (Voice Clone) | 🌟🌟🌟🌟 (Natural) | 🌟🌟🌟 (Robotic) |
| ความเร็ว | 🐌 ช้า | 🚀 เร็วมาก | 🏃 ปานกลาง |
| VRAM | ~8GB | 0 (Cloud) | ~2GB |
| Internet | ไม่ต้อง | ต้องการ | ไม่ต้อง |
| ภาษาไทย | ✅ ดีมาก | ✅ ดีมาก | ⚠️ พอใช้ |

| Translation | Ollama | Local-Transformer | Google |
|-------------|--------|-------------------|--------|
| คุณภาพ JP→TH | 🌟🌟🌟🌟 | 🌟🌟🌟🌟 | 🌟🌟🌟 |
| ความเร็ว | 🐌 ช้า | 🏃 ปานกลาง | 🚀 เร็ว |
| VRAM | ~4GB+ | ~2GB | 0 (Cloud) |
| Internet | ไม่ต้อง | ไม่ต้อง | ต้องการ |
| Setup | Ollama server | แค่ติดตั้ง lib | ไม่ต้อง |

---

## 🎮 คำแนะนำการใช้งาน

### สำหรับ GPU ไม่แรง (< 8GB VRAM):
```
TRANSLATION_PROVIDER=local-transformer  # ใช้ NLLB แทน Ollama
TTS_PROVIDER=edge-tts                   # ใช้ Edge แทน F5-TTS
```

### สำหรับ GPU แรง (>= 12GB VRAM):
```
TRANSLATION_PROVIDER=ollama             # ใช้ Ollama (ยืดหยุ่นกว่า)
TTS_PROVIDER=f5-tts                     # Voice cloning คุณภาพสูง
```

### สำหรับไม่มี GPU:
```
TRANSLATION_PROVIDER=google             # ใช้ Google Translate
TTS_PROVIDER=edge-tts                   # ใช้ Edge TTS
```

### สำหรับรวดเร็วที่สุด:
```
TRANSLATION_PROVIDER=google
TTS_PROVIDER=edge-tts
WHISPER_MODEL_SIZE=medium               # ใช้ model เล็กลง
```

---

## 🚀 ตัวอย่างการใช้งาน

### ผ่าน GUI:
1. เปิด `run.bat`
2. เลือกไฟล์ video
3. ระบบจะใช้ config ที่ตั้งไว้

### ผ่าน Command Line:
```bash
# แปล JP → TH ด้วย NLLB + Edge TTS
set TRANSLATION_PROVIDER=local-transformer
set TTS_PROVIDER=edge-tts
uv run main.py -i input.mp4 -o output.mp4
```

---

## 🔧 Troubleshooting

### NLLB โหลดช้า
- รอให้ download model ครั้งแรก (~600MB)
- Model จะถูก cache ไว้ ครั้งต่อไปเร็วขึ้น

### Edge TTS error "No internet"
- ตรวจสอบการเชื่อมต่ออินเทอร์เน็ต
- ลอง fallback เป็น `f5-tts` หรือ `mms`

### Out of VRAM
- ใช้ `local-transformer` แทน `ollama`
- ใช้ `edge-tts` แทน `f5-tts`
- ลด `WHISPER_MODEL_SIZE` เป็น `medium` หรือ `small`

---

สร้างโดย AI Video Dubbing Pro Pipeline 🎬
