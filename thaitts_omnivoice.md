# การใช้งาน OmniVoice (Text-to-Voice & Voice Cloning)

โปรเจกต์นี้ใช้ไลบรารี `omnivoice` ร่วมกับโมเดล `k2-fsa/OmniVoice` ในการสังเคราะห์เสียงและการโคลนเสียง

---

## 1. การเตรียมความพร้อม (Setup)

ติดตั้งไลบรารีที่จำเป็น (ระบุใน `backend/requirements.txt` แล้ว):
```bash
pip install omnivoice torch soundfile
```

---

## 2. ตัวอย่างการเขียนโค้ด (Usage Examples)

### การโหลดโมเดล
```python
import torch
from omnivoice import OmniVoice

# ตรวจสอบการใช้งาน GPU/CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# โหลดโมเดลจาก HuggingFace
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=dtype)
```

### การแปลงข้อความเป็นเสียงทั่วไป (Normal TTS)
```python
audio_list = model.generate(
    text="สวัสดีครับ ยินดีต้อนรับสู่ AI Audio Studio",
    speed=1.0,       # ความเร็วเสียง (ค่าเริ่มต้น 1.0)
    num_step=32      # ขั้นตอนการทำงานของโมเดล (ค่าเริ่มต้น 32)
)

# บันทึกเป็นไฟล์เสียง (OmniVoice มี Sample Rate 24000 Hz)
import soundfile as sf
import numpy as np

final_audio = np.asarray(audio_list[0])
sf.write("output.wav", final_audio, 24000)
```

### การโคลนเสียง (Voice Cloning)
ต้องใช้ไฟล์เสียงต้นแบบ (Reference Audio) ความยาวที่เหมาะสมคือ 3-10 วินาที
```python
audio_list = model.generate(
    text="ข้อความใหม่ที่ต้องการให้พูดด้วยเสียงโคลน",
    ref_audio="path/to/reference_voice.wav", # ไฟล์เสียงต้นแบบ
    ref_text="ข้อความที่พูดอยู่ในไฟล์เสียงต้นแบบ",    # (ใส่เพื่อช่วยให้เสียงแม่นยำขึ้น)
    speed=1.0,
    num_step=32
)
```

### การสร้างน้ำเสียงตามคำสั่ง (Instruct Mode)
ปรับอารมณ์หรือลักษณะการพูดตามคำสั่งตัวอักษร
```python
audio_list = model.generate(
    text="วันนี้ผมมีความสุขมากเลยครับ",
    instruct="พูดด้วยน้ำเสียงดีใจ ตื่นเต้น", # คำสั่งบอกน้ำเสียง/ความรู้สึก
    speed=1.0,
    num_step=32
)
```

---

## 3. โครงสร้างการแบ่งข้อความยาว (Chunking)

เนื่องจากโมเดลอาจจะทำเสียงเพี้ยนเมื่อป้อนข้อความที่ยาวเกินไป ในโปรเจกต์นี้จึงใช้เทคนิคการซอยคำย่อย (Chunking) ที่ดีต่อภาษาไทย:
1. **ตัดข้อความ**: แบ่งประโยคไม่เกิน 220 ตัวอักษร
2. **จุดตัดธรรมชาติ**: พยายามตัดบริเวณเครื่องหมายจบประโยค ช่องว่าง หรือสัญลักษณ์ (`. ! ? 、 。 ฯ`)
3. **การต่อเสียง**:
   - นำเสียงแต่ละ Chunk มาต่อกัน
   - ใส่เสียงเงียบ (Silence) ประมาณ `0.25` วินาทีคั่นกลางระหว่าง Chunk เพื่อให้เกิดจังหวะหยุดหายใจที่เป็นธรรมชาติ
