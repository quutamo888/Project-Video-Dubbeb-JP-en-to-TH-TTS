import logging
import json
import requests
import re
import os
import threading
import gc
from deep_translator import GoogleTranslator
from config import Config

logger = logging.getLogger(__name__)

# NLLB Language code mapping
NLLB_LANG_MAP = {
    "ja": "jpn_Jpan",
    "jpn": "jpn_Jpan",
    "jpn_Jpan": "jpn_Jpan",
    "th": "tha_Thai",
    "tha": "tha_Thai",
    "tha_Thai": "tha_Thai",
    "en": "eng_Latn",
    "eng": "eng_Latn",
    "eng_Latn": "eng_Latn",
    "zh": "zho_Hans",
    "zho_Hans": "zho_Hans",
    "ko": "kor_Hang",
    "kor_Hang": "kor_Hang",
    "auto": "jpn_Jpan"
}

# CTranslate2 cache
_ct2_translator = None
_ct2_tokenizer = None

def get_ct2_translator(source_lang="jpn_Jpan"):
    """
    Initializes and caches CTranslate2 Translator and Tokenizer.
    Runs on NVIDIA GPU with float16 Tensor Cores.
    """
    global _ct2_translator, _ct2_tokenizer
    if _ct2_translator is None or _ct2_tokenizer is None:
        import ctranslate2
        from transformers import AutoTokenizer
        from huggingface_hub import snapshot_download
        import torch

        model_ref = Config.CT2_NLLB_MODEL
        logger.info(f"Loading CTranslate2 NLLB model: {model_ref}...")
        
        if os.path.isdir(model_ref):
            model_dir = model_ref
        else:
            token = Config.HF_TOKEN if Config.HF_TOKEN else None
            model_dir = snapshot_download(model_ref, token=token)
            
        device = "cuda" if torch.cuda.is_available() and Config.USE_GPU else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        _ct2_translator = ctranslate2.Translator(model_dir, device=device, compute_type=compute_type)
        token_arg = Config.HF_TOKEN if Config.HF_TOKEN else None
        _ct2_tokenizer = AutoTokenizer.from_pretrained(Config.NLLB_MODEL, src_lang=source_lang, token=token_arg)
        logger.info(f"✅ CTranslate2 NLLB loaded on {device} ({compute_type})")
        
    return _ct2_translator, _ct2_tokenizer

def translate_with_ctranslate2(text_list, source_lang="jpn_Jpan", target_lang="tha_Thai", batch_size=32):
    """
    Translates text or list of texts using CTranslate2 NLLB model on GPU.
    """
    if not text_list:
        return "" if isinstance(text_list, str) else []
        
    is_single = isinstance(text_list, str)
    texts = [text_list] if is_single else text_list
    
    src_code = NLLB_LANG_MAP.get(source_lang, source_lang)
    tgt_code = NLLB_LANG_MAP.get(target_lang, target_lang)
    
    try:
        translator, tokenizer = get_ct2_translator(source_lang=src_code)
        
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            tokens_batch = [
                tokenizer.convert_ids_to_tokens(tokenizer.encode(t, max_length=512, truncation=True))
                for t in batch
            ]
            target_prefixes = [[tgt_code]] * len(tokens_batch)
            
            translated_batch = translator.translate_batch(
                tokens_batch,
                target_prefix=target_prefixes,
                beam_size=4,
                max_decoding_length=512,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
            
            for trans in translated_batch:
                out_tokens = trans.hypotheses[0]
                if out_tokens and out_tokens[0] == tgt_code:
                    out_tokens = out_tokens[1:]
                out_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(out_tokens))
                results.append(out_text.strip())
                
        return results[0] if is_single else results
    except Exception as e:
        logger.error(f"CTranslate2 translation failed: {e}")
        import traceback
        traceback.print_exc()
        return None if is_single else [None] * len(texts)

def unload_ct2_translator():
    """Unloads CTranslate2 model and releases GPU VRAM."""
    global _ct2_translator, _ct2_tokenizer
    if _ct2_translator is not None:
        del _ct2_translator
        del _ct2_tokenizer
        _ct2_translator = None
        _ct2_tokenizer = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("✅ CTranslate2 translator unloaded from VRAM.")

# Qwen LLM cache
_qwen_model = None
_qwen_tokenizer = None
_qwen_lock = threading.Lock()

def get_qwen_model_and_tokenizer(model_name=None):
    """
    Initializes and caches Qwen2.5 model and tokenizer on CUDA/CPU.
    Thread-safe and manages GPU memory strictly to avoid OOM.
    """
    global _qwen_model, _qwen_tokenizer
    with _qwen_lock:
        if _qwen_model is None or _qwen_tokenizer is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            # Clear cache and garbage before allocating ~6GB for Qwen 3B
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model_ref = model_name or getattr(Config, "QWEN_MODEL", "Qwen/Qwen2.5-3B-Instruct")
            logger.info(f"Loading Qwen LLM: {model_ref}...")

            token_arg = Config.HF_TOKEN if Config.HF_TOKEN else None
            _qwen_tokenizer = AutoTokenizer.from_pretrained(model_ref, token=token_arg)

            device = "cuda" if torch.cuda.is_available() and Config.USE_GPU else "cpu"
            dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device == "cuda" else torch.float32)

            _qwen_model = AutoModelForCausalLM.from_pretrained(
                model_ref,
                torch_dtype=dtype,
                device_map=device if device == "cuda" else None,
                token=token_arg,
                low_cpu_mem_usage=True
            )
            if device == "cpu":
                _qwen_model = _qwen_model.to("cpu")
            _qwen_model.eval()
            logger.info(f"✅ Qwen LLM loaded on {device} ({dtype})")

    return _qwen_model, _qwen_tokenizer

def unload_qwen_model():
    """Unloads Qwen model from VRAM and empties CUDA cache."""
    global _qwen_model, _qwen_tokenizer
    with _qwen_lock:
        if _qwen_model is not None:
            del _qwen_model
            del _qwen_tokenizer
            _qwen_model = None
            _qwen_tokenizer = None
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("✅ Qwen LLM unloaded from VRAM.")

def translate_with_qwen(target_text, context_history=None, source_lang="ja", target_lang="th", gender="female"):
    """
    Translates a subtitle line using Qwen2.5-Instruct with sliding window dialogue context.
    """
    if not target_text or not str(target_text).strip():
        return ""

    import torch
    model, tokenizer = get_qwen_model_and_tokenizer()

    tgt_lang_name = "Thai" if target_lang in ["th", "tha", "tha_Thai"] else "English"
    src_lang_name = "Japanese" if source_lang in ["ja", "jpn", "jpn_Jpan"] else "foreign"

    context_lines = []
    if context_history:
        for ch in context_history[-3:]:
            s = ch.get("source", "")
            t = ch.get("target", "")
            if s and t:
                context_lines.append(f"Original: {s} -> {tgt_lang_name}: {t}")

    context_str = "\n".join(context_lines) if context_lines else "None (start of scene)"

    g = (gender or "female").lower()
    gender_note = (
        "- Speaker is MALE: use 'ผม' for 1st person, 'ครับ' for ending particles."
        if g == "male"
        else "- Speaker is FEMALE: use 'ฉัน' or 'หนู' for 1st person, 'ค่ะ/คะ' for ending particles."
    )

    system_prompt = (
        f"You are a professional film and video subtitle translator specializing in {src_lang_name} to {tgt_lang_name} localization.\n"
        f"Translate the target dialogue line into natural spoken {tgt_lang_name} suitable for video subtitles.\n"
        "STRICT RULES:\n"
        "1. Output ONLY the translated text for the target line. Do NOT output notes, quotes, explanations, or line numbers.\n"
        "2. Keep the dialogue tone, honorifics, and character relationships consistent with the preceding dialogue context.\n"
        "3. Keep subtitle length concise and natural for reading.\n"
        f"{gender_note if tgt_lang_name == 'Thai' else ''}"
    )

    user_content = (
        f"Preceding Dialogue Context:\n{context_str}\n\n"
        f"Target Line to Translate:\n{target_text}\n\n"
        f"{tgt_lang_name} Translation:"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    input_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_len:]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    del inputs
    del outputs

    output_text = re.sub(r'^(คำแปล|Translation|Thai|ไทย)[:\s]*', '', output_text, flags=re.IGNORECASE).strip()
    output_text = re.sub(r'^["\']|["\']$', '', output_text).strip()

    if tgt_lang_name == "Thai" and not has_thai_chars(output_text):
        logger.warning(f"Qwen output has no Thai: '{output_text}'. Trying Google fallback.")
        google_trans = translate_with_google(target_text, target_lang="th")
        if google_trans and has_thai_chars(google_trans):
            return google_trans

    return output_text

def translate_srt_file(input_path, output_path=None, source_lang="ja", target_lang="th", engine="ctranslate2", progress_callback=None, log_callback=None):
    """
    Translates an SRT subtitle file line-by-line while preserving timecodes using CTranslate2 or Qwen2.5 LLM.
    """
    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}.{target_lang}{ext}"

    def log(msg):
        logger.info(msg)
        if log_callback: log_callback(msg)

    engine_name = "Qwen2.5 LLM (Context-Aware GPU)" if "qwen" in str(engine).lower() else "CTranslate2 NLLB (GPU Fast)"
    log(f"📄 Starting Subtitle Translation: {input_path}")
    log(f"   Engine: {engine_name}")
    log(f"   Source: {source_lang} -> Target: {target_lang}")

    with open(input_path, 'r', encoding='utf-8-sig', errors='replace') as f:
        content = f.read()

    blocks = re.split(r'\r?\n\r?\n+', content.strip())
    parsed_blocks = []
    texts_to_translate = []

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) >= 3:
            idx = lines[0].strip()
            timing = lines[1].strip()
            text = " ".join([l.strip() for l in lines[2:]])
            parsed_blocks.append({"idx": idx, "timing": timing, "text": text})
            texts_to_translate.append(text)
        elif len(lines) == 2 and "-->" in lines[0]:
            timing = lines[0].strip()
            text = lines[1].strip()
            parsed_blocks.append({"idx": str(len(parsed_blocks)+1), "timing": timing, "text": text})
            texts_to_translate.append(text)

    if not parsed_blocks:
        raise ValueError("No valid subtitle blocks found in file.")

    total = len(texts_to_translate)
    translated_texts = []

    if "qwen" in str(engine).lower():
        log(f"   Found {total} subtitle entries. Translating with Qwen2.5 Context-Aware LLM...")
        history = []
        try:
            for i, text in enumerate(texts_to_translate):
                res = translate_with_qwen(text, context_history=history, source_lang=source_lang, target_lang=target_lang)
                if not res:
                    res = text
                translated_texts.append(res)
                history.append({"source": text, "target": res})
                if len(history) > 5:
                    history.pop(0)

                pct = min(1.0, (i + 1) / total)
                if progress_callback:
                    progress_callback(pct, f"Translated {i+1}/{total} lines")
                if (i + 1) % 5 == 0 or (i + 1) == total:
                    log(f"   Translated {i+1}/{total} subtitles (Qwen Context)...")
        finally:
            unload_qwen_model()
    else:
        log(f"   Found {total} subtitle entries. Translating with CTranslate2 GPU...")
        batch_size = 32
        try:
            for i in range(0, total, batch_size):
                chunk = texts_to_translate[i:i+batch_size]
                res = translate_with_ctranslate2(chunk, source_lang=source_lang, target_lang=target_lang)
                translated_texts.extend(res)

                pct = min(1.0, (i + len(chunk)) / total)
                if progress_callback:
                    progress_callback(pct, f"Translated {min(total, i+len(chunk))}/{total} lines")
                log(f"   Translated {min(total, i+len(chunk))}/{total} subtitles...")
        finally:
            unload_ct2_translator()

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, item in enumerate(parsed_blocks):
            trans_text = translated_texts[i] if i < len(translated_texts) and translated_texts[i] else item["text"]
            f.write(f"{item['idx']}\n{item['timing']}\n{trans_text}\n\n")

    log(f"✅ Subtitle translated successfully: {output_path}")
    return output_path

def unload_ollama_model():
    """
    Unloads the Ollama model from VRAM by setting keep_alive to 0.
    This releases GPU memory for other tasks.
    """
    try:
        # Send a request with keep_alive=0 to unload the model
        response = requests.post(Config.OLLAMA_URL, json={
            "model": Config.OLLAMA_MODEL,
            "prompt": "",
            "stream": False,
            "keep_alive": 0  # Immediately unload
        }, timeout=10)
        if response.status_code == 200:
            logger.info(f"✅ Ollama model '{Config.OLLAMA_MODEL}' unloaded from VRAM.")
        else:
            logger.warning(f"Could not unload Ollama model: {response.status_code}")
    except Exception as e:
        logger.warning(f"Failed to unload Ollama model: {e}")

# NLLB Model cache
_nllb_model = None
_nllb_tokenizer = None

def translate_with_nllb(text, source_lang="jpn_Jpan", target_lang="tha_Thai"):
    """
    Translates text using NLLB (No Language Left Behind) model locally.
    This runs entirely on your machine without API calls.
    
    Language codes:
    - Japanese: jpn_Jpan
    - Thai: tha_Thai
    - English: eng_Latn
    """
    global _nllb_model, _nllb_tokenizer
    
    # Import torch at module level to avoid scope issues
    import torch
    
    try:
        # Lazy load model
        if _nllb_model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            
            logger.info(f"Loading NLLB model: {Config.NLLB_MODEL}...")
            _nllb_tokenizer = AutoTokenizer.from_pretrained(Config.NLLB_MODEL, src_lang=source_lang)
            _nllb_model = AutoModelForSeq2SeqLM.from_pretrained(Config.NLLB_MODEL)
            
            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() and Config.USE_GPU else "cpu"
            _nllb_model = _nllb_model.to(device)
            _nllb_model.eval()
            logger.info(f"NLLB model loaded on {device}")
        
        # Tokenize
        inputs = _nllb_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Move inputs to same device as model
        device = next(_nllb_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate translation
        with torch.no_grad():
            # Force target language using convert_tokens_to_ids
            # NLLB uses special tokens like "__tha_Thai__" for target language
            forced_bos_token_id = _nllb_tokenizer.convert_tokens_to_ids(target_lang)
            
            generated_tokens = _nllb_model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
        
        # Decode
        translation = _nllb_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        # Validate: Must contain Thai characters if target is Thai
        if target_lang == "tha_Thai" and not has_thai_chars(translation):
            logger.warning(f"NLLB Output Invalid (No Thai): '{translation}'")
            return None
            
        return translation.strip()
        
    except Exception as e:
        logger.error(f"NLLB translation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def unload_nllb_model():
    """Unload NLLB model from VRAM."""
    global _nllb_model, _nllb_tokenizer
    if _nllb_model is not None:
        del _nllb_model
        del _nllb_tokenizer
        _nllb_model = None
        _nllb_tokenizer = None
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("✅ NLLB model unloaded from VRAM.")

def has_thai_chars(text):
    """Checks if the text contains any Thai characters."""
    return bool(re.search(r'[\u0e00-\u0e7f]', text))

def apply_gender_pronouns(text, gender="female"):
    """
    Enforces gender-appropriate pronouns and sentence-ending particles for Thai.
    - male: '\u0e09\u0e31\u0e19'/'\u0e2b\u0e19\u0e39' -> '\u0e1c\u0e21', '\u0e04\u0e48\u0e30'/'\u0e04\u0e30' -> '\u0e04\u0e23\u0e31\u0e1a'
    - female: '\u0e1c\u0e21' -> '\u0e09\u0e31\u0e19', '\u0e04\u0e23\u0e31\u0e1a' -> '\u0e04\u0e48\u0e30'
    """
    if not text:
        return text

    g = (gender or "female").lower()
    if g == "male":
        # Female 1st person pronouns -> Male
        text = re.sub(r'\u0e09\u0e31\u0e19', '\u0e1c\u0e21', text)
        text = re.sub(r'(\u0e25\u0e39\u0e01\u0e2b\u0e19\u0e39)|\u0e2b\u0e19\u0e39', lambda m: m.group(1) if m.group(1) else '\u0e1c\u0e21', text)
        # Ending particles
        text = re.sub(r'(\u0e19\u0e30\u0e04\u0e30|\u0e19\u0e30\u0e04\u0e48\u0e30|\u0e04\u0e48\u0e30|\u0e04\u0e30)', '\u0e04\u0e23\u0e31\u0e1a', text)
    elif g == "female":
        # Male 1st person pronouns -> Female (preserve hair compounds)
        text = re.sub(r'(\u0e17\u0e23\u0e07\u0e1c\u0e21|\u0e40\u0e2a\u0e49\u0e19\u0e1c\u0e21|\u0e15\u0e31\u0e14\u0e1c\u0e21|\u0e2a\u0e23\u0e30\u0e1c\u0e21|\u0e2b\u0e27\u0e35\u0e1c\u0e21|\u0e17\u0e33\u0e1c\u0e21|\u0e22\u0e49\u0e2d\u0e21\u0e1c\u0e21)|\u0e1c\u0e21', lambda m: m.group(1) if m.group(1) else '\u0e09\u0e31\u0e19', text)
        # Ending particles
        text = re.sub(r'(\u0e19\u0e30\u0e04\u0e23\u0e31\u0e1a|\u0e04\u0e23\u0e31\u0e1a)', '\u0e04\u0e48\u0e30', text)

    return text

def translate_with_ollama(text, target_lang="th", source_lang="auto", temperature=0.3, gender="female"):
    """
    Translates text using a local Ollama instance with valid JSON options.
    Supports Japanese/Thai/English translations.
    """
    
    current_text = text
    
    # Bridge Translation: Japanese -> English first if target is Thai
    if source_lang == "ja" and target_lang == "th":
        try:
            # More rigid prompt for bridge
            bridge_prompt = f"Translate the following Japanese text to English. Return ONLY the translation.\n\nJapanese: {text}\nEnglish:"
            response = requests.post(Config.OLLAMA_URL, json={
                "model": Config.OLLAMA_MODEL,
                "prompt": bridge_prompt,
                "stream": False,
                "options": {"temperature": 0.1} # Low temp for accuracy
            })
            response.raise_for_status()
            bridge_result = response.json().get("response", "").strip()
            
            # Sanity check: If bridge result looks like a refusal, fail to fallback
            if "provide the text" in bridge_result.lower() or "i cannot" in bridge_result.lower():
                logger.warning(f"Bridge Refusal detected: {bridge_result}")
                return None
                
            if bridge_result:
                current_text = bridge_result
                logger.info(f"Bridge (JA->EN): {text} -> {current_text}")
            else:
                logger.warning("Bridge returned empty text. Using original.")
                
        except Exception as e:
            logger.error(f"Bridge translation (JA->EN) failed: {e}")
            return None # Fail whole generic to fallback to Google

    # System prompt based on target language
    lang_name = "English" if target_lang == "en" else "Thai"
    if target_lang == "en":
        system_prompt = f"""You are a professional video translator specialized in English localization.
Task: Translate the text into natural, spoken English.

STRICT Rules:
1. **Meaning**: Fix broken sentences. Keep the original intent.
2. **Style**: Natural Spoken English.
3. **Output**: Return ONLY the English translation.

Text to Translate: {current_text}
English Translation:"""
    else:
        g = (gender or "female").lower()
        if g == "male":
            gender_rules = """6. **Speaker Gender & Pronouns (เพศชาย)**:
   - Speaker is MALE (ผู้ชาย).
   - Use 'ผม' (polite) or 'กู' (informal/raw) for 1st-person pronouns. NEVER use 'ฉัน' or 'หนู'.
   - Use 'ครับ', 'วะ', 'เว้ย' for ending particles. NEVER use 'ค่ะ', 'คะ', 'นะคะ'."""
        else:
            gender_rules = """6. **Speaker Gender & Pronouns (เพศหญิง)**:
   - Speaker is FEMALE (ผู้หญิง).
   - Use 'ฉัน' or 'หนู' for 1st-person pronouns. NEVER use 'ผม' or 'กระผม'.
   - Use 'ค่ะ', 'คะ', 'นะคะ', 'จ้ะ' for ending particles. NEVER use 'ครับ', 'นะครับ'."""

        system_prompt = f"""You are a professional video translator specialized in Thai localization (นักพากย์มืออาชีพ).
Task: Translate the text into natural, spoken Thai.

STRICT Rules:
1. **Meaning**: Fix broken sentences. Keep the original intent.
2. **Context**: Use "Thai Context" (บริบทคนไทย).
3. **Uncensored**: Do NOT censor explicit/sexual terms (e.g., เงี่ยน, หี, ควย, เย็ด). Use raw words.
4. **Style**: Spoken Language (ภาษาพูด). Remove excessive polite particles.
5. **Output**: Return ONLY the Thai translation.
{gender_rules}

Text to Translate: {current_text}
Thai Translation:"""

    try:
        response = requests.post(Config.OLLAMA_URL, json={
            "model": Config.OLLAMA_MODEL,
            "prompt": system_prompt, # Using prompt as full context + input often works better for simple models
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        })
        response.raise_for_status()
        result = response.json().get("response", "").strip()
        
        # Validation: Must contain Thai characters if target is Thai
        if target_lang == "th" and not has_thai_chars(result):
            logger.warning(f"Ollama Output Invalid (No Thai): '{result}'. Triggering Fallback.")
            return None
            
        return result
    except requests.RequestException as e:
        logger.error(f"Ollama request failed: {e}")
        return None

def translate_with_google(text, target_lang="th"):
    """
    Translates text using Google Translate (deep-translator).
    """
    try:
        # Loop for robust bridge if needed, but Google usually handles JA->TH well enough directly
        # Or we can do JA->EN->TH manually if Google supports it, but direct is likely fine.
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception as e:
        logger.error(f"Google Translate failed: {e}")
        return None

from utils import is_hallucination

def translate_text(segments, target_lang="th", source_lang="auto", stop_event=None, temperature=0.3, log_callback=None, enable_multitask=False, max_workers=4):
    """
    Translates text segments using the configured provider.
    
    Args:
        segments (list): List of dicts with 'text', 'start', 'end'.
        target_lang (str): Target language code.
        source_lang (str): Source language code (from Whisper).
        stop_event (threading.Event): Event to signal cancellation.
        log_callback (function): Function to send log messages to UI.
        enable_multitask (bool): Whether to translate segments in parallel.
        max_workers (int): Number of parallel threads.
        
    Returns:
        list: Updated segments with 'translated_text'.
    """
    import concurrent.futures

    def log(msg):
        logger.info(msg)
        if log_callback:
            log_callback(msg)

    provider = Config.TRANSLATION_PROVIDER
    mode_str = f"Multitask ({max_workers} threads)" if (enable_multitask and max_workers > 1) else "Sequential"
    log(f"Translating {len(segments)} segments using {provider} [{mode_str}] (Source: {source_lang} -> Target: {target_lang})...")
    
    if source_lang == target_lang and source_lang != "auto":
        log(f"Source language matches target language ({source_lang}). Skipping translation.")
        for seg in segments:
            seg['translated_text'] = seg['text']
        return segments

    def process_segment(item):
        i, seg = item
        if stop_event and stop_event.is_set():
            return seg

        original = seg['text']
        gender = seg.get('gender', 'female').lower()
        log(f"   🌐 [Translate {i+1}/{len(segments)}] Translating ({gender}): \"{original}\"")
        translation = None

        # 0. Try CTranslate2 (GPU Fast Offline)
        if provider == "local-ctranslate2":
            translation = translate_with_ctranslate2(original, source_lang=source_lang, target_lang=target_lang)
            if not translation:
                log(f"   ⚠️ CTranslate2 failed for segment {i+1}, falling back to Google Translate.")
                translation = translate_with_google(original, target_lang)

        # 0.1 Try Local Qwen LLM (Context-Aware GPU)
        elif provider == "local-qwen":
            ctx = []
            if i > 0:
                for prev_seg in segments[max(0, i-3):i]:
                    prev_src = prev_seg.get('text', '')
                    prev_trans = prev_seg.get('translated_text', '')
                    if prev_src and prev_trans:
                        ctx.append({"source": prev_src, "target": prev_trans})
            translation = translate_with_qwen(original, context_history=ctx, source_lang=source_lang, target_lang=target_lang, gender=gender)
            if not translation:
                log(f"   ⚠️ Qwen failed for segment {i+1}, falling back to Google Translate.")
                translation = translate_with_google(original, target_lang)

        # 1. Try Ollama if selected
        elif provider == "ollama":
            translation = translate_with_ollama(original, target_lang, source_lang, temperature, gender=gender)
            if not translation:
                log(f"   ⚠️ Ollama failed for segment {i+1}, falling back to Google Translate.")
                translation = translate_with_google(original, target_lang)
        
        # 2. Try Local Transformer (NLLB) - runs locally without API
        elif provider == "local-transformer":
            nllb_lang_map = {
                "ja": "jpn_Jpan",
                "th": "tha_Thai",
                "en": "eng_Latn",
                "zh": "zho_Hans",
                "ko": "kor_Hang"
            }
            nllb_source = nllb_lang_map.get(source_lang, "jpn_Jpan")
            nllb_target = nllb_lang_map.get(target_lang, "tha_Thai")
            
            translation = translate_with_nllb(original, nllb_source, nllb_target)
            if not translation:
                log(f"   ⚠️ NLLB failed for segment {i+1}, falling back to Google Translate.")
                translation = translate_with_google(original, target_lang)
        
        # 3. Try Google directly
        elif provider == "google":
            translation = translate_with_google(original, target_lang)
            
        # 4. OpenAI (Legacy support)
        elif provider == "openai":
            log("   ⚠️ OpenAI provider selected but logic replaced for free-tier. Using Google.")
            translation = translate_with_google(original, target_lang)

        # Hallucination Check & Fallback
        if translation and is_hallucination(translation):
            log(f"   ⚠️ Hallucination detected in segment {i+1}: '{translation[:50]}...'. Falling back to Google Translate.")
            fallback_trans = translate_with_google(original, target_lang)
            if fallback_trans and not is_hallucination(fallback_trans):
                log(f"   ✅ Google fallback succeeded for segment {i+1}")
                translation = fallback_trans
            else:
                log(f"   ⚠️ Fallback failed. Deduplicating repetition pattern...")
                cleaned = re.sub(r'(.{4,})\1+', r'\1', translation)
                if has_thai_chars(cleaned):
                    translation = cleaned
                elif fallback_trans:
                    translation = fallback_trans

        # Final Fallback: Ensure Thai output is not empty and not raw foreign text
        if not translation or (target_lang == "th" and not has_thai_chars(translation)):
            log(f"   ⚠️ Translation missing Thai text for segment {i+1}. Attempting emergency Google Translate...")
            google_trans = translate_with_google(original, target_lang)
            if google_trans and has_thai_chars(google_trans):
                translation = google_trans
            elif not translation:
                translation = original

        # Apply gender-specific pronouns and particles if target is Thai
        if target_lang == "th" and translation:
            translation = apply_gender_pronouns(translation, gender)

        seg['translated_text'] = translation
        log(f"   ✅ [Translate {i+1}/{len(segments)}] Result: \"{translation}\"")
        return seg

    items = list(enumerate(segments))
    # Local LLMs (Qwen) must run sequentially:
    # 1. Dialogue context history relies on previous turns having translated text.
    # 2. Concurrently loading/running LLM on 1 GPU risks CUDA OOM / contention.
    is_local_llm = (provider == "local-qwen")
    if enable_multitask and is_local_llm:
        log("   ℹ️ Qwen LLM translation uses rolling context: processing sequentially to preserve dialogue continuity and protect GPU VRAM.")

    if enable_multitask and max_workers > 1 and not is_local_llm:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(process_segment, items))
    else:
        for item in items:
            if stop_event and stop_event.is_set():
                log("Translation stopped by user.")
                break
            process_segment(item)
        
    logger.info("Translation complete.")
    
    # Unload models to free VRAM for TTS
    if provider == "ollama":
        unload_ollama_model()
    elif provider == "local-transformer":
        unload_nllb_model()
    elif provider == "local-ctranslate2":
        unload_ct2_translator()
    elif provider == "local-qwen":
        unload_qwen_model()
    
    return segments
