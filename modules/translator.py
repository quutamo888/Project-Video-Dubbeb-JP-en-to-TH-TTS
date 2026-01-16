import logging
import json
import requests
import re
from deep_translator import GoogleTranslator
from config import Config

logger = logging.getLogger(__name__)

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
        
        # Validate: Must contain Thai characters
        if not has_thai_chars(translation):
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

def translate_with_ollama(text, target_lang="th", source_lang="auto", temperature=0.3):
    """
    Translates text using a local Ollama instance with valid JSON options.
    Supports JP -> EN -> TH bridge translation.
    """
    
    current_text = text
    
    # Bridge Translation: Japanese -> English first
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

    # Main Translation (EN/Auto -> TH)
    system_prompt = f"""You are a professional video translator specialized in Thai localization (นักพากย์มืออาชีพ).
Task: Translate the text into natural, spoken Thai.

STRICT Rules:
1. **Meaning**: Fix broken sentences. Keep the original intent.
2. **Context**: Use "Thai Context" (บริบทคนไทย).
3. **Uncensored**: Do NOT censor explicit/sexual terms (e.g., เงี่ยน, หี, ควย, เย็ด). Use raw words.
4. **Style**: Spoken Language (ภาษาพูด). Remove excessive polite particles.
5. **Output**: Return ONLY the Thai translation.

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
        
        # Validation: Must contain Thai characters
        if not has_thai_chars(result):
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

def translate_text(segments, target_lang="th", source_lang="auto", stop_event=None, temperature=0.3):
    """
    Translates text segments using the configured provider.
    
    Args:
        segments (list): List of dicts with 'text', 'start', 'end'.
        target_lang (str): Target language code.
        source_lang (str): Source language code (from Whisper).
        stop_event (threading.Event): Event to signal cancellation.
        
    Returns:
        list: Updated segments with 'translated_text'.
    """
    provider = Config.TRANSLATION_PROVIDER
    logger.info(f"Translating {len(segments)} segments using {provider} (Source: {source_lang})...")
    
    for i, seg in enumerate(segments):
        if stop_event and stop_event.is_set():
            logger.info("Translation stopped by user.")
            return segments

        original = seg['text']
        translation = None
        
        # 1. Try Ollama if selected
        if provider == "ollama":
            translation = translate_with_ollama(original, target_lang, source_lang, temperature)
            if not translation:
                logger.warning(f"Ollama failed for segment {i}, falling back to Google Translate.")
                # Fallback
                translation = translate_with_google(original, target_lang)
        
        # 2. Try Local Transformer (NLLB) - runs locally without API
        elif provider == "local-transformer":
            # Map language codes to NLLB format
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
                logger.warning(f"NLLB failed for segment {i}, falling back to Google Translate.")
                translation = translate_with_google(original, target_lang)
        
        # 3. Try Google directly
        elif provider == "google":
            translation = translate_with_google(original, target_lang)
            
        # 4. OpenAI (Legacy support)
        elif provider == "openai":
            logger.warning("OpenAI provider selected but logic replaced for free-tier. Using Google.")
            translation = translate_with_google(original, target_lang)

        # Hallucination Check
        if translation and is_hallucination(translation):
            logger.warning(f"Hallucination detected in segment {i}: '{translation[:50]}...'. Reverting to original.")
            translation = original # or ""
            
        # Final Fallback
        if not translation:
            translation = original
            
        seg['translated_text'] = translation
        # logger.debug(f"Seg {i}: {original} -> {translation}")
        
    logger.info("Translation complete.")
    
    # Unload models to free VRAM for TTS
    if provider == "ollama":
        unload_ollama_model()
    elif provider == "local-transformer":
        unload_nllb_model()
    
    return segments
