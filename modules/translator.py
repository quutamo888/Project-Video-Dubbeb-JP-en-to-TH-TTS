import logging
import json
import requests
import re
from deep_translator import GoogleTranslator
from config import Config

logger = logging.getLogger(__name__)

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
        
        # 2. Try Google directly
        elif provider == "google":
            translation = translate_with_google(original, target_lang)
            
        # 3. OpenAI (Legacy support)
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
    return segments
