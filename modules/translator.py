import logging
import json
import requests
from deep_translator import GoogleTranslator
from config import Config

logger = logging.getLogger(__name__)

def translate_with_ollama(text, target_lang="th"):
    """
    Translates text using a local Ollama instance.
    """
    prompt = f"""You are a professional video translator specialized in Thai localization.
Task: Translate the following text into natural Thai (ภาษาไทย).
Rules:
1. Output ONLY the Thai translation.
2. Do NOT explain or add notes.
3. Use natural, spoken, and raw Thai (ภาษาพูด/ภาษาตลาด).
4. Do NOT use polite particles (No 'ครับ'/'ค่ะ').
5. Do NOT censor explicit, vulgar, or sexual content. Use direct, raw, and rude words if the original uses them (supports 18+ terms).
6. If the input is already Thai, return it as is.

Original Text: {text}
Thai Translation:"""

    try:
        response = requests.post(Config.OLLAMA_URL, json={
            "model": Config.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        })
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except requests.RequestException as e:
        logger.error(f"Ollama request failed: {e}")
        return None

def translate_with_google(text, target_lang="th"):
    """
    Translates text using Google Translate (deep-translator).
    """
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception as e:
        logger.error(f"Google Translate failed: {e}")
        return None

from utils import is_hallucination

def translate_text(segments, target_lang="th", stop_event=None):
    """
    Translates text segments using the configured provider.
    
    Args:
        segments (list): List of dicts with 'text', 'start', 'end'.
        target_lang (str): Target language code.
        stop_event (threading.Event): Event to signal cancellation.
        
    Returns:
        list: Updated segments with 'translated_text'.
    """
    provider = Config.TRANSLATION_PROVIDER
    logger.info(f"Translating {len(segments)} segments using {provider}...")
    
    for i, seg in enumerate(segments):
        if stop_event and stop_event.is_set():
            logger.info("Translation stopped by user.")
            return segments

        original = seg['text']
        translation = None
        
        # 1. Try Ollama if selected
        if provider == "ollama":
            translation = translate_with_ollama(original, target_lang)
            if not translation:
                logger.warning(f"Ollama failed for segment {i}, falling back to Google Translate.")
                # Fallback
                translation = translate_with_google(original, target_lang)
        
        # 2. Try Google directly
        elif provider == "google":
            translation = translate_with_google(original, target_lang)
            
        # 3. OpenAI (Legacy support)
        elif provider == "openai":
            # We implemented this previously, but for this free-tier request, 
            # we'll skip adding the full OpenAI logic again to keep it clean,
            # or we could keep it if the user switches back. 
            # For now, let's treat it as "not implemented" or fallback to google
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
