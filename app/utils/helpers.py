from langdetect import detect, DetectorFactory
from typing import Tuple

# Ensure consistent results
DetectorFactory.seed = 0

def detect_language(text: str) -> str:
    """
    Detect the language of the given text
    Returns: 'en' for English, 'bn' for Bangla, or 'unknown'
    """
    try:
        lang = detect(text)
        return lang
    except:
        return "unknown"

def translate_text(text: str, target_lang: str = "en") -> str:
    """
    Simple translation function (placeholder for actual translation service)
    In a real implementation, you would use Google Translate API or similar
    """
    # This is a placeholder - in a real implementation, you would use a translation API
    return text