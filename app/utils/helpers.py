from langdetect import detect, DetectorFactory

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