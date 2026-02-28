from langdetect import detect

def detect_language(text: str) -> str:
    """
    Detect language of the text.
    Args:
        text: Text content to analyze
    Returns:
        str: Language code (e.g., 'en', 'zh-cn', 'ja')
    """
    if not text or len(text.strip()) == 0:
        return "en" # Default to English
    
    try:
        lang = detect(text)
        return lang
    except Exception as e:
        print(f"Language detection failed: {e}")
        return "en"

def get_language_name(lang_code: str) -> str:
    """
    Map language code to full language name for LLM prompt.
    """
    mapping = {
        'en': 'English',
        'zh-cn': 'Simplified Chinese',
        'zh-tw': 'Traditional Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'fr': 'French',
        'de': 'German',
        'es': 'Spanish',
        'ru': 'Russian',
        'it': 'Italian',
        'pt': 'Portuguese'
    }
    # Return mapped name or default description
    return mapping.get(lang_code.lower(), "the same language as the chat history")
