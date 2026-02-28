import json
import os
import streamlit as st

class I18nManager:
    def __init__(self, locales_dir="locales"):
        # Use absolute path to avoid issues with current working directory
        self.locales_dir = os.path.join(os.getcwd(), locales_dir)
        self.translations = {}
        self._load_translations()

    def _load_translations(self):
        if not os.path.exists(self.locales_dir):
            return
            
        for file in os.listdir(self.locales_dir):
            if file.endswith(".json"):
                lang = file.split(".")[0]
                try:
                    with open(os.path.join(self.locales_dir, file), "r", encoding="utf-8") as f:
                        self.translations[lang] = json.load(f)
                except Exception as e:
                    print(f"Error loading translation file {file}: {e}")

    def get_text(self, key, **kwargs):
        # Get current language from session state, default to English
        lang = st.session_state.get("language", "en")
        
        # Fallback logic:
        # 1. Try current language
        # 2. Try 'en' (English)
        # 3. Return key itself
        
        text = self.translations.get(lang, {}).get(key)
        if text is None:
            text = self.translations.get("en", {}).get(key, key)
            
        # Format string with kwargs if provided
        if kwargs and isinstance(text, str):
            try:
                return text.format(**kwargs)
            except:
                return text
                
        return text

# Global instance
i18n = I18nManager()

def t(key, **kwargs):
    return i18n.get_text(key, **kwargs)
