# translator.py — Final Production Version
# Multilingual support for SummarAI
# Pipeline: Input → Detect Language → Translate to English → Summarize → Translate Back
# Supported: English, Tamil, Spanish, German, French, Hindi, Arabic, Chinese
# Uses deep_translator (Google Translate free API) — no API key required

try:
    from deep_translator import GoogleTranslator
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0   # deterministic language detection
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False

# ── Supported Languages ────────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "English": "en",
    "Tamil":   "ta",
    "Spanish": "es",
    "German":  "de",
    "French":  "fr",
    "Hindi":   "hi",
    "Arabic":  "ar",
    "Chinese": "zh-CN",
}

CODE_TO_NAME = {v: k for k, v in SUPPORTED_LANGUAGES.items()}

# Google Translate max chars per request (safe limit)
_MAX_CHARS = 3800


def is_available():
    """Return True if translation libraries are installed."""
    return TRANSLATION_AVAILABLE


def detect_language(text: str) -> str:
    """
    Detect the language of input text.
    Returns a BCP-47 language code string (e.g. 'en', 'ta', 'de').
    Falls back to 'en' on any error.
    """
    if not TRANSLATION_AVAILABLE or not text.strip():
        return "en"
    try:
        sample = text.strip()[:500]
        code   = detect(sample)
        # langdetect returns 'zh-cn' — normalize to match our map
        if code.startswith("zh"):
            return "zh-CN"
        return code
    except Exception:
        return "en"


def get_language_name(code: str) -> str:
    """Return the display name for a language code, or the code itself if unknown."""
    return CODE_TO_NAME.get(code, code.upper())


def translate_to_english(text: str, source_lang: str):
    """
    Translate text from source_lang to English.

    Args:
        text:        Input text string
        source_lang: BCP-47 source language code (e.g. 'ta', 'de')

    Returns:
        (translated_text, success_flag)
        success_flag is False if translation was skipped or failed.
    """
    if not text or not text.strip():
        return text, True
    if not TRANSLATION_AVAILABLE:
        return text, False
    if source_lang == "en":
        return text, True

    try:
        chunks     = _split_into_chunks(text, max_chars=_MAX_CHARS)
        translator = GoogleTranslator(source=source_lang, target="en")
        parts      = []
        for chunk in chunks:
            try:
                result = translator.translate(chunk)
                parts.append(result if result else chunk)
            except Exception as e:
                print(f"[WARN] Chunk translation failed ({source_lang}→en): {e}")
                parts.append(chunk)   # keep original chunk on failure
        translated = " ".join(parts)
        return translated.strip(), True
    except Exception as e:
        print(f"[WARN] Translation to English failed ({source_lang}→en): {e}")
        return text, False


def translate_from_english(text: str, target_lang: str) -> str:
    """
    Translate an English summary to target_lang.

    Args:
        text:        English summary string
        target_lang: BCP-47 target language code (e.g. 'ta', 'de')

    Returns:
        Translated string, or original English string on failure.
    """
    if not text or not text.strip():
        return text
    if not TRANSLATION_AVAILABLE:
        return text
    if target_lang == "en":
        return text

    try:
        chunks     = _split_into_chunks(text, max_chars=_MAX_CHARS)
        translator = GoogleTranslator(source="en", target=target_lang)
        parts      = []
        for chunk in chunks:
            try:
                result = translator.translate(chunk)
                parts.append(result if result else chunk)
            except Exception as e:
                print(f"[WARN] Chunk translation failed (en→{target_lang}): {e}")
                parts.append(chunk)
        return " ".join(parts).strip()
    except Exception as e:
        print(f"[WARN] Translation from English failed (en→{target_lang}): {e}")
        return text   # return English on failure — better than empty


def _split_into_chunks(text: str, max_chars: int = 3800) -> list:
    """
    Split text into chunks ≤ max_chars without breaking sentences.
    Tries to split at sentence boundaries ('. '), then word boundaries (' ').
    """
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks    = []
    remaining = text.strip()

    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        # Try to split at last sentence boundary within limit
        split_at = remaining.rfind(". ", 0, max_chars)
        if split_at != -1:
            split_at += 1   # include the period
        else:
            # Fall back to last word boundary
            split_at = remaining.rfind(" ", 0, max_chars)
        if split_at == -1 or split_at == 0:
            split_at = max_chars  # hard split as last resort

        chunk     = remaining[:split_at].strip()
        remaining = remaining[split_at:].strip()

        if chunk:
            chunks.append(chunk)

    return [c for c in chunks if c.strip()]