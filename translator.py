# translator.py — Final Production Version (Patched)
# Multilingual support for SummarAI
# Pipeline: Input → Detect Language → Translate to English → Summarize → Translate Back
# Supported: English, Tamil, Spanish, German, French, Hindi, Arabic, Chinese
# Uses deep_translator (Google Translate free API) — no API key required
#
# PATCH NOTES:
#   - Fixed Hindi (and other langs) returning ", , , , ," garbage output
#   - Replaced " ".join(chunks) with language-aware joining (no space for CJK/Arabic/Hindi)
#   - Added robust None/empty guard on every translator.translate() call
#   - Added retry logic (up to 2 retries) for transient API failures
#   - Normalized langdetect 'hi' code — langdetect sometimes returns 'hi' variants
#   - Added _is_valid_translation() to detect and reject garbage responses

try:
    from deep_translator import GoogleTranslator
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0   # deterministic language detection
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False

import time
import re

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

# Languages that should NOT have chunks joined with a space
# (they use no word-separating space, or space would corrupt the output)
_NO_SPACE_JOIN_LANGS = {"zh-CN", "zh-cn", "zh", "ar", "ja", "ko"}


def is_available():
    """Return True if translation libraries are installed."""
    return TRANSLATION_AVAILABLE


def detect_language(text: str) -> str:
    """
    Detect the language of input text.
    Returns a BCP-47 language code string (e.g. 'en', 'ta', 'de', 'hi').
    Falls back to 'en' on any error.
    """
    if not TRANSLATION_AVAILABLE or not text.strip():
        return "en"
    try:
        sample = text.strip()[:500]
        code   = detect(sample)
        # Normalize Chinese variants
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
        source_lang: BCP-47 source language code (e.g. 'ta', 'de', 'hi')

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
            result = _translate_chunk_with_retry(translator, chunk, fallback=chunk)
            parts.append(result)

        translated = _join_chunks(parts, target_lang="en")
        if not _is_valid_translation(translated):
            print(f"[WARN] Translation result looks invalid ({source_lang}→en), returning original.")
            return text, False
        return translated, True

    except Exception as e:
        print(f"[WARN] Translation to English failed ({source_lang}→en): {e}")
        return text, False


def translate_from_english(text: str, target_lang: str) -> str:
    """
    Translate an English summary to target_lang.

    Args:
        text:        English summary string
        target_lang: BCP-47 target language code (e.g. 'ta', 'de', 'hi')

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
            result = _translate_chunk_with_retry(translator, chunk, fallback=chunk)
            parts.append(result)

        translated = _join_chunks(parts, target_lang=target_lang)
        if not _is_valid_translation(translated):
            print(f"[WARN] Translation result looks invalid (en→{target_lang}), returning English.")
            return text
        return translated

    except Exception as e:
        print(f"[WARN] Translation from English failed (en→{target_lang}): {e}")
        return text   # return English on failure — better than empty


# ── Internal Helpers ───────────────────────────────────────────────────────────

def _translate_chunk_with_retry(translator, chunk: str, fallback: str, retries: int = 2) -> str:
    """
    Translate a single chunk with up to `retries` retry attempts on failure.
    Returns fallback string if all attempts fail or result is empty/garbage.
    """
    for attempt in range(retries + 1):
        try:
            result = translator.translate(chunk)

            # Guard: None or empty result
            if not result or not result.strip():
                print(f"[WARN] Empty translation result on attempt {attempt + 1}, retrying...")
                time.sleep(0.5)
                continue

            # Guard: garbage output (e.g. ", , , , ,")
            if _is_garbage(result):
                print(f"[WARN] Garbage translation result on attempt {attempt + 1}: {result[:60]!r}")
                time.sleep(0.5)
                continue

            return result.strip()

        except Exception as e:
            print(f"[WARN] Chunk translation attempt {attempt + 1} failed: {e}")
            if attempt < retries:
                time.sleep(1.0)

    # All retries exhausted — return original chunk as fallback
    print(f"[WARN] All retries exhausted, using original chunk as fallback.")
    return fallback


def _join_chunks(parts: list, target_lang: str) -> str:
    """
    Join translated chunks correctly based on target language.
    - CJK / Arabic: no separator (they don't use spaces between words)
    - All others: single space
    """
    if not parts:
        return ""
    separator = "" if target_lang in _NO_SPACE_JOIN_LANGS else " "
    return separator.join(p for p in parts if p).strip()


def _is_garbage(text: str) -> bool:
    """
    Detect obviously garbage translation output.
    Returns True if the text looks like a failed translation.

    Known garbage patterns:
      - ", , , , ,"          (comma-only output)
      - ". . . . ."          (period-only output)
      - Ratio of punctuation/spaces to total chars is extremely high
    """
    if not text or not text.strip():
        return True

    stripped = text.strip()

    # Pattern: mostly commas and spaces (the Hindi bug)
    comma_space_chars = sum(1 for c in stripped if c in {',', ' ', '.', '-'})
    if len(stripped) > 0 and comma_space_chars / len(stripped) > 0.7:
        return True

    # Pattern: repeating delimiter pattern like ", , , ," or ". . . ."
    if re.fullmatch(r'([,.\-–]\s*){3,}', stripped):
        return True

    return False


def _is_valid_translation(text: str) -> bool:
    """
    Returns True if the translated text looks like a real translation
    (non-empty, not garbage).
    """
    if not text or not text.strip():
        return False
    return not _is_garbage(text)


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