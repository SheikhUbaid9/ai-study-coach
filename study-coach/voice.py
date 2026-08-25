"""Voice in and voice out.

**Listening** uses Whisper through whichever provider you're already on — Groq
serves `whisper-large-v3` on the same OpenAI-compatible endpoint and the same key,
so speech-to-text costs no extra setup.

**Speaking** uses gTTS, deliberately not the provider. It needs no key, no credits
and no quota, so the app never loses its voice because an API budget ran out. It
also handles Urdu and Hindi, which matters for a Roman-Urdu-speaking student.

Both are optional: if the pieces aren't installed or the provider can't transcribe,
the app falls back to typing and reading. Voice should never be load-bearing.
"""

from __future__ import annotations

import io
import os
import re

# Whisper on Groq. Other providers name their models differently, so this is
# overridable rather than assumed.
STT_MODEL = os.environ.get("STT_MODEL", "whisper-large-v3")

# gTTS language codes. Roman Urdu is written in Latin script but *sounds* like
# Urdu, so reading it with an English voice mangles it — 'ur' is the better match.
TTS_LANGS = {"Auto (match the reply)": None, "English": "en", "Urdu": "ur", "Hindi": "hi"}

# Roman Urdu is the hard case: Latin letters, Urdu words. Script detection can't
# see it, so we look for function words that are common in Urdu/Hindi and rare in
# English. Deliberately excludes ambiguous ones ("main" is English too, "hai"
# appears in names) and requires several hits, because mistakenly reading an
# English answer with an Urdu voice is worse than the reverse.
_ROMAN_URDU_HINTS = {
    "hain", "nahi", "nahin", "kya", "kyun", "kyu", "karo", "karna", "karta",
    "karte", "karke", "mujhe", "mujhay", "tumhe", "aap", "aapko", "acha",
    "accha", "theek", "thik", "bohat", "bahut", "zyada", "thora", "thoda",
    "samajh", "samjha", "samjhao", "batao", "bata", "dekho", "chahiye",
    "chahiyay", "kaise", "kaisay", "phir", "abhi", "yeh", "woh", "iska",
    "uska", "mera", "meri", "tera", "hoga", "hogi", "raha", "rahi", "rahe",
    "liye", "sath", "saath", "waqt", "din", "kaam", "baat", "bhi", "lekin",
    "magar", "aur", "shukriya", "zaroori", "asaan", "mushkil",
}


def detect_language(text: str) -> str:
    """Best guess at the language of `text`, as a gTTS code.

    Three tiers, cheapest first — no model call, no network:
      1. Arabic/Urdu script     -> 'ur'
      2. Devanagari script      -> 'hi'
      3. Latin script: count Roman-Urdu function words -> 'ur' if several
    Falls back to English, which is the safe default for a mixed-language reply
    full of technical terms.
    """
    if not text:
        return "en"

    # Script detection is decisive when it fires.
    if re.search(r"[؀-ۿݐ-ݿ]", text):
        return "ur"
    if re.search(r"[ऀ-ॿ]", text):
        return "hi"

    words = set(re.findall(r"[a-z]+", text.lower()))
    hits = len(words & _ROMAN_URDU_HINTS)
    # Two hits in a short message, three in a long one — scale the bar with
    # length so a single stray word in a long English answer doesn't flip it.
    threshold = 2 if len(words) < 40 else 3
    return "ur" if hits >= threshold else "en"


# --------------------------------------------------------------------------
# Speech -> text
# --------------------------------------------------------------------------

def stt_available() -> tuple[bool, str]:
    try:
        from llm import _resolve

        _resolve()
    except Exception as exc:
        return False, str(exc).splitlines()[0]
    return True, STT_MODEL


def transcribe(audio_bytes: bytes, filename: str = "speech.wav") -> tuple[str | None, str | None]:
    """Return (text, error). Never raises — a failed transcription should leave
    the student typing, not staring at a stack trace."""
    if not audio_bytes:
        return None, "no audio captured"

    try:
        from openai import OpenAI

        from llm import _resolve

        base_url, _model, api_key = _resolve()
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0)

        # The API needs a file-like object with a name to infer the format.
        buf = io.BytesIO(audio_bytes)
        buf.name = filename
        result = client.audio.transcriptions.create(model=STT_MODEL, file=buf)
        text = (getattr(result, "text", "") or "").strip()
        return (text, None) if text else (None, "transcription came back empty")
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
# Text -> speech
# --------------------------------------------------------------------------

def tts_available() -> bool:
    try:
        import gtts  # noqa: F401

        return True
    except ImportError:
        return False


def strip_for_speech(text: str, max_chars: int = 1200) -> str:
    """Turn written-for-the-eye text into something worth listening to.

    Markdown read aloud is miserable — "hash hash Step one, star star torque star
    star". Bullets, emphasis and citation markers all have to go, and long answers
    get truncated at a sentence boundary rather than mid-word.
    """
    t = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)      # code blocks
    t = re.sub(r"`([^`]*)`", r"\1", t)                         # inline code
    t = re.sub(r"^#{1,6}\s*", "", t, flags=re.MULTILINE)       # headings
    t = re.sub(r"\*\*|__|\*|_", "", t)                         # emphasis
    t = re.sub(r"^\s*[-•*]\s+", "", t, flags=re.MULTILINE)     # bullets
    t = re.sub(r"\[\d+\]", "", t)                              # [1] citations
    # Table separator rows (|---|---|) become "dash dash dash" if left in.
    t = re.sub(r"^\s*\|?[\s:|-]*\|[\s:|-]*$", "", t, flags=re.MULTILINE)
    t = re.sub(r"\|", " ", t)                                  # remaining pipes
    t = re.sub(r"(?<!\w)-{2,}(?!\w)", " ", t)                  # stray rules
    t = re.sub(r"[#>]+", " ", t)
    # Emoji and symbols read as noise.
    t = re.sub(r"[\U0001F000-\U0001FAFF☀-➿]", "", t)
    t = re.sub(r"\n{2,}", ". ", t)
    t = re.sub(r"\s+", " ", t).strip()

    if len(t) > max_chars:
        cut = t[:max_chars]
        stop = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
        t = (cut[: stop + 1] if stop > max_chars // 2 else cut) + " …"
    return t


def speak(text: str, language: str = "Auto (match the reply)") -> tuple[bytes | None, str | None]:
    """Return (mp3_bytes, error). Never raises.

    `language` is a key from TTS_LANGS. The Auto entry maps to None, which means
    "work it out from the text" — so an English answer is read in English and an
    Urdu one in Urdu without the student touching a setting.
    """
    if not tts_available():
        return None, "gTTS not installed — pip install gtts"

    clean = strip_for_speech(text)
    if not clean:
        return None, "nothing to read out"

    lang = TTS_LANGS.get(language) or detect_language(clean)

    try:
        from gtts import gTTS

        buf = io.BytesIO()
        gTTS(clean, lang=lang, slow=False).write_to_fp(buf)
        return buf.getvalue(), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
