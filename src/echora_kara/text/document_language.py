"""Language of one TTML lyric document, for manifest metadata and normalizer hints.

A document's ``xml:lang`` is often wrong (K-pop tagged ``en``, romanised Hindi tagged ``en``). The
script of the sung text is definitive where it is not Latin, so a dominant non-Latin script
decides; Han-only text is Cantonese when the document's own transliteration is Jyutping, Mandarin
otherwise; Latin text keeps the (base) tag. Nothing here changes acoustic labels on its own:
``normalizer_language`` maps the result onto the hints the shared normalizer accepts.
Elements are matched with or without a namespace prefix.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

_XML_LANG = re.compile(r'<(?:[\w.-]+:)?tt\b[^>]*\bxml:lang="([^"]*)"')
_TRANSLIT = re.compile(r'<(?:[\w.-]+:)?transliteration\b[^>]*\bxml:lang="([^"]*)"')
_BODY = re.compile(r"<(?:[\w.-]+:)?body\b.*?</(?:[\w.-]+:)?body>", re.S)
_TAG = re.compile(r"<[^>]+>")

# Unicode script (first word of the character name) -> language when that script dominates.
SCRIPT_LANGUAGE = {
    "HANGUL": "ko", "HIRAGANA": "ja", "KATAKANA": "ja", "CJK": "zh", "ARABIC": "ar",
    "DEVANAGARI": "hi", "GURMUKHI": "pa", "TAMIL": "ta", "TELUGU": "te", "THAI": "th",
    "CYRILLIC": "ru", "GREEK": "el", "HEBREW": "he", "BENGALI": "bn", "GUJARATI": "gu",
    "KANNADA": "kn", "MALAYALAM": "ml", "SINHALA": "si", "KHMER": "km", "LAO": "lo",
    "MYANMAR": "my", "GEORGIAN": "ka", "ARMENIAN": "hy", "ETHIOPIC": "am", "TIBETAN": "bo",
    "ORIYA": "or", "MONGOLIAN": "mn",
}
# Tags that already name a language written in the dominant script are kept (Urdu in Arabic
# script, Marathi in Devanagari, Ukrainian in Cyrillic, ...).
SCRIPT_TAGS = {
    "ARABIC": {"ar", "ur", "fa", "ps", "ku", "sd"}, "DEVANAGARI": {"hi", "mr", "ne", "sa", "bho"},
    "CYRILLIC": {"ru", "uk", "bg", "sr", "mk", "be", "kk", "ky", "mn", "tg"},
    "CJK": {"zh", "yue"}, "HANGUL": {"ko"}, "HIRAGANA": {"ja"}, "KATAKANA": {"ja"},
}
BASE_TAG = {"jp": "ja", "fil": "tl", "zxx": "und", "mul": "und", "": "und"}


def script_counts(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for char in text:
        if unicodedata.category(char)[0] != "L":
            continue
        counts[unicodedata.name(char, "UNKNOWN").split(" ", 1)[0]] += 1
    return counts


def base_tag(tag: str | None) -> str:
    tag = (tag or "").strip().lower()
    base = tag.split("-", 1)[0]
    return BASE_TAG.get(base, base)


def document_language(ttml: str) -> dict:
    tag = _XML_LANG.search(ttml)
    xml_lang = tag.group(1) if tag else ""
    translit = _TRANSLIT.search(ttml)
    translit_lang = translit.group(1).lower() if translit else ""
    body = _BODY.search(ttml)
    text = _TAG.sub(" ", body.group(0)) if body else ""
    counts = script_counts(text)
    letters = sum(counts.values())
    base = base_tag(xml_lang)
    result = {"xml_lang": xml_lang, "transliteration": translit_lang, "letters": letters,
              "scripts": {k: v for k, v in counts.most_common(4)}}
    # A non-Latin script is strong evidence even when English words outnumber it (K-pop
    # choruses, Latin loanwords), so a quarter of the letters is enough to decide.
    kana = counts["HIRAGANA"] + counts["KATAKANA"]
    han = counts["CJK"]
    if letters and kana + han >= 0.25 * letters and kana >= 0.05 * letters:
        return {**result, "language": "ja", "source": "script"}
    if letters and han >= 0.25 * letters:
        if "jyutping" in translit_lang or base == "yue":
            return {**result, "language": "yue", "source": "transliteration" if "jyutping" in translit_lang else "xml_lang"}
        return {**result, "language": "zh", "source": "script"}
    dominant, count = next(((k, v) for k, v in counts.most_common() if k != "LATIN"), ("", 0))
    if dominant in SCRIPT_LANGUAGE and count >= 0.25 * letters:
        if base in SCRIPT_TAGS.get(dominant, ()):
            return {**result, "language": base, "source": "xml_lang"}
        return {**result, "language": SCRIPT_LANGUAGE[dominant], "source": "script"}
    return {**result, "language": base or "und", "source": "xml_lang"}


def document_digest(ttml: str) -> str:
    """Content hash of one TTML document (cache key of its labels and language decision)."""
    import hashlib
    return hashlib.sha256(ttml.encode()).hexdigest()


def lyric_text(ttml: str) -> str:
    """Sung text of a TTML document for language identification (head dropped)."""
    text = re.sub(r"<[^>]+>", " ", re.sub(r"<(?:\w+:)?head\b.*?</(?:\w+:)?head>", " ", ttml, flags=re.S))
    return re.sub(r"\s+", " ", text).strip()


# Languages the shared normalizer accepts as an explicit hint (lyrics_normalizer.process_haruhi_line);
# everything else is ``auto`` there, exactly as the inference allowlist does. Label rules v2 also
# forward en/de/fr (pronunciation labels) and tr (uroman's Turkish rules).
V1_HINTS = {"ja": "ja", "ko": "ko", "yue": "yue", "zh": "zh", "ur": "ur", "hi": "hi", "pa": "pa"}
V2_HINTS = {**V1_HINTS, "en": "en", "de": "de", "fr": "fr", "tr": "tr"}

# Mirrors label_rules.NON_LATIN_SCRIPT_LANGUAGES without importing the normalizer dependencies.
V3_HINTS = {**V2_HINTS, **{language: language for language in (
    "ko", "ja", "zh", "yue", "th", "ar", "fa", "ur", "he", "hi", "bn", "ta", "te", "mr", "gu", "kn",
    "ml", "pa", "ru", "uk", "bg", "sr", "mk", "be", "kk", "el", "ka", "hy", "km", "lo", "my",
)}}

def normalizer_language(language: str, rules: str = "v1") -> str:
    return (V1_HINTS if rules == "v1" else V3_HINTS if rules in ("v3", "v3-orthographic")
            else V2_HINTS).get(language, "auto")
