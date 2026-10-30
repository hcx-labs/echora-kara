"""Training adapter for the acoustic-label normalizer.

Training labels use the same normalizer and language allowlist as inference.
The existing TTML builder supplies no override, so defaults to ``auto``;
callers reproducing an explicit inference override may pass ``language``.
This module does not guess a different transliteration.
"""

from __future__ import annotations

import functools
import os
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import TypedDict

from echora_kara.text.lyrics_normalizer import process_haruhi_line  # noqa: E402

from echora_kara.text import korean_pronunciation  # noqa: E402
from echora_kara.text import label_rules  # noqa: E402
from echora_kara.text import script_romanization  # noqa: E402

_LABEL = re.compile(r"^[a-z]+$")

# Hints the normalizer honours that the inference language allowlist does not forward. Training
# manifests may use them (Cantonese from a Jyutping transliteration); a model trained on them needs
# the allowlist extended before inference relies on the hint.
TRAINING_ONLY_LANGUAGES = frozenset({"yue"})
# Label rules (label_rules.RULES): "v1" is the upstream output; "v2" adds pronunciation labels for
# en/de/fr, numbers, acronyms, vocables, Indic/Tamil/Arabic/Japanese fixes; "v3" adds a lyric lexicon,
# homographs, dropped g, spell-outs and per-line language. The builder passes it explicitly; ECHORA_KARA_LABEL_RULES is the default for tests and tools.
LABEL_RULES = os.environ.get("ECHORA_KARA_LABEL_RULES", "v1")
# Hints inference forwards per rule set (see inference_language).
INFERENCE_LANGUAGES_V1 = frozenset({"auto", "ja", "jaen", "zhen", "ko", "ur", "hi", "pa", "indic"})
INFERENCE_LANGUAGES_V2 = INFERENCE_LANGUAGES_V1 | frozenset({"en", "de", "fr", "tr", "yue", "zh"})
INFERENCE_LANGUAGES_V3 = INFERENCE_LANGUAGES_V2 | label_rules.NON_LATIN_SCRIPT_LANGUAGES


def normalizer_version(rules: str | None = None) -> str:
    """Version string recorded with labels made under ``rules`` (default ``LABEL_RULES``): the Korean and script
    romanization versions, plus the label-rules version and name for v2 and later."""
    rules = label_rules.resolve(rules if rules is not None else LABEL_RULES)
    if not rules.v2:
        return f"{korean_pronunciation.VERSION}+{script_romanization.VERSION}"
    return f"{korean_pronunciation.VERSION_V2}+{script_romanization.VERSION}+{rules.version}:{rules.name}"


NORMALIZER_VERSION = normalizer_version("v1")


class AcousticToken(TypedDict):
    """One acoustic token of a lyric line: its a-z label, its surface text and the normalizer item type."""

    text: str
    surface: str
    normalizer_type: int


class NormalizationError(ValueError):
    """The inference normalizer could not produce safe acoustic labels."""


def inference_language(language: str | None = None, rules: str | None = None) -> str:
    """Mirror the inference language allowlist exactly, per label rule set.

    Dataset language metadata is not automatically a pronunciation override. Under v1 en/es/fr
    reach inference as auto; under v2 (ECHORA_KARA_LABEL_RULES=v2) en/de/fr/tr/yue/zh are forwarded
    because the v2 labels depend on them.
    """
    selected = label_rules.resolve(rules if rules is not None else LABEL_RULES)
    allowed = (
        INFERENCE_LANGUAGES_V3 if selected.v3 else INFERENCE_LANGUAGES_V2 if selected.v2 else INFERENCE_LANGUAGES_V1
    )
    return language if language in allowed else "auto"


def training_language(language: str | None = None, rules: str | None = None) -> str:
    """Inference selection plus the training-only hints (see TRAINING_ONLY_LANGUAGES)."""
    return language if language in TRAINING_ONLY_LANGUAGES else inference_language(language, rules)


def _normalize_line(text: str, language: str, rules: str | None = None) -> list[dict]:
    """Inference normalizer output, with Hangul relabelled by sound.

    Inference relabels Hangul the same way (inference.cli.normalize_source_lines).
    """
    rules = label_rules.resolve(rules if rules is not None else LABEL_RULES).name
    # v1 keeps the original call shape (normalizers patched in tests).
    extra = {} if rules == "v1" else {"rules": rules}
    if rules == "v1":
        return korean_pronunciation.apply_to_items(process_haruhi_line(text, language))
    return korean_pronunciation.apply_to_items(process_haruhi_line(text, language, **extra), v2=True)


def acoustic_tokens(text: str, language: str | None = None, rules: str | None = None) -> list[AcousticToken]:
    """Return acoustic pronunciations for a complete lyric line.

    Punctuation produces no acoustic token. Every retained pronunciation must
    be lowercase ASCII letters, the exact label alphabet accepted by the MMS
    fine-tuning manifest.
    """
    try:
        normalized = _normalize_line(text, training_language(language, rules), rules)
    except (AssertionError, IndexError, KeyError, TypeError, ValueError) as error:
        raise NormalizationError(f"normalizer_error:{error}") from error

    tokens: list[AcousticToken] = []
    for item in normalized:
        pronunciation = item.get("pron", "")
        if not pronunciation:
            continue
        if not _LABEL.fullmatch(pronunciation):
            raise NormalizationError(f"unsafe_pronunciation:{pronunciation!r}")
        tokens.append(
            {
                "text": pronunciation,
                "surface": str(item.get("orig", "")),
                "normalizer_type": int(item.get("type", -1)),
            }
        )
    if not tokens:
        raise NormalizationError("no_acoustic_tokens")
    return tokens


# Small kana, sokuon and the long-vowel mark extend the preceding kana.
_KANA_MODIFIERS = set("ぁぃぅぇぉゃゅょゎァィゥェォャュョヮッっー")
# Particle readings a standalone kana lookup can return either way.
_KANA_ALTERNATES = {"は": {"ha", "wa"}, "へ": {"he", "e"}, "を": {"wo", "o"}}
_SOKUON = "っッ"
_CONSONANTS = "bcdfghjkmnprstwyz"
# Rendaku and handakuten inside compounds: 言葉 koto+ba, 一杯 ip+pai.
_COMPOUND_ONSETS = (
    ("sh", "j"),
    ("ch", "j"),
    ("ts", "z"),
    ("k", "g"),
    ("s", "z"),
    ("t", "d"),
    ("h", "b"),
    ("h", "p"),
    ("f", "b"),
    ("f", "p"),
)
_SOKUON_TAILS = ("tsu", "chi", "ku", "ki")


DICTIONARY_DIR = Path(os.environ.get("ECHORA_KARA_DICTIONARY_DIR", Path(__file__).resolve().parent / "dictionaries"))


def _is_kana(char: str) -> bool:
    return "ぁ" <= char <= "ゟ" or ("゠" <= char <= "ヿ" and char != "・")


def _is_kanji(char: str) -> bool:
    return "一" <= char <= "鿿" or "㐀" <= char <= "䶿" or char in "々〆ヶ"


def _is_silent(char: str) -> bool:
    """Spaces, apostrophes and punctuation such as ・ carry no pronunciation."""
    return unicodedata.category(char)[0] in "PZ"


# Letters the normalizer may spell differently: garçons → garsons, fällt → faellt.
_LATIN_SPELLINGS = {
    "ç": {"s"},
    "ä": {"ae"},
    "ö": {"oe"},
    "ü": {"ue"},
    "ß": {"ss", "s"},
    "æ": {"ae"},
    "œ": {"oe"},
    "ø": {"o", "oe"},
    "ñ": {"ny"},
}
# v2: Turkish ı/ş/ğ and a few more letters need an anchor spelling, or Turkish tokens are masked at
# syllable timing boundaries; ç may read ch under the Turkish rules.
_LATIN_SPELLINGS_V2 = {
    **_LATIN_SPELLINGS,
    "ç": {"s", "ch"},
    "ı": {"i"},
    "ş": {"sh", "s"},
    "ğ": {"g", ""},
    "đ": {"d"},
    "ł": {"l", "w"},
    "ŋ": {"ng", "n"},
    "ı̇": {"i"},
    "ñ": {"ny", "n"},
    "ŧ": {"t"},
    "ħ": {"h"},
    "þ": {"th"},
    "ð": {"d", "dh"},
    "ư": {"uw", "u"},
    "ơ": {"ow", "o"},
}


def _latin_readings(char: str, rules: str | None = None) -> frozenset[str]:
    """Label spellings of one Latin character: ó → o, ç → c or s. Empty if not Latin."""
    lower = char.lower()
    folded = "".join(c for c in unicodedata.normalize("NFKD", lower) if not unicodedata.combining(c))
    readings = {folded} if len(folded) == 1 and "a" <= folded <= "z" else set()
    table = (
        _LATIN_SPELLINGS_V2 if label_rules.resolve(rules if rules is not None else LABEL_RULES).v2 else _LATIN_SPELLINGS
    )
    extra = table.get(lower, set())
    if not extra and table is _LATIN_SPELLINGS_V2:
        # Vietnamese tone-marked horn letters (ờ, ữ, ...) read like their toneless base (ơ, ư).
        base = "".join(c for c in unicodedata.normalize("NFD", lower) if c == "\u031b" or not unicodedata.combining(c))
        extra = table.get(unicodedata.normalize("NFC", base), set())
    return frozenset(readings | extra)


def _hiragana(kana: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in kana)


@functools.lru_cache(maxsize=8192)
def _kana_readings(kana: str, language: str, rules: str = "v1") -> frozenset[str]:
    """Readings of one kana unit, from the same normalizer as inference."""
    try:
        items = process_haruhi_line(
            kana, language, **({} if label_rules.resolve(rules).name == "v1" else {"rules": rules})
        )
    except (AssertionError, IndexError, KeyError, TypeError, ValueError):
        items = []
    readings = set(_KANA_ALTERNATES.get(kana, ()))
    if "".join(str(i.get("orig", "")) for i in items) == kana:
        pron = "".join(i.get("pron", "") for i in items)
        if _LABEL.fullmatch(pron):
            readings.add(pron)
    return frozenset(readings)


def _kana_unit_readings(kana: str, language: str, rules: str = "v1") -> frozenset[str]:
    """In context ずっと reads zutto: a trailing sokuon may double the next consonant."""
    readings = set(_kana_readings(kana, language, rules))
    if kana[-1] in _SOKUON:
        bases = _kana_readings(kana[:-1], language, rules) if kana[:-1] else {""}
        readings |= {base + consonant for base in bases for consonant in _CONSONANTS}
        readings.add(next(iter(bases)) if len(bases) == 1 else "")  # line-final っ: a glottal stop, no letter
    return frozenset(readings)


def _dictionary_file(name: str) -> Path:
    path = DICTIONARY_DIR / name
    if not path.is_file():
        raise RuntimeError(f"Normalizer dictionary is unavailable at {path}")
    return path


@functools.lru_cache(maxsize=1)
def _furigana_index() -> dict[str, list[tuple[str, str]]]:
    index: dict[str, list[tuple[str, str]]] = {}
    with _dictionary_file("JmdictFurigana.txt").open(encoding="utf-8-sig") as source:
        for line in source:
            parts = line.rstrip("\n").split("|")
            if len(parts) == 3:
                index.setdefault(parts[0], []).append((parts[1], parts[2]))
    return index


def _furigana_entries(word: str) -> list[tuple[str, str]]:
    """JmdictFurigana (reading, segments) entries, e.g. 強気 → (つよき, 0:つよ;1:き)."""
    return _furigana_index().get(word, [])


@functools.lru_cache(maxsize=1)
def _kanjidic_index() -> dict[str, tuple[str, ...]]:
    import gzip
    import xml.etree.ElementTree as ET

    index = {}
    with gzip.open(_dictionary_file("kanjidic2.xml.gz")) as source:
        for _, element in ET.iterparse(source):
            if element.tag != "character":
                continue
            readings = []
            for reading in element.iter("reading"):
                value = (reading.text or "").replace("-", "")
                if reading.get("r_type") == "ja_on":
                    readings.append(_hiragana(value))
                elif reading.get("r_type") == "ja_kun":
                    # つよ.い: the stem つよ is what a compound like 強気 uses.
                    stem, _, okurigana = value.partition(".")
                    readings += [stem, stem + okurigana]
            index[element.findtext("literal")] = tuple(dict.fromkeys(r for r in readings if r))
            element.clear()
    return index


def _kanji_dictionary_readings(char: str) -> tuple[str, ...]:
    """Every hiragana reading KANJIDIC2 and pykakasi list for one kanji."""
    from pykakasi.kanji import Kanwa

    entry = (Kanwa().load(char) or {}).get(char) or []
    return tuple(dict.fromkeys([*_kanjidic_index().get(char, ()), *(r for r, *_ in entry if r)]))


@functools.lru_cache(maxsize=8192)
def _kanji_readings(char: str, compound_position: bool, language: str, rules: str = "v1") -> frozenset[str]:
    """Romaji candidates for one kanji, including in-compound sound changes."""
    readings = set()
    for hiragana in _kanji_dictionary_readings(char):
        readings |= _kana_readings(hiragana, language, rules)
    variants = set(readings)
    for reading in readings:
        if compound_position:
            variants |= {
                voiced + reading[len(plain) :] for plain, voiced in _COMPOUND_ONSETS if reading.startswith(plain)
            }
        variants |= {
            reading[: -len(tail)] + consonant
            for tail in _SOKUON_TAILS
            if reading.endswith(tail) and len(reading) > len(tail)
            for consonant in _CONSONANTS
        }
    return frozenset(variants)


def _unique_split(units: list, readings: list, pron: str) -> list[tuple[int, int, str]] | None:
    """The only assignment of readings to units that spells ``pron``, or None.

    A ``None`` reading set lets that unit take any nonempty substring.
    """
    if any(r is not None and not r for r in readings):
        return None
    solutions: list[list[tuple[int, int, str]]] = []

    def search(position: int, offset: int, pieces: list) -> None:
        if len(solutions) > 1:
            return
        if position == len(units):
            if offset == len(pron):
                solutions.append(pieces)
            return
        a, b = units[position][:2]
        choices = (
            readings[position]
            if readings[position] is not None
            else [pron[offset:end] for end in range(offset + 1, len(pron) + 1)]
        )
        for reading in choices:
            if pron.startswith(reading, offset):
                search(position + 1, offset + len(reading), pieces + [(a, b, reading)])

    search(0, 0, [])
    return solutions[0] if len(solutions) == 1 else None


def _anchor_pieces(
    orig: str, pron: str, language: str, per_kanji: bool = False, rules: str = "v1"
) -> list[tuple[int, int, str]] | None:
    """Split one grouped reading at fixed anchors, e.g. 曲がり角 → ma|ga|ri|kado.

    Kana have fixed readings, Latin letters spell themselves and punctuation
    is silent, so they pin down where each kanji run's reading starts and
    ends. Consecutive kanji stay one piece unless ``per_kanji`` constrains
    each kanji to its dictionary readings: 世界 → se|kai, while 今日 (kyou)
    has no such split.
    """
    units: list[list] = []
    for index, char in enumerate(orig):
        if _is_kana(char):
            kind = "kana"
        elif _is_kanji(char):
            kind = "kanji"
        elif _latin_readings(char, rules) or _is_silent(char):
            kind = "fixed"
        else:
            return None
        # A held ー is often timed on its own, so it stays a separate unit.
        if (
            units
            and units[-1][2] == kind
            and char != "ー"
            and (char in _KANA_MODIFIERS if kind == "kana" else kind == "kanji" and not per_kanji)
        ):
            units[-1][1] = index + 1
        else:
            units.append([index, index + 1, kind])
    readings = []
    for a, b, kind in units:
        if orig[a:b] == "ー":
            # The long-vowel mark holds the preceding vowel: リー → ri + i.
            readings.append(frozenset({"", "a", "i", "u", "e", "o"}))
        elif kind == "kana":
            readings.append(_kana_unit_readings(orig[a:b], language, rules))
        elif kind == "fixed":
            readings.append(_latin_readings(orig[a], rules) or frozenset({""}))
        elif per_kanji:
            # 々 repeats the preceding kanji, usually voiced: 人々 hito+bito.
            source = orig[a - 1] if orig[a] == "々" and a and _is_kanji(orig[a - 1]) else orig[a]
            readings.append(_kanji_readings(source, a > 0, language, rules) if source != "々" else frozenset())
        else:
            readings.append(None)
    return _unique_split(units, readings, pron)


def _furigana_pieces(orig: str, pron: str, language: str, rules: str = "v1") -> list[tuple[int, int, str]] | None:
    """Split by JmdictFurigana's per-kanji readings: 木漏れ日 → ko|mo|re|bi.

    Jukujikun are listed whole (今日[きょう]), so they never split.
    """
    found = set()
    for _, segments in _furigana_entries(orig):
        units, covered = [], set()
        for segment in filter(None, segments.split(";")):
            span, _, kana = segment.partition(":")
            first, _, last = span.partition("-")
            start, end = int(first), int(last or first) + 1
            units.append([start, end, kana])
            covered.update(range(start, end))
        if any(i not in covered and not _is_kana(c) for i, c in enumerate(orig)):
            continue
        units += [[i, i + 1, c] for i, c in enumerate(orig) if i not in covered]
        units.sort()
        merged: list[list] = []
        for unit in units:
            if merged and unit[1] - unit[0] == 1 and unit[2] in _KANA_MODIFIERS and merged[-1][1] == unit[0]:
                merged[-1] = [merged[-1][0], unit[1], merged[-1][2] + unit[2]]
            else:
                merged.append(unit)
        pieces = _unique_split(merged, [_kana_unit_readings(kana, language, rules) for *_, kana in merged], pron)
        if pieces:
            found.add(tuple(pieces))
    return list(next(iter(found))) if len(found) == 1 else None


def normalize_timed_line(
    text: str, spans: list[dict], max_chars: int = 6, language: str | None = None, rules: str | None = None
) -> list[dict]:
    """Exact ordered orig mapping only; never fuzzy-match transformed surfaces.

    Never redistribute pronunciation over multiple source timing fragments.
    Items crossing a source boundary are masked unless pronunciation equals
    the lowercase source spelling, allowing exact character-offset attribution,
    or kana and dictionary kanji readings anchor a unique split of its reading.
    Timing inside a single source fragment remains a weak character estimate.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    for span in spans:
        if not all(math.isfinite(span[k]) for k in ("start", "end")) or not 0 <= span["start"] < span["end"]:
            raise NormalizationError("invalid_source_interval")
        if (
            not 0 <= span["char_start"] < span["char_end"] <= len(text)
            or text[span["char_start"] : span["char_end"]] != span["text"]
        ):
            raise NormalizationError("invalid_source_offsets")
    items = []
    rules = label_rules.resolve(rules if rules is not None else LABEL_RULES).name
    selected_language = training_language(language, rules)
    normalizer = f"echora.fa_kara.process_haruhi_line:{selected_language}+{normalizer_version(rules)}"

    def masked(selected, reason):
        return [
            {
                **s,
                "text": "",
                "source_text": s["text"],
                "display_text": text,
                "source_spans": [dict(s)],
                "ignore": True,
                "normalization_error": reason,
                "normalizer_items": items,
                "normalizer": normalizer,
            }
            for s in selected
            if s["end"] > s["start"]
        ]

    try:
        items = _normalize_line(text, selected_language, rules)
    except (AssertionError, IndexError, KeyError, TypeError, ValueError) as error:
        return masked(spans, f"normalizer_error:{error}")
    # Includes punctuation and whitespace: dropped or rewritten orig is unsafe.
    if any(not isinstance(i.get("orig"), str) for i in items) or "".join(i["orig"] for i in items) != text:
        return masked(spans, "ambiguous_surface_mapping")
    if items and not items[0]["orig"] and items[0].get("pron"):
        return masked(spans, "unattributable_empty_surface")
    # Empty surfaces continue the preceding glyph's reading. Group only for
    # source-span mapping; acoustic_tokens still exposes every mora separately.
    grouped = []
    for item in items:
        if not item.get("orig") and grouped:
            grouped[-1]["pron"] = grouped[-1].get("pron", "") + item.get("pron", "")
        else:
            grouped.append(dict(item))
    positioned = []
    cursor = 0
    for item in grouped:
        positioned.append((cursor, cursor + len(item["orig"]), item))
        cursor += len(item["orig"])
    # Split only literal spelling pronunciations at exact source offsets.
    # Transliterations/readings have no safe source-to-pronunciation offset map.
    split_positioned = []
    unsafe_boundaries = set()
    for a, b, item in positioned:
        cuts = sorted({s[key] for s in spans for key in ("char_start", "char_end") if a < s[key] < b})
        if cuts and not item.get("pron") and all(_is_silent(c) for c in item["orig"]):
            # Silent punctuation like ', ¿' may straddle two timed words.
            edges = [a, *cuts, b]
            split_positioned += [
                (left, right, {**item, "orig": text[left:right]}) for left, right in zip(edges, edges[1:])
            ]
            continue
        if cuts and item.get("pron"):
            if item["orig"].lower() == item["pron"] and len(item["orig"]) == len(item["pron"]):
                edges = [a, *cuts, b]
                for left, right in zip(edges, edges[1:]):
                    split_positioned.append(
                        (left, right, {**item, "orig": text[left:right], "pron": item["pron"][left - a : right - a]})
                    )
                continue
            # Grouped readings split safely only where anchors, a dictionary
            # word alignment or a unique combination of kanji readings fix them.
            finders = (
                ("anchor", lambda: _anchor_pieces(item["orig"], item["pron"], selected_language, rules=rules)),
                ("furigana_dictionary", lambda: _furigana_pieces(item["orig"], item["pron"], selected_language, rules)),
                (
                    "kanji_dictionary",
                    lambda: _anchor_pieces(item["orig"], item["pron"], selected_language, per_kanji=True, rules=rules),
                ),
            )
            for method, finder in finders:
                pieces = finder()
                if pieces and all(cut - a in {left for left, _, _ in pieces} for cut in cuts):
                    break
            else:
                pieces = None
            if pieces:
                edges = [a, *cuts, b]
                for left, right in zip(edges, edges[1:]):
                    split_positioned.append(
                        (
                            left,
                            right,
                            {
                                **item,
                                "orig": text[left:right],
                                "pron": "".join(r for l, _, r in pieces if left <= a + l < right),
                                "source_split": method,
                            },
                        )
                    )
                continue
            unsafe_boundaries.update(k for k, s in enumerate(spans) if a < s["char_end"] and b > s["char_start"])
        split_positioned.append((a, b, item))
    positioned = split_positioned
    result = []
    remaining = list(range(len(spans)))
    while remaining:
        group = {remaining.pop(0)}
        touched = set()
        while True:
            new_items = {
                j
                for j, (a, b, _) in enumerate(positioned)
                if any(a < spans[k]["char_end"] and b > spans[k]["char_start"] for k in group)
            }
            new_group = {
                k
                for k in remaining
                if any(
                    positioned[j][0] < spans[k]["char_end"] and positioned[j][1] > spans[k]["char_start"]
                    for j in new_items
                )
            }
            touched |= new_items
            if not new_group:
                break
            group |= new_group
            remaining = [k for k in remaining if k not in group]
        selected = [spans[k] for k in sorted(group)]
        normalized = [positioned[j][2] for j in sorted(touched)]
        covered = {c for s in selected for c in range(s["char_start"], s["char_end"])}
        reason = "unattributable_source_boundary" if group & unsafe_boundaries or len(selected) > 1 else None
        if any(
            c not in covered and not text[c].isspace()
            for j in touched
            for c in range(positioned[j][0], positioned[j][1])
        ):
            reason = "untimed_surface"
        if any(s["end"] <= s["start"] for s in selected) or any(
            abs(a["end"] - b["start"]) > 1e-6 for a, b in zip(selected, selected[1:])
        ):
            reason = "ambiguous_fragment_timing"
        for item in normalized:
            pron = item.get("pron", "")
            if pron and not _LABEL.fullmatch(pron):
                reason = "unsafe_pronunciation"
            if not pron and any(unicodedata.category(c)[0] in "LMNS" for c in item["orig"]):
                reason = "unsupported_sung_surface"
        if reason:
            result.extend(masked(selected, reason))
            continue
        pronunciation = "".join(i.get("pron", "") for i in normalized)
        start, end = selected[0]["start"], selected[-1]["end"]
        for offset in range(0, len(pronunciation), max_chars):
            right = min(offset + max_chars, len(pronunciation))
            result.append(
                {
                    "start": start + (end - start) * offset / len(pronunciation),
                    "end": start + (end - start) * right / len(pronunciation),
                    "text": pronunciation[offset:right],
                    "source_text": "".join(s["text"] for s in selected),
                    "display_text": text,
                    "source_spans": selected,
                    "normalizer_items": normalized,
                    "normalizer": normalizer,
                    "timing_method": "weak_characters_within_source_fragment",
                    "timing_quality": "weak",
                    "measured_syllable_boundaries": False,
                }
            )
    return sorted(result, key=lambda unit: unit["start"])
