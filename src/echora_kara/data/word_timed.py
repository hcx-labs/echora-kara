"""Timed label units from word-timed TTML lyrics, for the karaoke manifest builder.

Each complete lyric line (``<p>``) is normalized through the same path the aligner uses at
inference; its acoustic labels are split proportionally into pieces of at most six characters
inside each timed source fragment (``<span begin end>``). Spans with role ``x-bg`` are background
vocals. Elements and attributes are matched by local name, whatever namespace the document uses.
"""
from __future__ import annotations

import math
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter

from echora_kara.text.normalization import normalize_timed_line

MAX_LABEL_CHARS = 6
# Background vocals overlapping lead singing by more than this share are not
# supervised: a frame holds one label, and the lead keeps it.
MAX_BACKGROUND_OVERLAP = 0.1


def tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def timing_attribute(root: ET.Element) -> str | None:
    """The root ``timing`` attribute ("Word", "Line", ...) of word-timed lyric TTML, in any namespace."""
    return next((value for key, value in root.attrib.items() if key.rsplit("}", 1)[-1] == "timing"), None)


def timing_granularity(ttml: str) -> str | None:
    """Lower-cased root ``timing`` attribute of a document, or None."""
    try:
        value = timing_attribute(ET.fromstring(ttml))
    except ET.ParseError:
        return None
    return value.lower() if value else None


TTML_AUDIT: Counter[str] = Counter()


def parse_time(value: str | None) -> float | None:
    """Media seconds; legacy bare seconds are intentionally supported.

    Frames, ticks, signed offsets and non-finite values are not supported.
    """
    if value is None:
        return None
    value = value.strip()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(ms|s|m|h)?", value)
    if match:
        seconds = float(match[1]) * {None: 1, "ms": .001, "s": 1, "m": 60, "h": 3600}[match[2]]
    else:
        match = re.fullmatch(r"(?:(\d+):)?(\d+):(\d{2}(?:\.\d+)?)", value)
        if not match or (match[1] is not None and float(match[2]) >= 60) or float(match[3]) >= 60:
            return None
        seconds = float(match[1] or 0) * 3600 + float(match[2]) * 60 + float(match[3])
    return seconds if math.isfinite(seconds) else None


def split_unit(text: str, start: float, end: float) -> list[dict]:
    """Compatibility helper; production paths normalize complete lines."""
    return normalize_timed_line(text, [{"text": text, "start": start, "end": end,
                                      "char_start": 0, "char_end": len(text)}], MAX_LABEL_CHARS)


def _rules_kwargs(rules: str | None) -> dict:
    """Pass ``rules`` only when it is not the default, so v1 keeps the historical call shape."""
    return {} if rules in (None, "v1") else {"rules": rules}


def background_units(text: str, spans: list[dict], line_text: str, language: str | None = None,
                     rules: str | None = None) -> list[dict]:
    """Normalize one line's background vocals on their own, like a lead line."""
    if not spans:
        return []
    TTML_AUDIT["background_vocal"] += len(spans)
    return [{**unit, "vocal_role": "background", "line_text": line_text}
            for unit in normalize_timed_line(text, spans, MAX_LABEL_CHARS, language=language, **_rules_kwargs(rules))]


def finish_lines(lines: list[list[dict]], policy: str) -> list[list[dict]]:
    """v1: per-word background masking (resolve_background). v2: only label_policy.tag (parenthesised
    echoes → background); the audio-dependent steps (tail trim, lead overlaps, background-head targets,
    per-phrase main-head decision) run per row in build_manifest via label_policy.finalize."""
    if policy == "v1":
        return resolve_background(lines)
    from echora_kara.data import label_policy as label_policy_v2
    lines, stats = label_policy_v2.tag(lines)
    for key, value in stats.items():
        TTML_AUDIT[key] += value
    return lines


def resolve_background(lines: list[list[dict]]) -> list[list[dict]]:
    """Supervise standalone background vocals; let the lead win where they overlap.

    Lyric text at inference usually includes background parts in parentheses, so
    inference aligns them. Standalone backing vocals (echoes, call and
    response) keep their letters. Overlapped ones become soft masks: frames
    the lead labels stay supervised, only background-only frames are ignored.
    """
    lead = [(u["start"], u["end"]) for line in lines for u in line
            if not u.get("ignore") and u.get("vocal_role") != "background"]
    fragments: dict[tuple[float, float], list[dict]] = {}
    for line in lines:
        for unit in line:
            if unit.get("vocal_role") == "background":
                span = (unit.get("source_spans") or [unit])[0]
                fragments.setdefault((span["start"], span["end"]), []).append(unit)
    for (start, end), members in fragments.items():
        overlap = sum(max(0.0, min(end, e) - max(start, s)) for s, e in lead)
        overlapped = overlap > MAX_BACKGROUND_OVERLAP * (end - start)
        TTML_AUDIT["background_overlapping_lead" if overlapped else "background_standalone"] += 1
        for unit in members:
            if overlapped and not unit.get("ignore"):
                unit.update(source_text=unit.get("source_text") or unit["text"], text="", ignore=True,
                            normalization_error="overlapping_background_vocal")
                unit.pop("timing_method", None)
            if unit.get("ignore"):
                unit["soft_ignore"] = True
    return lines


def units_from_ttml(ttml: str, language: str | None = None, rules: str | None = None,
                    policy: str = "v1") -> list[list[dict]]:
    """Timed label units per paragraph. ``language`` is the normalizer hint (None = auto); ``rules``
    the label rules (None = the adapter default, v1); ``policy`` the label-construction policy."""
    try:
        root = ET.fromstring(ttml)
    except ET.ParseError:
        TTML_AUDIT["malformed_xml"] += 1
        return []

    # Reject the document rather than silently relabel unsupported timing.
    intervals = {}
    inherited_properties = {}
    # Word-timed lyric TTML (marked by a root ``timing`` attribute) uses absolute media clocks at
    # every nested level; other TTML keeps parent-relative offsets. Select by document metadata,
    # never by a heuristic comparing numerical timestamp values.
    absolute = timing_attribute(root) is not None
    try:
        def resolve(element, parent_start=0.0, parent_end=None, parent_background=False, parent_preserve=False):
            attrs = {key.rsplit("}", 1)[-1]: value for key, value in element.attrib.items()}
            if attrs.get("timeBase", "media") != "media" or attrs.get("timeContainer", "par") != "par":
                raise ValueError("unsupported_timing_semantics")
            if any(key in attrs for key in ("frameRate", "frameRateMultiplier", "subFrameRate", "tickRate")):
                raise ValueError("unsupported_frames_or_ticks")
            values = {}
            for key in ("begin", "end", "dur"):
                if key in attrs:
                    values[key] = parse_time(attrs[key])
                    if values[key] is None:
                        raise ValueError("invalid_or_unsupported_time")
            origin = 0.0 if absolute else parent_start
            start = origin + values["begin"] if "begin" in values else parent_start
            ends = [v for v in (parent_end,
                    origin + values["end"] if "end" in values else None,
                    start + values["dur"] if "dur" in values else None) if v is not None]
            end = min(ends) if ends else None
            start = max(parent_start, start)
            if not math.isfinite(start) or (end is not None and (not math.isfinite(end) or end <= start)):
                raise ValueError("invalid_interval")
            intervals[element] = (start, end, 'begin' in values and any(k in values for k in ('end', 'dur')))
            background = parent_background or "x-bg" in attrs.get("role", "").split()
            preserve = element.get("{http://www.w3.org/XML/1998/namespace}space",
                                   "preserve" if parent_preserve else "default") == "preserve"
            inherited_properties[element] = (background, preserve)
            for child in element:
                resolve(child, start, end, background, preserve)
        resolve(root)
    except (ValueError, OverflowError) as error:
        TTML_AUDIT[str(error) if isinstance(error, ValueError) else "invalid_or_unsupported_time"] += 1
        return []

    lines = []
    for paragraph in root.iter():
        if tag(paragraph) != "p":
            continue
        chunks, spans, masks = [], [], []
        background_chunks, background_spans = [], []
        cursor = 0
        unlocatable_background = False
        background_gap = False
        background_space = False  # v2: whitespace seen between two background words

        def walk(element, localized=False, background=False, preserve=False):
            nonlocal cursor
            start, end, explicit = intervals[element]
            localized = localized or (element is not paragraph and explicit)
            role = " ".join(v for k, v in element.attrib.items() if k.rsplit("}", 1)[-1] == "role")
            background = background or "x-bg" in role.split()
            preserve = element.get("{http://www.w3.org/XML/1998/namespace}space",
                                   "preserve" if preserve else "default") == "preserve"

            def append(raw):
                nonlocal cursor, unlocatable_background, background_gap, background_space
                text = raw or ""
                if policy != "v1" and text and not text.strip() and background_chunks and not background_chunks[-1].endswith(" "):
                    # v2: whitespace between background words (an untimed container, or x-bg words
                    # under the paragraph) separates them: "(Wait for you)", not "(Waitforyou)".
                    background_space = True
                if not preserve:
                    text = re.sub(r"[ \t\r\n]+", " ", text)
                    if chunks and chunks[-1].endswith(" "):
                        text = text.lstrip(" ")
                if not text:
                    return
                if background and end is None and text.strip():
                    unlocatable_background = True
                if background and localized:
                    # Separate background phrases keep a word boundary between them.
                    if (background_gap or background_space) and background_chunks and not background_chunks[-1].endswith(" "):
                        background_chunks.append(" ")
                    background_gap = background_space = False
                    offset = sum(map(len, background_chunks))
                    if end is not None and text.strip():
                        background_spans.append({"text": text, "raw_text": raw, "start": start, "end": end,
                                                 "char_start": offset, "char_end": offset + len(text)})
                    background_chunks.append(text)
                elif text.strip():
                    background_gap = True
                if end is not None and text.strip() and not (background and localized):
                    span = {"text": text, "raw_text": raw, "start": start, "end": end,
                            "char_start": cursor, "char_end": cursor + len(text)}
                    if background or not localized:
                        reason = "background_vocal" if background else "unlocalized_paragraph_text"
                        masks.append({**span, "source_text": text, "text": "", "ignore": True,
                                      "normalization_error": reason})
                        TTML_AUDIT[reason] += 1
                    else:
                        spans.append(span)
                # Background surface remains a boundary, but cannot be normalized as lead lyrics.
                chunks.append(" " * len(text) if background else text)
                cursor += len(text)

            append(element.text)
            for child in element:
                if tag(child) == "br":
                    chunks.append("\n")
                    cursor += 1
                else:
                    walk(child, localized, background, preserve)
                append(child.tail)
        background, preserve = inherited_properties[paragraph]
        walk(paragraph, background=background, preserve=preserve)
        text = "".join(chunks)
        if unlocatable_background or (not spans and not masks and not background_spans and any(
                unicodedata.category(ch)[0] in "LMNS" for ch in text)):
            TTML_AUDIT["unlocatable_sung_paragraph"] += 1
            return []
        covered = {c for span in spans for c in range(span["char_start"], span["char_end"])}
        if any(c not in covered and unicodedata.category(ch)[0] in "LMNS" for c, ch in enumerate(text)):
            units = [{**span, "source_text": span["text"], "text": "", "ignore": True,
                      "display_text": text, "normalization_error": "untimed_line_surface"} for span in spans]
        else:
            units = normalize_timed_line(text, spans, MAX_LABEL_CHARS, language=language, **_rules_kwargs(rules)) if spans else []
        units.extend({**mask, "display_text": text} for mask in masks)
        units.extend(background_units("".join(background_chunks), background_spans, text, language, rules))
        if units:
            lines.append(sorted(units, key=lambda unit: unit["start"]))
    return finish_lines(lines, policy)


def timing_lines(ttml: str, language: str | None = None, rules: str | None = None,
                 policy: str = "v1") -> list[list[dict]]:
    """``language``: normalizer hint. ``rules``: label rules (None = adapter default). ``policy``:
    "v1" (per-word background masking) or "v2" (label_policy)."""
    return units_from_ttml(ttml, language, rules, policy)
