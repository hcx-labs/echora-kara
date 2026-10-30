"""Label-construction policy v2 for the karaoke manifest (schema 7): background phrases, parenthesised
echoes, lead-lead overlaps, the tail trim and repeat metadata.

All functions take the per-line unit lists ``word_timed.timing_lines`` produces and return
new lists. Two stages:

``tag`` (cached with the normalised labels, no audio):
1. ``parenthesised_lead_to_background``: a parenthesised run of lead words that the document did not
   tag ``x-bg`` becomes background when it starts at or before the end of the
   preceding lead fragment (an echo sung straight after or under the lead). One with clear air before it
   stays lead (an ad-lib line).

``finalize`` (per row in build_manifest, with the song's stem activity when there is one):
2. Tail trim (tail_trim.py): a fragment's tail where the vocal stem is quiet becomes blank. Only the
   stem: the VAD-head estimate on the mix is not reliable enough for trimming.
3. ``trim_lead_overlaps``: a small share of lead fragments overlap the next one, mostly a line-final word
   whose ``end`` runs past the next line's ``begin``; the earlier fragment ends at the later onset when
   the overlap is at most half of it (≤ 2 s); larger overlaps (duets) stay and record ``overlap``.
4. ``background_targets``: every supervised background unit, whole phrase, with its (trimmed) source
   timing, for the background head (overlap with the lead allowed; ``lead_overlap`` recorded).
5. ``resolve_background_phrases``: the main head decides per *phrase* (never per word) against the
   trimmed lead word fragments: ``sequential`` (lead under ≤ 10 % of it: a same-singer echo or ad-lib)
   → supervised on the main head like lead words (and on the background head); ``simultaneous`` →
   soft-masked on the main head (the lead underneath stays supervised), supervised on the background
   head only. v1 decided per word and truncated phrases ("(Wait for you)" kept "wait", "for").

``repeat_metadata``: ``line_repeat[i]`` = how many immediately preceding lines have the same transcript,
plus the gap to the previous line, for a repeat-aware decoder prior / harness slicing — no label change.
"""
from __future__ import annotations

from collections import defaultdict

from echora_kara.data.tail_trim import apply_trims, fragment_key

POLICY_VERSION = "label_policy_v2"
PHRASE_GAP = 0.25           # background words closer than this belong to one phrase
MAX_PHRASE_OVERLAP = 0.10   # share of a background phrase under lead that still keeps it supervised
MAX_TRIM_SHARE = 0.5        # a fragment's end moves to the next onset only for overlaps up to this share
MAX_TRIM_SECONDS = 2.0
ECHO_SLACK = 0.05           # a parenthesised phrase starting within this of the previous lead end "abuts"
PARENS = {"(": ")", "[": "]", "（": "）"}


def _is_lead(unit: dict) -> bool:
    return not unit.get("ignore") and unit.get("vocal_role") != "background" and bool(unit.get("text"))


def _fragments(units: list[dict]) -> dict[tuple[float, float], list[dict]]:
    groups: dict[tuple[float, float], list[dict]] = defaultdict(list)
    for unit in units:
        key = fragment_key(unit)
        if key is not None:
            groups[key].append(unit)
    return groups


# ---------------------------------------------------------------------------------------------
# 1. Parenthesised lead phrases

def _paren_phrases(line: list[dict]) -> list[list[tuple[float, float]]]:
    """Fragment keys of each parenthesised run of lead words in the line, in order."""
    frags = sorted(_fragments([u for u in line if _is_lead(u)]).items())
    phrases, current, closing = [], [], None
    for key, members in frags:
        text = (members[0].get("source_text") or "").strip()
        if closing is None:
            for opener, close in PARENS.items():
                if text.startswith(opener):
                    closing, current = close, [key]
                    break
            else:
                continue
            if text.rstrip(".,!?").endswith(closing) and len(text) > 1:
                phrases.append(current)
                current, closing = [], None
        else:
            current.append(key)
            if text.rstrip(".,!?").endswith(closing):
                phrases.append(current)
                current, closing = [], None
    return phrases


def parenthesised_lead_to_background(lines: list[list[dict]]) -> tuple[list[list[dict]], dict]:
    stats = {"paren_phrases": 0, "paren_phrases_to_background": 0}
    lead_ends = sorted((key[1], key[0]) for line in lines for key in _fragments([u for u in line if _is_lead(u)]))
    for line in lines:
        for phrase in _paren_phrases(line):
            stats["paren_phrases"] += 1
            start = phrase[0][0]
            previous_end = max((end for end, begin in lead_ends if begin < start and end > 0 and (begin, end) not in phrase), default=None)
            if previous_end is None or start > previous_end + ECHO_SLACK:
                continue
            stats["paren_phrases_to_background"] += 1
            keys = set(phrase)
            for unit in line:
                if fragment_key(unit) in keys and unit.get("vocal_role") != "background":
                    unit["vocal_role"] = "background"
                    unit["background_source"] = "parenthesised_echo"
    return lines, stats


# ---------------------------------------------------------------------------------------------
# 2. Phrase-level background resolution

def _phrases_of(line: list[dict]) -> list[list[tuple[float, float]]]:
    keys = sorted(_fragments([u for u in line if u.get("vocal_role") == "background" and u.get("text") and not u.get("ignore")]))
    phrases: list[list[tuple[float, float]]] = []
    for key in keys:
        if phrases and key[0] - phrases[-1][-1][1] < PHRASE_GAP:
            phrases[-1].append(key)
        else:
            phrases.append([key])
    return phrases




def _mask(unit: dict, reason: str) -> None:
    if unit.get("ignore"):
        return
    unit.update(source_text=unit.get("source_text") or unit["text"], text="", ignore=True, soft_ignore=True,
                normalization_error=reason)
    unit.pop("timing_method", None)


def resolve_background_phrases(lines: list[list[dict]]) -> tuple[list[list[dict]], dict]:
    """Main-head decision per background phrase, never per word.

    Simultaneity is measured against the lead *word fragments* (after tail trim and lead-overlap
    trimming), not the line span: documents often time an echo sung by the same singer straight after
    the line ("I'm so lonely (so lonely)") as overlapping the line.

    - ``sequential``: the lead overlaps at most 10 % of the phrase → the whole phrase is supervised on
      the main head like lead words (a word running into a lead onset ends at that onset, so no frame
      is labelled twice), and it is also a background-head target.
    - ``simultaneous``: more overlap → the whole phrase is soft-masked on the main head
      (``overlapping_background_phrase``; the lead underneath stays supervised) and supervised on the
      background head only.
    Every unit records ``phrase`` and ``background_category``."""
    stats = {"background_phrases": 0, "background_phrases_sequential": 0, "background_phrases_simultaneous": 0,
             "background_words_trimmed": 0, "background_units_sequential": 0, "background_units_simultaneous": 0}
    lead = sorted(key for line in lines for key in _fragments([u for u in line if _is_lead(u)]))
    phrase_index = 0
    trims: dict[tuple[float, float], float] = {}
    for line in lines:
        for phrase in _phrases_of(line):
            phrase_index += 1
            stats["background_phrases"] += 1
            start, end = phrase[0][0], phrase[-1][1]
            keys = set(phrase)
            units = [u for u in line if fragment_key(u) in keys]
            overlap = sum(max(0.0, min(end, e) - max(start, s)) for s, e in lead)
            sequential = overlap <= MAX_PHRASE_OVERLAP * (end - start)
            category = "sequential" if sequential else "simultaneous"
            for unit in units:
                unit["phrase"] = phrase_index
                unit["background_category"] = category
            stats[f"background_phrases_{category}"] += 1
            stats[f"background_units_{category}"] += len(units)
            if sequential:
                for key in phrase:  # a word running into a lead onset ends there
                    onset = next((s for s, e in lead if key[0] < s < key[1]), None)
                    if onset is not None and key[1] - onset <= MAX_TRIM_SHARE * (key[1] - key[0]):
                        trims[key] = round(onset, 3)
                        stats["background_words_trimmed"] += 1
            else:
                for unit in units:
                    _mask(unit, "overlapping_background_phrase")
    if trims:
        lines, _ = apply_trims(lines, trims, "blank", "lead_onset")
    return lines, stats


# ---------------------------------------------------------------------------------------------
# 3. Lead-lead overlaps

def trim_lead_overlaps(lines: list[list[dict]]) -> tuple[list[list[dict]], dict]:
    stats = {"lead_overlaps": 0, "lead_overlaps_trimmed": 0, "lead_overlap_seconds": 0.0, "lead_overlaps_kept": 0}
    line_of: dict[tuple[float, float], int] = {}
    for index, line in enumerate(lines):
        for key in _fragments([u for u in line if _is_lead(u)]):
            line_of.setdefault(key, index)
    keys = sorted(line_of)
    trims: dict[tuple[float, float], float] = {}
    overlaps: dict[tuple[float, float], dict] = {}
    for previous, current in zip(keys, keys[1:]):
        overlap = previous[1] - current[0]
        if overlap <= 1e-6:
            continue
        stats["lead_overlaps"] += 1
        stats["lead_overlap_seconds"] += overlap
        duration = previous[1] - previous[0]
        trimmed = overlap <= MAX_TRIM_SHARE * duration and overlap <= MAX_TRIM_SECONDS and current[0] > previous[0]
        record = {"seconds": round(overlap, 3), "with_line": line_of[current], "resolution": "trimmed_to_next_onset" if trimmed else "kept"}
        overlaps[previous] = record
        overlaps.setdefault(current, {"seconds": round(overlap, 3), "with_line": line_of[previous],
                                      "resolution": "onset_kept" if trimmed else "kept"})
        if trimmed:
            trims[previous] = round(current[0], 3)
            stats["lead_overlaps_trimmed"] += 1
        else:
            stats["lead_overlaps_kept"] += 1
    for line in lines:
        for unit in line:
            key = fragment_key(unit)
            if key in overlaps:
                unit["overlap"] = overlaps[key]
    if trims:
        lines, _ = apply_trims(lines, trims, "blank", "next_onset")
    stats["lead_overlap_seconds"] = round(stats["lead_overlap_seconds"], 3)
    return lines, stats


# ---------------------------------------------------------------------------------------------
# 4. Repeats

def repeat_metadata(lines: list[list[dict]]) -> list[dict]:
    """Per line: transcript repeat count against the previous lines and the gap to the previous line."""
    transcripts = ["".join(u.get("text") or "" for u in line if u.get("vocal_role") != "background") for line in lines]
    bounds = []
    for line in lines:
        sung = [u for u in line if u.get("text") and not u.get("ignore")]
        bounds.append((min(u["start"] for u in sung), max(u["end"] for u in sung)) if sung else None)
    result = []
    for index, transcript in enumerate(transcripts):
        repeat = 0
        back = index - 1
        while back >= 0 and transcript and transcripts[back] == transcript:
            repeat += 1
            back -= 1
        gap = None
        if index and bounds[index] and bounds[index - 1]:
            gap = round(bounds[index][0] - bounds[index - 1][1], 3)
        result.append({"repeat_of_previous": repeat, "gap_to_previous": gap})
    return result


def background_targets(lines: list[list[dict]]) -> list[dict]:
    """Copies of the supervised background units (with their line index) for the background head."""
    lead = sorted(key for line in lines for key in _fragments([u for u in line if _is_lead(u)]))
    targets = []
    for index, line in enumerate(lines):
        for unit in line:
            if unit.get("vocal_role") != "background" or unit.get("ignore") or not unit.get("text"):
                continue
            key = fragment_key(unit)
            overlap = sum(max(0.0, min(key[1], e) - max(key[0], s)) for s, e in lead) if key else 0.0
            targets.append({**{k: v for k, v in unit.items() if k not in ("normalizer_items", "display_text", "normalizer")},
                            "line": index, "lead_overlap": round(overlap, 3)})
    return targets


def tag(lines: list[list[dict]]) -> tuple[list[list[dict]], dict]:
    """Cached step (no audio): parenthesised echoes become background (step 1)."""
    return parenthesised_lead_to_background(lines)


def finalize(lines: list[list[dict]], activity: dict | None = None) -> tuple[list[list[dict]], list[dict], dict]:
    """Row step, after ``tag``: (main-head lines, background-head targets, stats).

    1. Tail trim of lead and background fragments where the vocal stem is quiet (tail_trim.py;
       only stem activity: the VAD-head estimate on the mix is not reliable enough for trimming);
       trimmed frames are blank targets.
    2. Lead-lead overlaps: the earlier fragment ends at the later onset (``trim_lead_overlaps``).
    3. Background-head targets: every supervised background unit, whole, with its (trimmed) timing.
    4. Main-head decision per background phrase against the trimmed lead fragments.
    """
    from echora_kara.data import tail_trim
    stats: dict = {"activity": None}
    if activity is not None and activity.get("method") == "stem_energy":
        units = [u for line in lines for u in line]
        trims = tail_trim.plan_trims(units, activity["activity"])
        lines, s = tail_trim.apply_trims(lines, trims, "blank", activity["method"])
        stats.update(activity="stem_energy", tail_fragments_trimmed=s["fragments_trimmed"], tail_seconds_trimmed=s["seconds_trimmed"])
    lines, s = trim_lead_overlaps(lines)
    stats.update(s)
    background = background_targets(lines)
    lines, s = resolve_background_phrases(lines)
    stats.update(s)
    category = {}
    for line in lines:
        for unit in line:
            if unit.get("background_category"):
                category[fragment_key(unit)] = unit["background_category"]
    for unit in background:
        unit["background_category"] = category.get(fragment_key(unit), "sequential")
    for line in lines:
        for unit in line:
            if unit.get("ignore") and unit.get("vocal_role") == "background":
                unit["soft_ignore"] = True
    return lines, background, stats


def apply(lines: list[list[dict]], background: list | None = None) -> tuple[list[list[dict]], dict]:
    """All steps without audio (tests, tools): ``tag`` then ``finalize`` without activity."""
    lines, stats = tag(lines)
    lines, targets, s = finalize(lines)
    stats.update(s)
    if background is not None:
        background.extend(targets)
    return lines, stats
