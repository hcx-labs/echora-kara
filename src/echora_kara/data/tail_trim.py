"""Tail trim: shorten a timed fragment where the singer is not active (label schema 7).

In word-timed lyrics a word's ``end`` is usually the next word's ``begin`` (most within-line
fragments abut, and many line transitions abut or overlap the next line), so a blank-free region loss
grades every pause as the previous fragment's last letter. With a vocal-activity array
(``vocal_activity.py``: the stem's loudness, or the VAD head's estimate on the mix) each
fragment's tail is cut back to the last active frame plus a margin, never below the frames its
letters need nor below ``min_keep`` of its length. Trimmed frames become blank targets when the
activity came from a stem (``label: "blank"``) and *ignore* masks when it is an estimate
(``label: "ignore"``). The original source end is kept as ``provider_end`` on every unit.

Fragments are identified by their measured source span (start, end); a trim moves the span's end,
and the weak proportional per-character timings inside the fragment are rescaled to the new span,
so ``integrity.check_row`` still sees every unit inside its single source fragment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FRAME = 0.02


@dataclass(frozen=True)
class TrimConfig:
    threshold: float = 0.35      # scaled activity below this is "not singing"
    min_inactive_frames: int = 3  # at least 60 ms of inactive tail before anything is trimmed
    margin_frames: int = 1        # frames kept after the last active one (decay)
    min_keep: float = 0.4         # never cut a fragment below this share of its length
    min_frames_per_letter: int = 1


def fragment_key(unit: dict) -> tuple[float, float] | None:
    spans = unit.get("source_spans") or []
    if len(spans) != 1:
        return None
    return (round(float(spans[0]["start"]), 4), round(float(spans[0]["end"]), 4))


def plan_trims(units: list[dict], activity: np.ndarray, config: TrimConfig = TrimConfig()) -> dict[tuple[float, float], float]:
    """{(span start, span end): new end} for every supervised fragment whose tail is inactive."""
    letters: dict[tuple[float, float], int] = {}
    for unit in units:
        if unit.get("ignore") or not unit.get("text"):
            continue
        key = fragment_key(unit)
        if key is not None:
            letters[key] = letters.get(key, 0) + len(unit["text"])
    frames = len(activity)
    trims: dict[tuple[float, float], float] = {}
    for (start, end), count in letters.items():
        first, stop = int(round(start / FRAME)), int(round(end / FRAME))
        first, stop = max(0, first), min(frames, stop)
        if stop - first < 2:
            continue
        window = activity[first:stop]
        active = np.flatnonzero(window >= config.threshold)
        last_active = int(active[-1]) if len(active) else -1
        inactive_tail = (stop - first - 1) - last_active
        if inactive_tail < config.min_inactive_frames:
            continue
        new_stop = first + last_active + 1 + config.margin_frames
        floor = first + max(int(np.ceil((stop - first) * config.min_keep)), count * config.min_frames_per_letter + 1)
        new_stop = max(new_stop, floor)
        if new_stop >= stop:
            continue
        trims[(start, end)] = round(new_stop * FRAME, 3)
    return trims


def apply_trims(lines: list[list[dict]], trims: dict[tuple[float, float], float], label: str,
                method: str) -> tuple[list[list[dict]], dict]:
    """Shorten the fragments in ``trims`` (``label``: "blank" or "ignore"); returns (lines, stats)."""
    if label not in ("blank", "ignore"):
        raise ValueError("label must be blank or ignore")
    stats = {"fragments_trimmed": 0, "seconds_trimmed": 0.0}
    result = []
    for line in lines:
        out, masks = [], []
        done: set[tuple[float, float]] = set()
        for unit in line:
            key = fragment_key(unit)
            if key is None or key not in trims:
                out.append(unit)
                continue
            old_start, old_end = key
            new_end = trims[key]
            span = dict(unit["source_spans"][0])
            span["provider_end"] = span["end"]
            span["end"] = new_end
            scale = (new_end - old_start) / (old_end - old_start)
            changed = {**unit, "source_spans": [span], "provider_end": unit["end"],
                       "tail_trim": {"method": method, "seconds": round(old_end - new_end, 3), "label": label}}
            if not unit.get("ignore"):
                changed["start"] = round(old_start + (unit["start"] - old_start) * scale, 6)
                changed["end"] = round(old_start + (unit["end"] - old_start) * scale, 6)
                if changed["end"] <= changed["start"]:
                    changed["end"] = min(new_end, changed["start"] + FRAME)
            else:
                changed["end"] = min(unit["end"], new_end) if unit["end"] > new_end else unit["end"]
                if changed["end"] <= changed["start"]:
                    out.append(unit)
                    continue
            out.append(changed)
            if key not in done:
                done.add(key)
                stats["fragments_trimmed"] += 1
                stats["seconds_trimmed"] += old_end - new_end
                if label == "ignore":
                    masks.append({"start": new_end, "end": old_end, "text": "", "source_text": unit.get("source_text", ""),
                                  "ignore": True, "soft_ignore": True, "normalization_error": "trimmed_tail_estimate",
                                  "vocal_role": unit.get("vocal_role", "lead"), "tail_trim": changed["tail_trim"],
                                  "source_spans": [{"start": new_end, "end": old_end, "text": span.get("text", "")}]})
        result.append(sorted(out + masks, key=lambda u: (u["start"], u["end"])))
    stats["seconds_trimmed"] = round(stats["seconds_trimmed"], 3)
    return result, stats
