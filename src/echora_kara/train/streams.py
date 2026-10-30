"""Assign each timed word to one of two CTC output streams (SURT/HEAT-style).

Multi-talker ASR keeps one ordinary, in-order token sequence per voice (SURT, arXiv 2011.13148,
assigns utterances to output channels by start order). Here every word goes to stream 1 unless it
starts while a stream-1 word is still being sung (for at least ``min_overlap`` seconds): then it goes
to stream 2. Each stream is then labelled and trained like the single-stream model, so frames where two
words are sung at once are supervised in both streams instead of ignored.

A word that would need a third stream (two others still sung) is masked out.
"""

from __future__ import annotations

MIN_OVERLAP = 0.1  # seconds; shorter overlaps are treated as source timing noise


def assign_streams(tokens: list[dict], min_overlap: float = MIN_OVERLAP) -> tuple[list[dict], list[dict]]:
    """Split window tokens into (stream 1, stream 2) token lists.

    Explicit masks (``ignore``) go to both streams. ``soft_ignore`` (background under the lead, which
    a single stream cannot supervise) is dropped: the overlapping word now has its own stream.
    """
    masks = [t for t in tokens if t.get("ignore") or not t.get("ids")]
    words = sorted((t for t in tokens if not t.get("ignore") and t.get("ids")), key=lambda t: (t["start"], t["end"]))
    first, second, third = [], [], []
    first_end = second_end = float("-inf")
    for token in words:
        clean = {k: v for k, v in token.items() if k != "soft_ignore"}
        if token["start"] <= first_end - min_overlap:
            if token["start"] <= second_end - min_overlap:
                third.append({**clean, "ids": [], "ignore": True})
                continue
            second.append(clean)
            second_end = max(second_end, token["end"])
        else:
            first.append(clean)
            first_end = max(first_end, token["end"])
    return [*first, *masks, *third], [*second, *masks, *third]
