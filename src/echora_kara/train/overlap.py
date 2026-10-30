"""Multi-label frame targets for the overlap head.

The softmax frame labels give every frame one letter: where two words overlap, the frame is ignored
(or the background word is). The overlap head instead has one independent (sigmoid) output per
letter, so a frame can carry the letters of every word sung there. Targets come from the same
measured token intervals and the same frame clock as ``targets``.

Within a word, the letter at a frame is the same weak proportional estimate the softmax labels use, so the
neighbouring letters' outputs are not graded on that frame. One frame at each token edge is not
graded either, matching the softmax labels' boundary tolerance.
"""

from __future__ import annotations

import math

import numpy as np

from echora_kara.train.targets import FRAME_OFFSET_SECONDS, FRAME_STRIDE_SECONDS, PAD_TOKEN_ID

LETTERS = tuple(range(1, 27))
CHANNELS = (*LETTERS, PAD_TOKEN_ID)  # graded outputs; any other vocabulary entry is never graded


def frame_span(start: float, end: float, frames: int) -> range:
    """Frames (clipped to ``[0, frames)``) whose receptive-field centers fall in ``[start, end)`` seconds."""
    first = max(0, math.ceil((start - FRAME_OFFSET_SECONDS) / FRAME_STRIDE_SECONDS - 1e-9))
    stop = min(frames, math.ceil((end - FRAME_OFFSET_SECONDS) / FRAME_STRIDE_SECONDS - 1e-9))
    return range(first, stop)


def multilabel_targets(tokens: list[dict], frames: int, vocab_size: int, edge_tolerance: int = 1):
    """Returns (targets float32 [frames, vocab], graded bool [frames, vocab], overlap bool [frames]).

    ``overlap`` marks frames where two or more tokens are sung at once. Tokens marked ``ignore`` mask
    their frames entirely; ``soft_ignore`` (background under the lead) is supervised here, which is
    the point of this head.
    """
    targets = np.zeros((frames, vocab_size), dtype=np.float32)
    graded = np.zeros((frames, vocab_size), dtype=bool)
    graded[:, list(CHANNELS)] = True
    active = np.zeros(frames, dtype=np.int16)
    masked = np.zeros(frames, dtype=bool)
    ungraded: list[tuple[int, int]] = []
    for token in tokens:
        start, end = float(token["start"]), float(token["end"])
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        span = frame_span(start, end, frames)
        if token.get("ignore") or not token.get("ids"):
            masked[span.start : span.stop] = True
            continue
        ids = token["ids"]
        active[span.start : span.stop] += 1
        for frame in span:
            center = FRAME_OFFSET_SECONDS + frame * FRAME_STRIDE_SECONDS
            index = min(len(ids) - 1, max(0, int((center - start) * len(ids) / (end - start))))
            targets[frame, ids[index]] = 1.0
            # The split inside a word is a guess: do not grade the neighbouring letters here.
            for neighbour in (index - 1, index + 1):
                if 0 <= neighbour < len(ids) and ids[neighbour] != ids[index]:
                    ungraded.append((frame, ids[neighbour]))
        for frame in [*range(span.start - edge_tolerance, span.start), *range(span.stop, span.stop + edge_tolerance)]:
            if 0 <= frame < frames:
                ungraded.extend((frame, letter) for letter in set(ids))
                ungraded.append((frame, PAD_TOKEN_ID))
    for frame, channel in ungraded:
        if targets[frame, channel] == 0:
            graded[frame, channel] = False
    letters = list(LETTERS)
    silent = targets[:, letters].sum(axis=1) == 0
    targets[silent, PAD_TOKEN_ID] = 1.0
    graded[masked] = False
    return targets, graded, active >= 2


def aligned_overlap_targets(
    tokens: list[dict], frames: int, vocab_size: int, edge_tolerance: int = 1, overlap_token_weight: float = 4.0
):
    """Aligned targets: each word is aligned in order over its own frames, no guessed letter split.

    Returns a dict:
    ``regions``  [(first, stop, ids, weight)]: every word whose frames can hold its letters. The loss
                 sums over every in-order, blank-free filling of those frames by the letters, so the
                 split inside the word is learned, not guessed. Words that overlap another word
                 weigh ``overlap_token_weight``: they are the rare cases this head exists for.
    ``negative`` bool [frames, vocab]: letters to push down, those of no word sung at the frame
                 (within ``edge_tolerance`` frames). A sung word's other letters are never pushed
                 down, so two words at once are never contradicted.
    ``silence``, ``silence_graded`` [frames]: the blank output, on where nothing is sung.
    ``overlap`` bool [frames]: two or more words sung at once.
    """
    allowed = np.zeros((frames, vocab_size), dtype=bool)
    active = np.zeros(frames, dtype=np.int16)
    near = np.zeros(frames, dtype=bool)  # sung, or within the edge tolerance of a sung word
    masked = np.zeros(frames, dtype=bool)
    words = []
    for token in tokens:
        start, end = float(token["start"]), float(token["end"])
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        span = frame_span(start, end, frames)
        if token.get("ignore") or not token.get("ids"):
            masked[span.start : span.stop] = True
            continue
        ids = [int(i) for i in token["ids"]]
        active[span.start : span.stop] += 1
        lo, hi = max(0, span.start - edge_tolerance), min(frames, span.stop + edge_tolerance)
        near[lo:hi] = True
        allowed[lo:hi][:, sorted(set(ids))] = True
        words.append((span.start, span.stop, ids))
    overlap = active >= 2
    regions = []
    for first, stop, ids in words:
        if stop - first >= len(ids):
            weight = overlap_token_weight if overlap[first:stop].any() else 1.0
            regions.append((first, stop, ids, weight))
    letters = list(LETTERS)
    negative = np.zeros((frames, vocab_size), dtype=bool)
    negative[:, letters] = ~allowed[:, letters]
    negative[masked] = False
    silence = (active == 0).astype(np.float32)
    silence_graded = ~masked & ((active > 0) | ~near)
    return {
        "regions": regions,
        "negative": negative,
        "silence": silence,
        "silence_graded": silence_graded,
        "overlap": overlap,
    }
