"""Weak timed-character labels on the MMS convolution frame-center clock."""
import math

PAD_TOKEN_ID = 29
IGNORE_INDEX = -100
# MMS/Wav2Vec2: kernels 10,3,3,3,3,2,2; strides 5,2,2,2,2,2,2.
# Effective stride is 320 samples; receptive field is 400 samples at 16 kHz.
FRAME_STRIDE_SECONDS = 320 / 16000
FRAME_OFFSET_SECONDS = 199.5 / 16000


def frame_labels(alignment_tokens: list[dict], audio_duration_seconds: float,
                 output_frames: int, frame_stride_seconds: float = FRAME_STRIDE_SECONDS,
                 frame_offset_seconds: float = FRAME_OFFSET_SECONDS) -> list[int]:
    """Assign each output frame using its receptive-field center in audio.

    Token character boundaries remain weak proportional estimates inside the
    original source interval. Never stretch the frame clock to fit a window.
    Overlaps and explicit masks are sticky ignore-index labels. Soft masks
    (overlapped background vocals) only ignore frames nothing else labels,
    so the lead stays supervised underneath them.
    """
    if output_frames <= 0 or not math.isfinite(audio_duration_seconds) or audio_duration_seconds <= 0:
        raise ValueError("positive finite audio duration and output frame count required")
    if not math.isfinite(frame_stride_seconds) or frame_stride_seconds <= 0 or not math.isfinite(frame_offset_seconds):
        raise ValueError("invalid frame clock")
    labels = [PAD_TOKEN_ID] * output_frames
    soft = [token for token in alignment_tokens if token.get("soft_ignore")]
    for token in [*(t for t in alignment_tokens if not t.get("soft_ignore")), *soft]:
        start, end = token["start"], token["end"]
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError("invalid token interval")
        ids = token["ids"]
        if not ids and not token.get("ignore"):
            raise ValueError("unmasked token without labels")
        if any(not isinstance(value, int) or not 1 <= value <= 26 for value in ids):
            raise ValueError("token IDs must be lowercase pronunciation characters")
        first = max(0, math.ceil((start - frame_offset_seconds) / frame_stride_seconds - 1e-9))
        stop = min(output_frames, math.ceil((end - frame_offset_seconds) / frame_stride_seconds - 1e-9))
        for frame in range(first, stop):
            if token.get("soft_ignore"):
                if labels[frame] == PAD_TOKEN_ID:
                    labels[frame] = IGNORE_INDEX
                continue
            if token.get("ignore") or labels[frame] != PAD_TOKEN_ID:
                labels[frame] = IGNORE_INDEX
                continue
            center = frame_offset_seconds + frame * frame_stride_seconds
            index = min(len(ids) - 1, max(0, int((center - start) * len(ids) / (end - start))))
            labels[frame] = ids[index]
    return labels


def frame_index(seconds: float, frame_stride_seconds: float = FRAME_STRIDE_SECONDS,
                frame_offset_seconds: float = FRAME_OFFSET_SECONDS) -> int:
    """First frame whose receptive-field center is at or after ``seconds``."""
    return math.ceil((seconds - frame_offset_seconds) / frame_stride_seconds - 1e-9)


def ctc_regions(alignment_tokens: list[dict], labels: list[int], audio_duration_seconds: float,
                frame_stride_seconds: float = FRAME_STRIDE_SECONDS,
                frame_offset_seconds: float = FRAME_OFFSET_SECONDS,
                repeat_blanks: bool = True) -> list[tuple[int, int, list[int]]]:
    """Whole source fragments whose letters the region loss may place freely.

    Each region is one timed source fragment, not one 6-letter chunk: chunk
    edges are proportional guesses, fragment edges are measured. Returns
    (first_frame, stop_frame, ids). A fragment falls back to frame labels
    when it is cut by the window, overlaps a mask, or is too short for CTC.
    """
    groups: dict[tuple[float, float], list[dict]] = {}
    for token in alignment_tokens:
        if not token.get("ignore") and "region_start" in token:
            groups.setdefault((token["region_start"], token["region_end"]), []).append(token)
    regions = []
    for (start, end), members in sorted(groups.items()):
        first = max(0, frame_index(start, frame_stride_seconds, frame_offset_seconds))
        stop = min(len(labels), frame_index(end, frame_stride_seconds, frame_offset_seconds))
        ids = [i for token in sorted(members, key=lambda t: t["start"]) for i in token["ids"]]
        # CTC needs a blank between repeated letters; blank-free alignment does not.
        needed = len(ids) + (sum(a == b for a, b in zip(ids, ids[1:])) if repeat_blanks else 0)
        if start < 0 or end > audio_duration_seconds + 1e-6 or not ids or stop - first < needed:
            continue
        if any(labels[f] == IGNORE_INDEX for f in range(first, stop)):
            continue
        regions.append((first, stop, ids))
    return regions


def tolerant_frame_targets(labels: list[int], regions: list[tuple[int, int, list[int]]],
                           tolerance_frames: int) -> list[int]:
    """Frame-loss targets once CTC owns its regions.

    Region frames leave the frame loss. Blank frames within ``tolerance_frames``
    of a region edge are not graded, so starting or ending a fragment one
    frame early or late against silence is not penalized.
    """
    targets = list(labels)
    for first, stop, _ in regions:
        targets[first:stop] = [IGNORE_INDEX] * (stop - first)
    for first, stop, _ in regions:
        for frame in [*range(first - tolerance_frames, first), *range(stop, stop + tolerance_frames)]:
            if 0 <= frame < len(targets) and targets[frame] == PAD_TOKEN_ID:
                targets[frame] = IGNORE_INDEX
    return targets


def ignore_unregioned(alignment_tokens: list[dict], labels: list[int], regions: list[tuple[int, int, list[int]]],
                      frame_stride_seconds: float = FRAME_STRIDE_SECONDS,
                      frame_offset_seconds: float = FRAME_OFFSET_SECONDS) -> list[int]:
    """``--region-fallback ignore``: frames of a measured fragment that got no region (cut by the
    window, overlapped by a mask or another fragment, or too short for its letters) become ignore
    instead of the proportional per-letter guess (in a typical manifest ~1.7 % of fragments are
    window-cut and ~1.5 % overlap another lead fragment). Frames owned by a fragment that
    did get a region are left alone."""
    owned = [False] * len(labels)
    for first, stop, _ in regions:
        for frame in range(first, stop):
            owned[frame] = True
    result = list(labels)
    regioned = {(first, stop) for first, stop, _ in regions}
    for token in alignment_tokens:
        if token.get("ignore") or "region_start" not in token:
            continue
        first = max(0, frame_index(token["region_start"], frame_stride_seconds, frame_offset_seconds))
        stop = min(len(labels), frame_index(token["region_end"], frame_stride_seconds, frame_offset_seconds))
        if (first, stop) in regioned:
            continue
        for frame in range(first, stop):
            if not owned[frame] and result[frame] != PAD_TOKEN_ID:
                result[frame] = IGNORE_INDEX
    return result
