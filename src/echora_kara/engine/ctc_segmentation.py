"""Global CTC forced alignment over an already-computed emission lattice.

Plain Viterbi (with optional soft onset priors, which every current caller leaves empty), duration-floor and
duration-model Viterbi variants, vocal-activity penalties, sub-frame start refinement and the exact frame clock of
the stitched wav2vec2 / w2v-BERT lattice. It does not load an acoustic model or crop audio.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

import torch
from torchaudio.functional import TokenSpan


@dataclass(frozen=True)
class SourcePrior:
    """Soft onset prior: entering flattened target token ``target_index`` far from ``time_seconds`` costs a Huber
    penalty scaled by ``weight``."""

    target_index: int
    time_seconds: float
    weight: float = 1.0


def _huber(value: float, delta: float) -> float:
    value = abs(value)
    if value <= delta:
        return 0.5 * value * value / delta
    return value - 0.5 * delta


def minimum_ctc_frames(token_groups: list[list[int]]) -> int:
    """CTC needs one frame per token plus blanks between repeated labels."""
    targets = [token for group in token_groups for token in group]
    return len(targets) + sum(left == right for left, right in zip(targets, targets[1:]))


def align_with_source_priors(
    emission: torch.Tensor,
    token_groups: list[list[int]],
    *,
    blank: int,
    frame_shift_seconds: float,
    source_priors: list[SourcePrior],
    prior_weight: float = 0.8,
    prior_delta_seconds: float = 0.75,
) -> list[list[TokenSpan]]:
    """Align token groups globally using CTC Viterbi and soft onset priors.

    ``emission`` must contain log probabilities with shape ``[frames, labels]``.
    A prior is charged only when the path enters its target token. It therefore
    influences the line onset without pulling every token toward that onset.
    """
    if emission.ndim != 2 or emission.shape[0] == 0:
        raise ValueError("emission must have shape [frames, labels] and contain frames")
    targets = [token for group in token_groups for token in group]
    if not targets:
        raise ValueError("cannot align an empty token sequence")
    if emission.shape[0] < minimum_ctc_frames(token_groups):
        raise RuntimeError("CTC emission has fewer frames than target tokens")
    if blank < 0 or blank >= emission.shape[1]:
        raise ValueError("blank token is outside the emission vocabulary")
    if any(token < 0 or token >= emission.shape[1] for token in targets):
        raise ValueError("target token is outside the emission vocabulary")

    # Interleave blanks and labels. State 2*i+1 represents target i.
    state_labels = [blank]
    for token in targets:
        state_labels.extend((token, blank))
    labels = torch.tensor(state_labels, dtype=torch.long, device=emission.device)
    state_count = len(state_labels)
    frame_count = int(emission.shape[0])
    negative_inf = torch.tensor(float("-inf"), device=emission.device)

    prior_by_target = {prior.target_index: (prior.time_seconds, prior.weight) for prior in source_priors}
    if any(target < 0 or target >= len(targets) for target in prior_by_target):
        invalid = next(target for target in prior_by_target if target < 0 or target >= len(targets))
        raise ValueError(f"source prior target {invalid} is outside the transcript")
    prior_targets = list(prior_by_target)
    prior_states = torch.tensor(
        [target * 2 + 1 for target in prior_targets],
        dtype=torch.long,
        device=emission.device,
    )
    prior_times = torch.tensor(
        [prior_by_target[target][0] for target in prior_targets],
        dtype=emission.dtype,
        device=emission.device,
    )
    prior_weights = torch.tensor(
        [prior_by_target[target][1] for target in prior_targets],
        dtype=emission.dtype,
        device=emission.device,
    )
    scores = torch.full((state_count,), float("-inf"), device=emission.device)
    scores[0] = emission[0, blank]
    if targets:
        initial = emission[0, targets[0]]
        if 0 in prior_by_target:
            expected_time, weight = prior_by_target[0]
            initial -= prior_weight * weight * _huber(expected_time, prior_delta_seconds)
        scores[1] = initial

    # 0 means stay, 1 advance one state, 2 skip a blank. One byte per state.
    backpointers = torch.zeros((frame_count, state_count), dtype=torch.uint8, device=emission.device)
    state_indexes = torch.arange(state_count, device=emission.device)
    is_label = state_indexes.remainder(2).eq(1)
    skip_allowed = torch.zeros(state_count, dtype=torch.bool, device=emission.device)
    if state_count > 2:
        skip_allowed[2:] = is_label[2:] & labels[2:].ne(labels[:-2])

    for frame in range(1, frame_count):
        stay = scores
        advance = torch.cat((negative_inf.reshape(1), scores[:-1]))
        skip = torch.cat((negative_inf.repeat(2), scores[:-2]))
        skip = torch.where(skip_allowed, skip, negative_inf)
        choices = torch.stack((stay, advance, skip), dim=0)

        # Priors belong to transitions into acoustic tokens. Compute all
        # penalties together; per-prior Python loops dominate long songs.
        if prior_states.numel():
            residual = (frame * frame_shift_seconds - prior_times).abs()
            huber = torch.where(
                residual <= prior_delta_seconds,
                0.5 * residual.square() / prior_delta_seconds,
                residual - 0.5 * prior_delta_seconds,
            )
            penalties = prior_weight * prior_weights * huber
            choices[1:, prior_states] -= penalties.unsqueeze(0)

        best_scores, best_moves = choices.max(dim=0)
        scores = best_scores + emission[frame].index_select(0, labels)
        backpointers[frame] = best_moves.to(dtype=torch.uint8)

    final_states = [state_count - 1, state_count - 2]
    state = max(final_states, key=lambda index: float(scores[index]))
    if not math.isfinite(float(scores[state])):
        raise RuntimeError("no finite CTC path exists for the supplied transcript")

    backpointers = backpointers.cpu()
    path = [state]
    for frame in range(frame_count - 1, 0, -1):
        move = int(backpointers[frame, state])
        state -= move
        path.append(state)
    path.reverse()

    flat_spans: list[TokenSpan] = []
    for target_index, token in enumerate(targets):
        token_state = target_index * 2 + 1
        frames = [frame for frame, path_state in enumerate(path) if path_state == token_state]
        if not frames:
            raise RuntimeError(f"CTC path omitted target token {target_index}")
        token_scores = emission[frames, token].exp()
        flat_spans.append(TokenSpan(token, frames[0], frames[-1] + 1, float(token_scores.mean())))

    grouped: list[list[TokenSpan]] = []
    offset = 0
    for group in token_groups:
        grouped.append(flat_spans[offset : offset + len(group)])
        offset += len(group)
    return grouped


def token_minimum_frames(
    token_groups: list[list[int]],
    letter_frames: int,
    group_frames: int,
    vowel_ids: set[int],
) -> list[int]:
    """Minimum frames per target: each letter at least ``letter_frames``, each acoustic token
    (group) at least ``group_frames``, its extra frames given to vowels (what singers hold)."""
    minimums: list[int] = []
    for group in token_groups:
        frames = [max(1, letter_frames)] * len(group)
        extra = group_frames - sum(frames)
        slots = [index for index, token in enumerate(group) if token in vowel_ids] or [0]
        step = 0
        while extra > 0 and frames:
            frames[slots[step % len(slots)]] += 1
            extra -= 1
            step += 1
        minimums.extend(frames)
    return minimums


def align_with_minimum_durations(
    emission: torch.Tensor,
    token_groups: list[list[int]],
    *,
    blank: int,
    minimum_frames: list[int],
    acoustic_token_groups: list[list[int]] | None = None,
    edge_logits: torch.Tensor | None = None,
    edge_weight: float = 0.0,
    blank_free_tokens: bool = False,
    line_ranges: list[tuple[int, int]] | None = None,
    in_line_gap_cost: float = 0.0,
) -> list[list[TokenSpan]]:
    """Global CTC Viterbi where target i occupies at least ``minimum_frames[i]`` frames.

    ``acoustic_token_groups`` optionally supplies repetition identities independently of
    emission-column IDs. Returned spans still use emission-column IDs.

    Plain CTC lets a letter take one frame, so with flat scores the best path crams words or holds
    one and squeezes the next. Each target becomes a chain of states walked one frame at a time
    before its last state may repeat; blanks are unchanged. All ones is the ordinary topology.

    Edge head (opt-in, SOFA-style): ``edge_logits`` [frames] are a boundary head's logits (the
    probability that an acoustic token starts on that frame). With ``edge_weight`` w > 0 every
    move that *enters* a token group's first letter at frame t pays ``w * log sigmoid(e_t)`` and
    every other move (stay, letter-internal advance, blank) pays ``w * log(1 - sigmoid(e_t))``.
    Zero weight (the default) leaves the path exactly as before.

    ``blank_free_tokens`` (opt-in) drops the blank between letters of the same token group, as
    training fills each timed fragment blank-free: a held note must stay on a letter of its word and
    blanks (silence) only fall between tokens. Off (the default) leaves the path exactly as before.

    ``in_line_gap_cost`` (opt-in, with ``line_ranges`` over token groups) charges that much per blank
    frame between two tokens of the same line; blanks before a line's first token stay free. Reference
    word timings put 95 % of in-line gaps at zero, so a long instrumental pause lands between lines
    rather than inside one ("See the sunset | And I'm gonna stay", not "... And | I'm gonna ...").
    """
    # Emission columns can be unique pseudo-labels; CTC repetition uses acoustic identity.
    acoustic_groups = token_groups if acoustic_token_groups is None else acoustic_token_groups
    if [len(g) for g in acoustic_groups] != [len(g) for g in token_groups]:
        raise ValueError("acoustic token groups must match emission token groups")
    acoustic_targets = [token for group in acoustic_groups for token in group]
    targets = [token for group in token_groups for token in group]
    if emission.ndim != 2 or emission.shape[0] == 0 or not targets:
        raise ValueError("emission and transcript must be non-empty")
    if len(minimum_frames) != len(targets) or any(frames < 1 for frames in minimum_frames):
        raise ValueError("every target needs a minimum of at least one frame")
    group_first = {sum(len(group) for group in token_groups[:index]) for index in range(len(token_groups))}
    group_last = {first - 1 for first in group_first if first} | {len(targets) - 1}
    in_line_after = set()  # targets ending a group whose next group is on the same line
    if line_ranges and in_line_gap_cost > 0:
        for start, end in line_ranges:
            for g in range(start, end - 1):
                in_line_after.add(sum(len(group) for group in token_groups[: g + 1]) - 1)
    in_line_blank = [False]
    labels = [blank]
    kinds = [0]  # 0 blank, 1 first sub-state, 2 inner, 3 last, 4 first and last
    owners = [-1]
    entries = [False]  # the first sub-state of a token group's first letter
    for index, (token, frames) in enumerate(zip(targets, minimum_frames)):
        for step in range(frames):
            labels.append(token)
            kinds.append(4 if frames == 1 else 1 if step == 0 else 3 if step == frames - 1 else 2)
            owners.append(index)
            entries.append(step == 0 and index in group_first)
        if blank_free_tokens and index not in group_last:
            continue
        labels.append(blank)
        kinds.append(0)
        owners.append(-1)
        entries.append(False)
        in_line_blank.append(index in in_line_after)
    if emission.shape[0] < sum(minimum_frames):
        raise RuntimeError("CTC emission has fewer frames than the minimum durations need")
    device = emission.device
    state_labels = torch.tensor(labels, dtype=torch.long, device=device)
    state_kinds = torch.tensor(kinds, device=device)
    can_stay = (state_kinds == 0) | (state_kinds == 3) | (state_kinds == 4)
    can_skip = torch.zeros(len(labels), dtype=torch.bool, device=device)
    for state, kind in enumerate(kinds):
        if (
            kind in (1, 4)
            and state >= 2
            and kinds[state - 1] == 0
            and kinds[state - 2] in (3, 4)
            and acoustic_targets[owners[state - 2]] != acoustic_targets[owners[state]]
        ):
            can_skip[state] = True
    use_edges = edge_logits is not None and edge_weight > 0
    if use_edges:
        if edge_logits.shape[0] != emission.shape[0]:
            raise ValueError("edge logits must have one value per emission frame")
        edge = edge_logits.to(device=device, dtype=emission.dtype).flatten()
        log_edge = edge_weight * torch.nn.functional.logsigmoid(edge)
        log_no_edge = edge_weight * torch.nn.functional.logsigmoid(-edge)
        entry_states = torch.tensor(entries, dtype=torch.bool, device=device)
    negative_inf = torch.tensor(float("-inf"), device=device)
    gap_penalty = None
    if in_line_after:
        flags, flag_iter = [], iter(in_line_blank)
        for kind in kinds:
            flags.append(next(flag_iter) if kind == 0 else False)
        gap_penalty = torch.tensor(flags, device=device, dtype=emission.dtype) * in_line_gap_cost
    scores = torch.full((len(labels),), float("-inf"), device=device)
    scores[0] = emission[0, blank]
    scores[1] = emission[0, labels[1]]
    if use_edges:
        scores[0] += log_no_edge[0]
        scores[1] += log_edge[0]
    backpointers = torch.zeros((emission.shape[0], len(labels)), dtype=torch.uint8, device=device)
    for frame in range(1, emission.shape[0]):
        stay = torch.where(can_stay, scores, negative_inf)
        advance = torch.cat((negative_inf.reshape(1), scores[:-1]))
        skip = torch.where(can_skip, torch.cat((negative_inf.repeat(2), scores[:-2])), negative_inf)
        if use_edges:
            entering = torch.where(entry_states, log_edge[frame], log_no_edge[frame])
            stay = stay + log_no_edge[frame]
            advance = advance + entering
            skip = skip + entering
        best_scores, best_moves = torch.stack((stay, advance, skip)).max(dim=0)
        scores = best_scores + emission[frame].index_select(0, state_labels)
        if gap_penalty is not None:
            scores = scores - gap_penalty
        backpointers[frame] = best_moves.to(dtype=torch.uint8)
    state = max((len(labels) - 1, len(labels) - 2), key=lambda index: float(scores[index]))
    if not math.isfinite(float(scores[state])):
        raise RuntimeError("no finite CTC path exists for the supplied transcript")
    backpointers = backpointers.cpu()
    path = [state]
    for frame in range(emission.shape[0] - 1, 0, -1):
        state -= int(backpointers[frame, state])
        path.append(state)
    path.reverse()
    frames_by_target: list[list[int]] = [[] for _ in targets]
    for frame, path_state in enumerate(path):
        if owners[path_state] >= 0:
            frames_by_target[owners[path_state]].append(frame)
    flat_spans = []
    for index, token in enumerate(targets):
        frames = frames_by_target[index]
        if not frames:
            raise RuntimeError(f"CTC path omitted target token {index}")
        token_scores = emission[frames, token].exp()
        flat_spans.append(TokenSpan(token, frames[0], frames[-1] + 1, float(token_scores.mean())))
    grouped: list[list[TokenSpan]] = []
    offset = 0
    for group in token_groups:
        grouped.append(flat_spans[offset : offset + len(group)])
        offset += len(group)
    return grouped


def paced_minimum_frames(
    token_groups: list[list[int]],
    plain_spans: list[list[TokenSpan]],
    line_ranges: list[tuple[int, int]],
    *,
    ratio: float,
    letter_cap: int,
    group_cap: int,
    pause_frames: int,
    vowel_ids: set[int],
) -> list[int]:
    """Minimum frames per target, scaled to each line's sung pace in a plain (unconstrained) pass.

    A fixed floor is right for most singing but longer than fast rap actually takes, which forces
    lines out of place. A line's pace is its span minus pauses longer than ``pause_frames`` between
    acoustic tokens: cramming can't shrink a span and pauses don't inflate it. Tokens get ``ratio``
    x the line's frames per token (at most ``group_cap``) and letters ``ratio`` x its frames per
    letter (at most ``letter_cap``), so fast rap keeps short syllables and one-frame letters.
    """
    minimums: list[int] = []
    for start, end in line_ranges:
        spans = [(group[0].start, group[-1].end) for group in plain_spans[start:end] if group]
        if not spans:
            continue
        spans.sort()
        total = max(span_end for _, span_end in spans) - spans[0][0]
        pauses = sum(
            max(0, later[0] - earlier[1])
            for earlier, later in zip(spans, spans[1:])
            if later[0] - earlier[1] > pause_frames
        )
        sung = max(1.0, total - pauses)
        letters = sum(len(group) for group in token_groups[start:end])
        group_floor = max(1, min(group_cap, int(ratio * sung / len(spans))))
        letter_floor = max(1, min(letter_cap, int(ratio * sung / max(1, letters))))
        minimums.extend(token_minimum_frames(token_groups[start:end], letter_floor, group_floor, vowel_ids))
    return minimums


def vocal_activity(waveform: torch.Tensor, frame_count: int, hop: int = 320, window: int = 400) -> torch.Tensor:
    """Per-frame singing activity in [0, 1] from a 16 kHz vocal stem.

    Loudness (dB) over each frame's receptive field, scaled between the song's 10th and 90th
    percentile, so quiet passages read near 0 and full singing near 1.
    """
    waveform = waveform.flatten().float()
    padded = torch.nn.functional.pad(waveform, (0, max(0, frame_count * hop + window - waveform.numel())))
    frames = padded.unfold(0, window, hop)[:frame_count]
    loudness = 10 * torch.log10(frames.square().mean(dim=1) + 1e-10)
    low, high = torch.quantile(loudness, 0.1), torch.quantile(loudness, 0.9)
    return ((loudness - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)


def with_vocal_activity(emission: torch.Tensor, activity: torch.Tensor, blank: int, weight: float) -> torch.Tensor:
    """Letters cost ``weight`` x (1 - activity) extra per frame, so the path can't hold a word
    through silence or start a line in an instrumental; blanks (pauses) are unchanged."""
    penalty = weight * (1 - activity.to(emission.device, emission.dtype))
    adjusted = emission - penalty.unsqueeze(1)
    adjusted[:, blank] = emission[:, blank]
    return adjusted


# --- Exact frame clock of the stitched 45 s chunk inference ------------------------------------

CONV_KERNELS = (10, 3, 3, 3, 3, 2, 2)
CONV_STRIDES = (5, 2, 2, 2, 2, 2, 2)
FRAME_STRIDE_SAMPLES = 320
RECEPTIVE_FIELD_CENTRE_SAMPLES = 199.5  # centre of the 400-sample receptive field of frame 0
# w2v-BERT 2.0: Kaldi fbank frames of 400 samples every 160, stacked in pairs by the
# SeamlessM4T feature extractor. Model frame j spans samples [320 j, 320 j + 560): same 20 ms
# stride as wav2vec2, centre 279.5 samples; floor(fbank / 2) frames with fbank = floor((N - 400) / 160) + 1.
W2V_BERT_FBANK_WINDOW, W2V_BERT_FBANK_HOP, W2V_BERT_STACK = 400, 160, 2
W2V_BERT_RECEPTIVE_FIELD_CENTRE_SAMPLES = 279.5
BACKBONES = ("wav2vec2", "wav2vec2-bert")


def conv_frame_count(samples: int) -> int:
    """Output frames of the wav2vec2 feature encoder for ``samples`` input samples."""
    length = samples
    for kernel, stride in zip(CONV_KERNELS, CONV_STRIDES):
        length = (length - kernel) // stride + 1
    return max(0, length)


def w2v_bert_frame_count(samples: int) -> int:
    """Valid (unpadded) model frames of the w2v-BERT front end for ``samples`` input samples."""
    if samples < W2V_BERT_FBANK_WINDOW:
        return 0
    fbank = (samples - W2V_BERT_FBANK_WINDOW) // W2V_BERT_FBANK_HOP + 1
    return fbank // W2V_BERT_STACK


def frame_count(samples: int, backbone: str = "wav2vec2") -> int:
    """Model frames produced for one chunk of ``samples`` input samples by the given backbone's front end."""
    if backbone == "wav2vec2-bert":
        return w2v_bert_frame_count(samples)
    if backbone == "wav2vec2":
        return conv_frame_count(samples)
    raise ValueError(f"unknown backbone {backbone!r}")


def receptive_field_centre_samples(backbone: str = "wav2vec2") -> float:
    """Offset in samples from a chunk start to the receptive-field centre of its first frame (199.5 for wav2vec2,
    279.5 for w2v-BERT); later frames follow every 320 samples."""
    if backbone == "wav2vec2-bert":
        return W2V_BERT_RECEPTIVE_FIELD_CENTRE_SAMPLES
    if backbone == "wav2vec2":
        return RECEPTIVE_FIELD_CENTRE_SAMPLES
    raise ValueError(f"unknown backbone {backbone!r}")


def stitched_frame_times(
    total_samples: int,
    *,
    sample_rate: int = 16000,
    chunk_seconds: int = 45,
    overlap_seconds: int = 2,
    backbone: str = "wav2vec2",
) -> torch.Tensor:
    """Receptive-field-centre time (seconds) of every frame of the stitched lattice that
    ``Wav2Vec2ForcedAligner.infer`` builds from 45 s chunks with 2 s shared context.

    Each chunk yields floor((N - 400) / 320) + 1 frames, one fewer than the nominal N / 320, so a
    constant frame shift drifts by 20 ms per chunk boundary (one frame lost per chunk) and the
    duration / frames shift of the legacy clock stretches the whole song to absorb it. Here every
    frame keeps its true centre: chunk start + 199.5 samples + 320 j (wav2vec2), or
    chunk start + 279.5 samples + 320 j with floor(fbank / 2) frames per chunk (wav2vec2-bert).
    """
    chunk_samples, overlap_samples = chunk_seconds * sample_rate, overlap_seconds * sample_rate
    step_samples = chunk_samples - overlap_samples
    centre = receptive_field_centre_samples(backbone)
    starts = [
        start
        for start in range(0, total_samples, step_samples)
        if start == 0 or total_samples - start > overlap_samples
    ]
    pieces = []
    for chunk_index, start in enumerate(starts):
        length = min(total_samples, start + chunk_samples) - start
        frames = frame_count(length, backbone)
        overlap_frames = round(frames * overlap_samples / max(1, length))
        trim_left = overlap_frames // 2 if chunk_index else 0
        trim_right = overlap_frames - overlap_frames // 2 if chunk_index + 1 < len(starts) else 0
        kept = torch.arange(trim_left, frames - trim_right, dtype=torch.float64)
        pieces.append((start + centre + FRAME_STRIDE_SAMPLES * kept) / sample_rate)
    return torch.cat(pieces) if pieces else torch.zeros(0, dtype=torch.float64)


# --- Sub-frame start refinement and onset classes ------------------------------------------------


def refine_group_starts(
    emission: torch.Tensor,
    spans: list[list[TokenSpan]],
    *,
    blank: int,
    limit: float = 0.5,
) -> list[float]:
    """Fractional start (in frames, added to the first frame) of every token group.

    The Viterbi puts a group's first letter c on frame f, so the real onset lies between frame
    f - 1 (still the previous symbol p: blank, or the previous group's last letter when the two
    abut) and frame f. The log-odds d_t = log P(c | t) - log P(p | t) are interpolated linearly
    between f - 1 and f: the zero crossing is the sub-frame onset (MAPS / SOFA style). If the
    odds do not cross (both frames already favour c, or neither does) the three frames f - 1,
    f, f + 1 are fitted by least squares instead. Clipped to +-``limit`` frames.
    """
    frames = emission.shape[0]
    out: list[float] = []
    previous: TokenSpan | None = None
    for group in spans:
        if not group:
            out.append(0.0)
            continue
        first = group[0]
        f, c = first.start, first.token
        p = previous.token if previous is not None and previous.end == f and previous.token != c else blank
        if f - 1 < 0:
            out.append(0.0)
            previous = group[-1]
            continue
        d_prev = float(emission[f - 1, c] - emission[f - 1, p])
        d_here = float(emission[f, c] - emission[f, p])
        if d_prev < 0 < d_here:
            # Crossing between the two frame centres: f - 1 + d_prev / (d_prev - d_here).
            delta = d_prev / (d_prev - d_here) - 1.0
        else:
            xs = [-1.0, 0.0]
            ys = [d_prev, d_here]
            if f + 1 < frames:
                xs.append(1.0)
                ys.append(float(emission[f + 1, c] - emission[f + 1, p]))
            n = len(xs)
            mean_x, mean_y = sum(xs) / n, sum(ys) / n
            slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / max(
                1e-9, sum((x - mean_x) ** 2 for x in xs)
            )
            delta = (mean_x - mean_y / slope) if slope > 1e-6 else 0.0
        if not math.isfinite(delta):
            delta = 0.0
        out.append(max(-limit, min(limit, delta)))
        previous = group[-1]
    return out


# Interjections as duration_viterbi.INTERJECTION counts them (held "ooh" runs).
INTERJECTION_PATTERN = re.compile(r"^(o+h*|o+o+h*|a+h+|whoa+|wo+a+h*|yeah|na|la|hey|uh|ooh+|mm+)$")

ONSET_CLASSES = ("stop_vl", "stop_vd", "fricative", "nasal", "approximant", "vowel", "interjection", "other")
_ONSET_BY_LETTER = {
    **dict.fromkeys("ptkcq", "stop_vl"),
    **dict.fromkeys("bdg", "stop_vd"),
    **dict.fromkeys("fshvzx", "fricative"),
    **dict.fromkeys("mn", "nasal"),
    **dict.fromkeys("lrwyj", "approximant"),
    **dict.fromkeys("aeiou", "vowel"),
}


def onset_class(text: str) -> str:
    """Word-initial sound class of a romanised token (for signed-bias tables and offsets)."""
    letters = "".join(ch for ch in text.lower() if "a" <= ch <= "z")
    if not letters:
        return "other"
    if INTERJECTION_PATTERN.match(letters):
        return "interjection"
    return _ONSET_BY_LETTER.get(letters[0], "other")


# --- Duration-model (semi-Markov) decoder ---------------------------------------------------------

DURATION_VOWELS = re.compile(r"[aeiouy]+")
DURATION_LATIN = {
    "en",
    "de",
    "fr",
    "es",
    "pt",
    "it",
    "id",
    "ms",
    "nl",
    "pl",
    "af",
    "sv",
    "no",
    "da",
    "fi",
    "tr",
    "vi",
    "ro",
    "cs",
    "hu",
    "hr",
    "sk",
    "sl",
    "ca",
    "tl",
    "sw",
    "xh",
    "zu",
    "et",
    "lv",
    "lt",
}
DURATION_SYLLABIC = {"zh", "yue", "ja", "ko"}


def duration_features(text: str, ln_pace: float, final: bool, initial: bool, language: str | None) -> list[float]:
    """Per-token features of the duration model (its mu / ln_sigma are linear in these; keep the
    model fit on the same features)."""
    group = (
        "en"
        if language == "en"
        else "latin"
        if language in DURATION_LATIN
        else "syllabic"
        if language in DURATION_SYLLABIC
        else "other"
    )
    syllables = min(6, max(1, len(DURATION_VOWELS.findall(text))))
    return [
        1.0,
        *[1.0 if syllables == k else 0.0 for k in range(2, 7)],
        math.log(max(1, len(text))),
        ln_pace,
        float(final),
        float(initial),
        float(final) * ln_pace,
        *[1.0 if group == g else 0.0 for g in ("latin", "syllabic", "other")],
    ]


def _line_paces(plain_spans, line_ranges, frame_shift_seconds: float, pause_frames: int) -> list[float]:
    """ln(sung seconds per token) of each line in the plain pass (as paced_minimum_frames measures pace)."""
    paces = []
    for start, end in line_ranges:
        spans = sorted((group[0].start, group[-1].end) for group in plain_spans[start:end] if group)
        total = max(e for _, e in spans) - spans[0][0]
        pauses = sum(max(0, b[0] - a[1]) for a, b in zip(spans, spans[1:]) if b[0] - a[1] > pause_frames)
        paces.append(math.log(max(0.02, max(1.0, total - pauses) * frame_shift_seconds / (end - start))))
    return paces


def align_with_duration_model(
    emission: torch.Tensor,
    token_groups: list[list[int]],
    token_texts: list[str],
    line_ranges: list[tuple[int, int]],
    plain_spans: list[list[TokenSpan]],
    *,
    blank: int,
    frame_shift_seconds: float,
    model: dict,
    language: str | None = None,
    weight: float = 1.0,
    gap_weight: float = 1.0,
    band_seconds: float = 15.0,
    max_gap_seconds: float = 4.0,
    max_sigma: float = 3.5,
    one_sided_sigma: float | None = None,
    token_minimum_frames: list[int] | None = None,
) -> list[list[TokenSpan]]:
    """Semi-Markov Viterbi with a learned token-duration and gap model.

    Each acoustic token is one segment [t, t + d) filled blank-free by its letters (as ``blank_free_tokens``);
    blanks only fall between tokens. A path scores the emission log-probabilities of its frames plus,
    per token, ``weight`` x the log-normal log-density of its duration (``model``: mu / ln_sigma linear
    in ``duration_features``), and per gap inside a line ``gap_weight`` x log P(gap) (zero-gap mass +
    log-normal over pace). Gaps between lines cost nothing. ``weight`` 0 is a floor-free blank-free
    decoder. Token starts are kept within ``band_seconds`` of the plain pass (``plain_spans``); a
    token lasts at most mu + ``max_sigma`` sigma. Raises RuntimeError when no path fits the band.

    ``one_sided_sigma`` k: durations up to mu + k sigma cost nothing and longer ones pay
    ``weight`` x (z - k)^2 / 2 (z the duration's standard score), so plausible held notes are not
    pulled towards the typical length; ``token_minimum_frames`` (per token, e.g. the sum of the
    paced letter floors) is then the hard lower bound instead of the density's left tail.
    """
    device = emission.device
    frames = emission.shape[0]
    shift = frame_shift_seconds
    negative_inf = float("-inf")
    mu_beta = torch.tensor(model["mu"], dtype=torch.float64)
    sigma_beta = torch.tensor(model["ln_sigma"], dtype=torch.float64)
    gap_model = model["gap"]
    line_of, final, initial = {}, set(), set()
    for line_index, (start, end) in enumerate(line_ranges):
        for index in range(start, end):
            line_of[index] = line_index
        final.add(end - 1)
        initial.add(start)
    paces = _line_paces(plain_spans, line_ranges, shift, round(0.4 / shift))
    blank_cumulative = torch.cat(
        (torch.zeros(1, device=device, dtype=torch.float64), emission[:, blank].to(torch.float64).cumsum(0))
    )
    band = round(band_seconds / shift)
    max_gap = max(1, round(max_gap_seconds / shift))
    gap_frames = torch.arange(1, max_gap + 1, dtype=torch.float64, device=device)

    def gap_scores(ln_pace: float) -> tuple[float, torch.Tensor, float]:
        """(log P(gap = 0), log P(gap = g frames) for g = 1..max_gap, tail cost beyond)."""
        p_zero = min(0.999, max(1e-4, gap_model["p_zero"]))
        ln_mu = gap_model["ln_mu"][0] + gap_model["ln_mu"][1] * ln_pace
        ln_sigma = gap_model["ln_sigma"]
        seconds = gap_frames * shift
        log_density = (
            -torch.log(seconds)
            - math.log(ln_sigma)
            - (torch.log(seconds) - ln_mu) ** 2 / (2 * ln_sigma**2)
            + math.log(shift)
        )
        positive = math.log(1 - p_zero) + log_density
        return gap_weight * math.log(p_zero), gap_weight * positive, float(gap_weight * positive[-1])

    previous_end_score = None  # F_{i-1}(e) for e in 0..frames
    back_start, back_previous = [], []
    for index, letters in enumerate(token_groups):
        letter_count = len(letters)
        x = torch.tensor(
            duration_features(token_texts[index], paces[line_of[index]], index in final, index in initial, language),
            dtype=torch.float64,
        )
        mu, sigma = float(x @ mu_beta), float(torch.exp(x @ sigma_beta))
        max_duration = max(
            letter_count,
            math.ceil(math.exp(mu + max_sigma * sigma) / shift),
            token_minimum_frames[index] if token_minimum_frames is not None else 0,
        )
        centre = plain_spans[index][0].start
        first, last = max(0, centre - band), min(frames - letter_count, centre + band)
        if last < first:
            raise RuntimeError("duration decoder: token band outside the song")
        starts = torch.arange(first, last + 1, device=device)
        # G_i(t): best score of tokens < i with token i starting at t (blank frames before it included).
        if index == 0:
            start_score = blank_cumulative[starts]
            start_previous = torch.zeros_like(starts)
        else:
            h = previous_end_score - blank_cumulative  # F(e) - Cb[e]
            same_line = line_of[index] == line_of[index - 1]
            if same_line:
                zero_cost, positive_cost, tail_cost = gap_scores(paces[line_of[index]])
                best = h[starts] + zero_cost
                arg = starts.clone()
                for g in range(1, max_gap + 1):
                    candidate_end = starts - g
                    valid = candidate_end >= 0
                    value = torch.where(valid, h[candidate_end.clamp_min(0)] + positive_cost[g - 1], negative_inf)
                    better = value > best
                    best, arg = torch.where(better, value, best), torch.where(better, candidate_end, arg)
                # longer pauses inside a line: the prefix maximum, at the cost of the longest gap
                prefix_value, prefix_arg = torch.cummax(h, 0)
                candidate_end = starts - max_gap - 1
                valid = candidate_end >= 0
                value = torch.where(valid, prefix_value[candidate_end.clamp_min(0)] + tail_cost, negative_inf)
                better = value > best
                best = torch.where(better, value, best)
                arg = torch.where(better, prefix_arg[candidate_end.clamp_min(0)], arg)
            else:
                prefix_value, prefix_arg = torch.cummax(h, 0)
                best, arg = prefix_value[starts], prefix_arg[starts]
            start_score = best + blank_cumulative[starts]
            start_previous = arg
        # A_i(t, d) for every start in the band, one duration step at a time; F_i(e = t + d).
        token_emission = emission[:, letters].to(torch.float64)
        end_score = torch.full((frames + 1,), negative_inf, dtype=torch.float64, device=device)
        end_start = torch.zeros(frames + 1, dtype=torch.long, device=device)
        state = torch.full((len(starts), letter_count), negative_inf, dtype=torch.float64, device=device)
        state[:, 0] = token_emission[starts, 0]
        for duration in range(1, max_duration + 1):
            if duration > 1:
                frame = starts + duration - 1
                inside = frame < frames
                if not bool(inside.any()):
                    break
                moved = torch.cat(
                    (torch.full((len(starts), 1), negative_inf, dtype=torch.float64, device=device), state[:, :-1]), 1
                )
                state = torch.maximum(state, moved) + token_emission[frame.clamp_max(frames - 1)]
                state[~inside] = negative_inf
            if duration < letter_count or (token_minimum_frames is not None and duration < token_minimum_frames[index]):
                continue
            seconds = duration * shift
            if one_sided_sigma is not None:
                over = max(0.0, (math.log(seconds) - mu) / sigma - one_sided_sigma)
                duration_cost = -weight * over * over / 2
            else:
                duration_cost = weight * (
                    -math.log(seconds) - math.log(sigma) - (math.log(seconds) - mu) ** 2 / (2 * sigma**2)
                )
            value = start_score + state[:, -1] + duration_cost
            ends = starts + duration
            keep = ends <= frames
            ends, value, origin = ends[keep], value[keep], starts[keep]
            better = value > end_score[ends]
            end_score[ends[better]] = value[better]
            end_start[ends[better]] = origin[better]
        if not bool(torch.isfinite(end_score).any()):
            raise RuntimeError("duration decoder: no path fits the band")
        back_start.append(end_start)
        back_previous.append((starts, start_previous))
        previous_end_score = end_score
    final_score = previous_end_score + (blank_cumulative[-1] - blank_cumulative)
    end = int(torch.argmax(final_score))
    if not math.isfinite(float(final_score[end])):
        raise RuntimeError("duration decoder: no finite path")
    segments = [None] * len(token_groups)
    for index in range(len(token_groups) - 1, -1, -1):
        start = int(back_start[index][end])
        segments[index] = (start, end)
        starts, previous = back_previous[index]
        end = int(previous[start - int(starts[0])])
    grouped = []
    for (start, end), letters in zip(segments, token_groups):
        grouped.append(_blank_free_letters(emission, letters, start, end))
    return grouped


def _blank_free_letters(emission: torch.Tensor, letters: list[int], start: int, end: int) -> list[TokenSpan]:
    """Letter spans of one token filling [start, end) blank-free, in order, each at least one frame."""
    scores = emission[start:end][:, letters].to(torch.float64).cpu()
    count, length = len(letters), end - start
    best = torch.full((length, count), float("-inf"), dtype=torch.float64)
    came_from_previous = torch.zeros((length, count), dtype=torch.bool)
    best[0, 0] = scores[0, 0]
    for frame in range(1, length):
        stay = best[frame - 1]
        move = torch.cat((torch.tensor([float("-inf")], dtype=torch.float64), best[frame - 1, :-1]))
        came_from_previous[frame] = move > stay
        best[frame] = torch.maximum(stay, move) + scores[frame]
    boundaries, letter = [length], count - 1
    for frame in range(length - 1, 0, -1):
        if came_from_previous[frame, letter]:
            boundaries.append(frame)
            letter -= 1
    boundaries.append(0)
    boundaries.reverse()
    spans = []
    for k, token in enumerate(letters):
        a, b = start + boundaries[k], start + boundaries[k + 1]
        spans.append(TokenSpan(token, a, b, float(emission[a:b, token].exp().mean())))
    return spans
