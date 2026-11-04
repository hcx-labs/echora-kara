"""Decoder-margin loss (2026-11-04, off by default: --decoder-margin-weight 0).

The region loss sums the probability of every path that fills each measured fragment, but inference
keeps one path: the production decoder's Viterbi over the song. Loss can fall while a wrong path still
wins that max (clean100k: eval loss 6.36 -> 6.12 from 20k to 24k while decoded mean went 196 -> 214 ms).

This is the structured hinge on that decoder (cost-augmented decoding), per training window:

    loss = max over decoder paths [score(path) + cost(path)] - score(gold)

The decoder is production's (evaluation.decode.decode + decoded_eval._bonus_decode, the decode behind
``decoded_bonus_*``), batched on the GPU:

1. Vocal-activity penalty: letters cost ``activity`` x (1 - loudness activity) per frame; blank is free.
2. Plain pass (no gradient): ordinary CTC Viterbi over the window's fragments.
3. Paced floors from that pass (engine.ctc_segmentation.paced_minimum_frames: line pace x 0.9, at most 2 frames
   a letter and 5 a fragment, pauses over 20 frames left out).
4. Final lattice: those floors, blank-free fragments, ``gap_cost`` per blank frame between two fragments
   of the same line, and the text head's bonus (``bonus`` x log P(token | singing) on each letter).

score is the summed log-probabilities along a path in the final lattice, minus its gap costs. cost is
``scale`` per frame that each fragment's first letter starts away from its measured start, beyond
``tolerance`` frames and capped at ``cap``. gold is the best path that puts every fragment's letters
exactly on its measured frames and blank everywhere else (letter splits free). The gold path is one of
the decoder's paths with zero cost, so the loss is never negative; its gradient raises the gold path
and lowers the path the decoder would pick (into the CTC head, and the text head through the bonus).

Differences from inference, all unavoidable in 25 s windows: activity loudness is scaled to the window
(inference: the song), the plain pass and the line paces see only the window, groups are the measured
fragments rather than whole words (so the gold path stays one of the decoder's paths), and a floor is
lowered where the measured fragment is shorter than it (otherwise no path could reach gold). Frames
outside every fragment, ignored ones included (background masks, overlaps, window-cut fragments), are
blank for gold. Windows whose gold path is impossible (touching fragments repeating a letter) are skipped.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from echora_kara.engine.ctc_segmentation import paced_minimum_frames, vocal_activity

# Finite "impossible" far below any real window score (25 s of frames at -10 nats is -1.25e4); max-plus never
# takes a gradient through it, so unlike the region loss's logaddexp it needs no care beyond that.
_NEG = -1e9
# Production decoder constants (evaluation.decode: PACE, ACTIVITY, LETTER_CAP, GROUP_CAP, PAUSE, VOWELS; the
# inference in-line gap cost and text bonus).
PACE, ACTIVITY, LETTER_CAP, GROUP_CAP, PAUSE = 0.9, 2.0, 2, 5, 20
VOWELS = {1, 5, 9, 15, 21, 25}
GAP_COST, BONUS = 0.2, 0.3


def lattice(groups, floors, lines, blank, *, blank_free, gap_cost):
    """States of align_with_minimum_durations for one window, as lists: label, letter (target index or
    -1), group (or -1), entry (first sub-state of a group's first letter), stay, skip, gap penalty."""
    first_of = {sum(len(g) for g in groups[:k]) for k in range(len(groups))}
    last_of = {f - 1 for f in first_of if f} | {sum(map(len, groups)) - 1}
    group_of = [k for k, g in enumerate(groups) for _ in g]
    in_line_after = (
        {sum(len(g) for g in groups[: k + 1]) - 1 for k in range(len(groups) - 1) if lines[k] == lines[k + 1]}
        if gap_cost > 0
        else set()
    )
    targets = [letter for g in groups for letter in g]
    out = SimpleNamespace(label=[blank], letter=[-1], group=[-1], entry=[False], kind=[0], gap=[0.0])
    for index, (token, frames) in enumerate(zip(targets, floors)):
        for step in range(frames):
            out.label.append(token)
            out.letter.append(index)
            out.group.append(group_of[index])
            out.entry.append(step == 0 and index in first_of)
            out.kind.append(4 if frames == 1 else 1 if step == 0 else 3 if step == frames - 1 else 2)
            out.gap.append(0.0)
        if blank_free and index not in last_of:
            continue
        out.label.append(blank)
        out.letter.append(-1)
        out.group.append(-1)
        out.entry.append(False)
        out.kind.append(0)
        out.gap.append(gap_cost if index in in_line_after else 0.0)
    out.stay = [kind in (0, 3, 4) for kind in out.kind]
    out.skip = [
        kind in (1, 4)
        and s >= 2
        and out.kind[s - 1] == 0
        and out.kind[s - 2] in (3, 4)
        and out.label[s - 2] != out.label[s]
        for s, kind in enumerate(out.kind)
    ]
    return out


def stack(lattices, device):
    """Pad per-window lattices into [R, S] tensors (padding states are blank, never gold, never stay)."""
    states = max(len(x.label) for x in lattices)

    def pad(name, fill, dtype):
        rows = [getattr(x, name) + [fill] * (states - len(x.label)) for x in lattices]
        return torch.tensor(rows, dtype=dtype, device=device)

    return SimpleNamespace(
        label=pad("label", lattices[0].label[0], torch.long),
        letter=pad("letter", -1, torch.long),
        group=pad("group", -2, torch.long),
        entry=pad("entry", False, torch.bool),
        stay=pad("stay", False, torch.bool),
        skip=pad("skip", False, torch.bool),
        gap=pad("gap", 0.0, torch.float32),
        last=torch.tensor([len(x.label) - 1 for x in lattices], device=device),
    )


def viterbi(emissions, lat, frames, *, entry_cost=None, gold_owner=None, gold_blank=None, backtrace=False):
    """Max-score paths through padded lattices. ``emissions`` [R, T, S] per state; ``entry_cost(t)`` [R, S]
    added when a path enters an entry state at frame t; ``gold_owner`` [R, T] restricts row r to states
    whose group equals the frame's (-1 = blank) where ``gold_owner`` is not -3 for that row, and also to blank
    where ``gold_blank`` [R, T] is set. Returns the
    best end score per row, and with ``backtrace`` the state path [R, T] (numpy) as well."""
    rows, width, states = emissions.shape
    dtype = emissions.dtype
    neg = torch.full((rows, 1), _NEG, device=emissions.device, dtype=dtype)
    gap = lat.gap.to(dtype)
    restricted = None if gold_owner is None else (gold_owner[:, :1] != -3)

    def allowed(score, t):
        if gold_owner is None:
            return score
        ok = (gold_owner[:, t : t + 1] == lat.group) | ~restricted
        if gold_blank is not None:
            ok = ok | (gold_blank[:, t : t + 1] & (lat.group == -1))
        return torch.where(ok, score, _NEG)

    score = torch.full((rows, states), _NEG, device=emissions.device, dtype=dtype)
    start = emissions[:, 0, :2]
    if entry_cost is not None:
        start = start + torch.where(lat.entry[:, :2], entry_cost(0)[:, :2], 0)
    score = allowed(torch.cat((start, score[:, 2:]), 1), 0)
    moves = [] if backtrace else None
    for t in range(1, width):
        stay = torch.where(lat.stay, score, _NEG)
        advance = torch.cat((neg, score[:, :-1]), 1)
        jump = torch.where(lat.skip, torch.cat((neg, neg, score[:, :-2]), 1), _NEG)
        if entry_cost is not None:
            cost = torch.where(lat.entry, entry_cost(t), 0)
            advance, jump = advance + cost, jump + cost
        best, move = torch.stack((stay, advance, jump)).max(0)
        step = allowed(best + emissions[:, t] - gap, t)
        live = (t < frames).unsqueeze(1)
        score = torch.where(live, step, score)
        if backtrace:
            moves.append(torch.where(live, move, 0).to(torch.uint8))
    ends = torch.stack((score.gather(1, lat.last[:, None]), score.gather(1, (lat.last - 1)[:, None])), 1).squeeze(2)
    best, which = ends.max(1)
    if not backtrace:
        return best
    moves = torch.stack(moves, 1).cpu().numpy() if moves else np.zeros((rows, 0, states), np.uint8)
    state = (lat.last - which).cpu().numpy()
    path = np.zeros((rows, width), dtype=np.int64)
    counts = frames.cpu().numpy()
    for r in range(rows):
        s = int(state[r])
        path[r, counts[r] - 1] = s
        for t in range(counts[r] - 1, 0, -1):
            s -= int(moves[r, t - 1, s])
            path[r, t - 1] = s
    return best, path


def window_floors(groups, lines, path, letter_of_state, frames):
    """Production's paced floors from one window's plain path, then lowered to fit each measured fragment."""
    spans = [[None] * len(g) for g in groups]
    flat = [(k, i) for k, g in enumerate(groups) for i in range(len(g))]
    for t in range(frames):
        index = letter_of_state[path[t]]
        if index >= 0:
            k, i = flat[index]
            spans[k][i] = (spans[k][i][0], t + 1) if spans[k][i] else (t, t + 1)
    spans = [[SimpleNamespace(start=s[0], end=s[1]) for s in g] for g in spans]
    ranges, start = [], 0
    for k in range(1, len(groups) + 1):
        if k == len(groups) or lines[k] != lines[start]:
            ranges.append((start, k))
            start = k
    return paced_minimum_frames(
        groups,
        spans,
        ranges,
        ratio=PACE,
        letter_cap=LETTER_CAP,
        group_cap=GROUP_CAP,
        pause_frames=PAUSE,
        vowel_ids=VOWELS,
    )


def fit(floors, groups, regions):
    """Lower floors (largest first) until each fragment's letters fit its measured frames."""
    out, offset = list(floors), 0
    for g, (first, stop, _) in zip(groups, regions):
        part = out[offset : offset + len(g)]
        while sum(part) > stop - first:
            part[part.index(max(part))] -= 1
        out[offset : offset + len(g)] = part
        offset += len(g)
    return out


def gold_bounds(found):
    """Frames each fragment's letters fill on the gold path. CTC needs a blank between two letters that repeat,
    so where touching fragments repeat a letter (4.5 % of fragment pairs, 60 % of windows) the decoder can't
    start the second on its measured frame either: one junction frame goes to blank. These bounds take it
    from the second fragment (else the first) where it has a frame to spare, so the floors are fitted to a
    gold path that exists; the gold pass may take it from either side (a 1-frame shift, within tolerance).
    None when no gold path exists (overlapping fragments, or no spare frame)."""
    bounds = [[first, stop] for first, stop, *_ in found]
    for k in range(1, len(found)):
        before, after = bounds[k - 1], bounds[k]
        if before[1] > after[0]:
            return None
        if before[1] == after[0] and found[k - 1][2][-1] == found[k][2][0]:
            if after[1] - after[0] > len(found[k][2]):
                after[0] += 1
            elif before[1] - before[0] > len(found[k - 1][2]):
                before[1] -= 1
            else:
                return None
    return [tuple(x) for x in bounds]


def decoder_margin_loss(
    log_probs,
    ctc_regions,
    ctc_targets,
    frame_counts,
    *,
    blank,
    region_lines=None,
    region_columns=None,
    token_log_probs=None,
    waveforms=None,
    sample_counts=None,
    scale=0.1,
    tolerance=1,
    cap=25,
    activity=ACTIVITY,
    gap_cost=GAP_COST,
    bonus=BONUS,
):
    """Hinge per valid frame over the batch, and stats.

    ``log_probs`` [B, T, V] letter log-probabilities; ``ctc_regions`` rows (first, stop, letters), letters 0
    for padding; ``ctc_targets`` their letters; ``frame_counts`` [B] valid frames. ``region_lines`` [B, R]
    each fragment's lyric line (None: one line, no gap cost); ``region_columns`` [B, R, L] the text-head
    lead-token column of each letter (-1: none) and ``token_log_probs`` [B, T, N] that head's log P(token |
    singing) (None: no bonus); ``waveforms`` [B, samples] with ``sample_counts`` [B] for the activity
    penalty (None: no penalty)."""
    rows = []
    for b in range(log_probs.shape[0]):
        found = []
        for column, (first, stop, length) in enumerate(ctc_regions[b].tolist()):
            if length > 0:
                line = int(region_lines[b, column]) if region_lines is not None else 0
                cols = region_columns[b, column, :length].tolist() if region_columns is not None else [-1] * length
                found.append((first, stop, ctc_targets[b, column, :length].tolist(), line, cols))
        found.sort()
        bounds = gold_bounds(found)
        if found and bounds is not None:
            rows.append((b, [(*f, *g) for f, g in zip(found, bounds)]))
    zero = log_probs.sum() * 0
    if not rows:
        return zero, {"windows": torch.tensor(0.0), "per_window": torch.tensor(0.0)}
    device = log_probs.device
    index = torch.tensor([b for b, _ in rows], device=device)
    frames = frame_counts.to(device)[index]
    width = int(frames.max())
    groups = [[f[2] for f in found] for _, found in rows]
    lines = [[f[3] for f in found] for _, found in rows]
    letters = log_probs[index, :width].float()
    if waveforms is not None and activity:
        with torch.no_grad():
            penalty = torch.zeros(len(rows), width, device=device)
            for r, (b, _) in enumerate(rows):
                n = int(frames[r])
                wave = waveforms[b, : int(sample_counts[b])].float()
                penalty[r, :n] = activity * (1 - vocal_activity(wave, n))
        not_blank = torch.ones(letters.shape[-1], device=device)
        not_blank[blank] = 0
        letters = letters - penalty.unsqueeze(-1) * not_blank

    # Plain pass (no gradient) -> paced floors, lowered to fit each fragment.
    with torch.no_grad():
        plain = [
            lattice(g, [1] * sum(map(len, g)), l, blank, blank_free=False, gap_cost=0.0) for g, l in zip(groups, lines)
        ]
        plain_lat = stack(plain, device)
        plain_em = letters.gather(2, plain_lat.label.unsqueeze(1).expand(-1, width, -1))
        _, path = viterbi(plain_em, plain_lat, frames, backtrace=True)
        floors = []
        for r, (g, l) in enumerate(zip(groups, lines)):
            regions = [(f[5], f[6], f[2]) for f in rows[r][1]]
            floors.append(fit(window_floors(g, l, path[r], plain[r].letter, int(frames[r])), g, regions))

    final = [lattice(g, f, l, blank, blank_free=True, gap_cost=gap_cost) for g, f, l in zip(groups, floors, lines)]
    lat = stack(final, device)
    emissions = letters.gather(2, lat.label.unsqueeze(1).expand(-1, width, -1))
    if token_log_probs is not None and bonus:
        columns = []
        for (b, found), lat_row in zip(rows, final):
            flat = [c for f in found for c in f[4]]
            columns.append(
                [flat[i] if i >= 0 else -1 for i in lat_row.letter] + [-1] * (lat.label.shape[1] - len(lat_row.letter))
            )
        columns = torch.tensor(columns, device=device)
        tokens = token_log_probs[index, :width].float()
        share = tokens.gather(2, columns.clamp_min(0).unsqueeze(1).expand(-1, width, -1))
        emissions = emissions + bonus * torch.where(columns.unsqueeze(1) >= 0, share, 0)

    # One batched pass: rows 0..R-1 cost-augmented and free, rows R..2R-1 gold and cost-free.
    count = len(rows)
    owner = torch.full((count, width), -1, dtype=torch.long)
    junction = torch.zeros((count, width), dtype=torch.bool)
    for r, (_, found) in enumerate(rows):
        for k, f in enumerate(found):
            owner[r, f[0] : f[1]] = k
            if (f[5], f[6]) != (f[0], f[1]):  # a repeated-letter junction: blank may take either side
                junction[r, f[0] if f[5] != f[0] else f[1] - 1] = True
                junction[r, f[0] - 1 if f[5] != f[0] else f[1]] = True
    owner = torch.cat((torch.full_like(owner, -3), owner)).to(device)
    junction = torch.cat((torch.zeros_like(junction), junction)).to(device)
    entry = torch.full((count, lat.label.shape[1]), -1, dtype=torch.long)
    for r, (lat_row, (_, found)) in enumerate(zip(final, rows)):
        for s, (is_entry, group) in enumerate(zip(lat_row.entry, lat_row.group)):
            if is_entry:
                entry[r, s] = found[group][0]
    entry = entry.to(device)
    weight = torch.cat((torch.full((count, 1), float(scale)), torch.zeros(count, 1))).to(device)
    both = SimpleNamespace(**{k: torch.cat((v, v)) for k, v in vars(lat).items()})

    def entry_cost(t):
        miss = ((t - entry).abs() - tolerance).clamp(0, cap).float()
        return weight * torch.cat((miss, miss))

    scores = viterbi(
        torch.cat((emissions, emissions)),
        both,
        torch.cat((frames, frames)),
        entry_cost=entry_cost,
        gold_owner=owner,
        gold_blank=junction,
    )
    augmented, gold = scores[:count], scores[count:]
    feasible = gold > _NEG / 2
    hinge = torch.where(feasible, (augmented - gold).clamp_min(0), torch.zeros_like(gold))
    used = (frames * feasible).sum().clamp_min(1)
    stats = {
        "windows": feasible.sum().detach().float(),
        "per_window": (hinge.sum() / feasible.sum().clamp_min(1)).detach().float(),
    }
    return hinge.sum() / used, stats
