#!/usr/bin/env python3
"""Reference-free label-quality scores for timed lyrics on full mixes (CPU only).

Follows the idea of PCMI/WACS (Akavarapu, Daniel, Jäger 2026, arXiv 2608.28508): alignment
quality can be judged without gold timestamps by how strongly the aligned labels predict
unsupervised acoustic clusters (PCMI) and how consistently repeated words sound alike (WACS).
That paper clusters MMS/XLSR layer-15 features; a full-collection pass of a large encoder is costly
on GPU, so this module uses 39-dim MFCC+Δ frames on the 20 ms label clock and a
per-song k-means instead. Scores are therefore comparable *within* this build (same features
everywhere), not with the paper's absolute numbers. Swapping the feature function for cached
encoder states keeps everything else unchanged (``frame_features``).

Per song:
- ``pcmi``: normalised mutual information between the letter at each labelled frame (the
  proportional within-fragment estimate the trainer also uses) and the frame's k-means cluster,
  over labelled frames. Wrong timings decorrelate letters and sound: the score drops.
- ``wacs``: for label strings that recur in the song (same unit text, at least two instances),
  the DTW distance between the instances' MFCC sequences, divided by the mean distance to
  random units of similar length. Below 1 means repeated words sound alike; near 1 means the
  timing does not pick out the word.
- ``activity_auc``: how well vocal-band energy separates labelled frames from unlabelled frames
  inside the sung part of the song (AUC); ~0.5 means the timing is not where the singing is.
- ``score``: the three combined (mean of pcmi, 1-min(wacs,1), 2*(auc-0.5)) clipped to [0, 1].
Per line: ``lines[i]`` is the mean per-frame PMI of that line's labelled frames under the
song's joint table, standardised across the song's lines (negative = the line fits worse than
the song's other lines; a shifted or wrong line shows up here).

Validation of the scores themselves is the ``perturb`` command: it re-scores the same songs with
every timing shifted by +0.5 s / +1 s and with lines shuffled, and prints how far the scores
fall. The manifest builder stores the raw scores and never filters by default
(``--min-label-quality`` is opt-in).

Whole-song offsets (``label_quality_mfcc_v2``): the lead-label
activity indicator is cross-correlated with the smoothed vocal-band energy over lags of ±45 s on
the 20 ms clock. A song is *flagged* when the best lag is at least 0.3 s away from zero and the
correlation there exceeds the zero-lag correlation by 8 %, and *confirmed* when ``activity_auc``
recomputed with the labels shifted by that lag rises by at least 0.05 (an independent cue; a
wrong shift lowers it). ``offset_s`` is positive when the audio is later than the labels. The
manifest builder drops confirmed songs with ``--offset-gate drop`` and records the rest.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from scipy.fft import dct

VERSION = "label_quality_mfcc_v2"
OFFSET_MAX_LAG_SECONDS = 45.0
OFFSET_MIN_LAG_SECONDS = 0.3
OFFSET_MIN_RATIO = 1.08
OFFSET_MIN_AUC_GAIN = 0.05
SAMPLE_RATE = 16000
FRAME = 0.02  # label clock
WINDOW = 400  # samples, as the MMS receptive field
HOP = 320
N_MELS = 40
N_MFCC = 13
CLUSTERS = 32
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def decode(path: Path) -> np.ndarray:
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path), "-vn",
               "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def _mel_filterbank(n_fft: int) -> np.ndarray:
    def hz_to_mel(f): return 2595 * np.log10(1 + f / 700)
    def mel_to_hz(m): return 700 * (10 ** (m / 2595) - 1)
    mels = np.linspace(hz_to_mel(50), hz_to_mel(SAMPLE_RATE / 2), N_MELS + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mels) / SAMPLE_RATE).astype(int)
    bank = np.zeros((N_MELS, n_fft // 2 + 1))
    for m in range(1, N_MELS + 1):
        left, centre, right = bins[m - 1], bins[m], bins[m + 1]
        for k in range(left, centre):
            bank[m - 1, k] = (k - left) / max(centre - left, 1)
        for k in range(centre, right):
            bank[m - 1, k] = (right - k) / max(right - centre, 1)
    return bank


def frame_features(waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(frames × 39 MFCC+Δ+ΔΔ, frames vocal-band log energy) on the 20 ms label clock.

    Frame t covers samples [t*320, t*320+400), the same centres as the trainer's labels.
    """
    n_fft = 512
    count = max(0, (len(waveform) - WINDOW) // HOP + 1)
    if count < 10:
        return np.zeros((0, 3 * N_MFCC)), np.zeros(0)
    index = np.arange(WINDOW)[None, :] + HOP * np.arange(count)[:, None]
    frames = waveform[index] * np.hanning(WINDOW)[None, :]
    spectrum = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2
    bank = _mel_filterbank(n_fft)
    mel = np.log(spectrum @ bank.T + 1e-10)
    mfcc = dct(mel, type=2, axis=1, norm="ortho")[:, :N_MFCC]
    delta = np.gradient(mfcc, axis=0)
    features = np.concatenate([mfcc, delta, np.gradient(delta, axis=0)], axis=1)
    features = (features - features.mean(0)) / (features.std(0) + 1e-6)
    freqs = np.fft.rfftfreq(n_fft, 1 / SAMPLE_RATE)
    band = (freqs >= 300) & (freqs <= 3400)
    energy = np.log(spectrum[:, band].sum(1) + 1e-10)
    return features.astype(np.float32), energy.astype(np.float32)


def kmeans(features: np.ndarray, k: int, seed: int = 0, iterations: int = 15) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = min(k, len(features))
    centres = features[rng.choice(len(features), k, replace=False)]
    labels = np.zeros(len(features), dtype=int)
    for _ in range(iterations):
        distances = ((features[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        labels = distances.argmin(1)
        for cluster in range(k):
            members = features[labels == cluster]
            if len(members):
                centres[cluster] = members.mean(0)
    return labels


def frame_letters(units: list[dict], frames: int) -> np.ndarray:
    """Letter index (0-25) per frame from the proportional within-unit estimate, -1 elsewhere."""
    letters = np.full(frames, -1, dtype=int)
    for unit in units:
        text = unit.get("text") or ""
        if unit.get("ignore") or not text:
            continue
        first, last = int(round(unit["start"] / FRAME)), int(round(unit["end"] / FRAME))
        first, last = max(0, first), min(frames, max(last, first + 1))
        if last <= first:
            continue
        positions = np.arange(first, last)
        letter = ((positions - first) * len(text) // max(last - first, 1)).clip(0, len(text) - 1)
        letters[first:last] = [LETTERS.index(text[i]) for i in letter]
    return letters


def mutual_information(a: np.ndarray, b: np.ndarray, na: int = len(LETTERS),
                       nb: int = CLUSTERS) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Normalised MI of two label arrays plus the full PMI table (for per-line scores)."""
    joint = np.zeros((na, nb))
    np.add.at(joint, (a, b), 1)
    joint /= joint.sum()
    pa, pb = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.where(joint > 0, np.log(joint / (pa @ pb)), 0.0)
    mi = float((joint * pmi).sum())
    ha = float(-(pa[pa > 0] * np.log(pa[pa > 0])).sum())
    hb = float(-(pb[pb > 0] * np.log(pb[pb > 0])).sum())
    return mi / math.sqrt(ha * hb) if ha > 0 and hb > 0 else 0.0, pmi, pa, pb


def dtw(a: np.ndarray, b: np.ndarray) -> float:
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        row = acc[i - 1]
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(row[j], row[j - 1], acc[i, j - 1])
    return float(acc[n, m] / (n + m))


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    if not len(positive) or not len(negative):
        return 0.5
    ranks = np.argsort(np.argsort(np.concatenate([positive, negative]))) + 1
    return float((ranks[:len(positive)].sum() - len(positive) * (len(positive) + 1) / 2)
                 / (len(positive) * len(negative)))


def score_song(features: np.ndarray, energy: np.ndarray, lines: list[list[dict]],
               rng_seed: int = 0, max_pairs: int = 60) -> dict:
    """All scores for one song from its frame features and its per-line timed units."""
    frames = len(features)
    units = [unit for line in lines for unit in line if unit.get("vocal_role") != "background"]
    letters = frame_letters(units, frames)
    labelled = letters >= 0
    result = {"method": VERSION, "frames": int(frames), "labelled_frames": int(labelled.sum())}
    if labelled.sum() < 200:
        return {**result, "pcmi": None, "wacs": None, "activity_auc": None, "score": None, "lines": []}
    clusters = kmeans(features, CLUSTERS, seed=rng_seed)
    nmi, pmi, _, _ = mutual_information(letters[labelled], clusters[labelled])
    # Activity: labelled frames vs unlabelled frames between the first and last labelled frame.
    first, last = np.flatnonzero(labelled)[[0, -1]]
    inside = np.zeros(frames, dtype=bool)
    inside[first:last + 1] = True
    activity = auc(energy[labelled], energy[inside & ~labelled])
    # WACS: repeated unit texts (lead units only), DTW against each other and against random units.
    rng = np.random.default_rng(rng_seed)
    by_text: dict[str, list[dict]] = {}
    usable = [u for u in units if u.get("text") and not u.get("ignore") and (u["end"] - u["start"]) >= 0.12]
    for unit in usable:
        by_text.setdefault(unit["text"], []).append(unit)
    pairs = [(group[i], group[j]) for group in by_text.values() if len(group) > 1
             for i in range(len(group)) for j in range(i + 1, len(group))]
    rng.shuffle(pairs)
    same, other = [], []

    def clip(unit):
        return features[int(round(unit["start"] / FRAME)):max(int(round(unit["end"] / FRAME)), int(round(unit["start"] / FRAME)) + 1)]
    for a, b in pairs[:max_pairs]:
        same.append(dtw(clip(a), clip(b)))
        candidates = [u for u in usable if u["text"] != a["text"] and abs((u["end"] - u["start"]) - (a["end"] - a["start"])) < 0.15]
        if candidates:
            other.append(dtw(clip(a), clip(candidates[rng.integers(len(candidates))])))
    wacs = float(np.mean(same) / np.mean(other)) if same and other and np.mean(other) > 0 else None
    # Per line: mean PMI of the line's frames, standardised over the song's lines.
    line_scores = []
    for line in lines:
        line_letters = frame_letters([u for u in line if u.get("vocal_role") != "background"], frames)
        mask = line_letters >= 0
        line_scores.append(float(pmi[line_letters[mask], clusters[mask]].mean()) if mask.sum() >= 5 else None)
    values = np.array([v for v in line_scores if v is not None])
    if len(values) >= 3 and values.std() > 0:
        line_scores = [None if v is None else round(float((v - values.mean()) / values.std()), 3) for v in line_scores]
    else:
        line_scores = [None if v is None else 0.0 for v in line_scores]
    parts = [nmi, (1 - min(wacs, 1.0)) if wacs is not None else None, 2 * (activity - 0.5)]
    parts = [p for p in parts if p is not None]
    return {**result, "pcmi": round(nmi, 4), "wacs": round(wacs, 4) if wacs is not None else None,
            "activity_auc": round(activity, 4), "repeated_pairs": len(same),
            "score": round(float(min(1.0, max(0.0, float(np.mean(parts))))), 4), "lines": line_scores}


def lead_indicator(lines: list[list[dict]], frames: int) -> np.ndarray:
    """1 on frames inside a supervised lead unit (the activity indicator), 0 elsewhere."""
    indicator = np.zeros(frames, dtype=np.float32)
    for line in lines:
        for unit in line:
            if unit.get("ignore") or unit.get("vocal_role") == "background" or not unit.get("text"):
                continue
            first, stop = int(unit["start"] / FRAME), int(unit["end"] / FRAME) + 1
            indicator[max(0, first):min(frames, stop)] = 1.0
    return indicator


def activity_auc_of(energy: np.ndarray, indicator: np.ndarray) -> float | None:
    """Vocal-band energy AUC of labelled vs unlabelled frames inside the labelled span."""
    labelled = np.flatnonzero(indicator > 0)
    if len(labelled) < 100:
        return None
    inside = np.zeros(len(energy), dtype=bool)
    inside[labelled[0]:labelled[-1] + 1] = True
    return auc(energy[indicator > 0], energy[inside & (indicator == 0)])


def shift_indicator(indicator: np.ndarray, frames: int) -> np.ndarray:
    shifted = np.zeros_like(indicator)
    if frames >= 0:
        shifted[frames:] = indicator[:len(indicator) - frames]
    else:
        shifted[:len(indicator) + frames] = indicator[-frames:]
    return shifted


def offset_search(energy: np.ndarray, lines: list[list[dict]], max_lag_seconds: float = OFFSET_MAX_LAG_SECONDS) -> dict:
    """Whole-song lag between the lead labels and the vocal-band energy.

    Positive ``offset_s``: the audio is later than the labels. ``offset_flagged`` follows the
    cross-correlation peak alone; ``offset_confirmed`` additionally needs the activity AUC to
    rise by ``OFFSET_MIN_AUC_GAIN`` when the labels are shifted by the lag.
    """
    frames = len(energy)
    indicator = lead_indicator(lines, frames)
    empty = {"offset_s": None, "offset_ratio": None, "offset_corr0": None, "offset_auc0": None,
             "offset_auc_at": None, "offset_flagged": False, "offset_confirmed": False}
    if indicator.sum() < 100 or frames < 200:
        return empty
    kernel = np.ones(10) / 10  # 200 ms smoothing
    smooth = np.convolve(energy, kernel, mode="same")
    smooth = (smooth - smooth.mean()) / (smooth.std() + 1e-6)
    centred = (indicator - indicator.mean()) / (indicator.std() + 1e-6)
    max_lag = min(int(max_lag_seconds / FRAME), frames - 1)
    size = 1 << int(np.ceil(np.log2(2 * frames)))
    correlation = np.fft.irfft(np.fft.rfft(smooth, size) * np.conj(np.fft.rfft(centred, size)), size)
    correlation = np.concatenate([correlation[-max_lag:], correlation[:max_lag + 1]])
    lags = np.arange(-max_lag, max_lag + 1)
    correlation = correlation / np.maximum(frames - np.abs(lags), 1)
    zero = max_lag
    best = int(np.argmax(correlation))
    corr0, corr_best = float(correlation[zero]), float(correlation[best])
    lag_frames = int(lags[best])
    ratio = corr_best / corr0 if corr0 > 0 else float("inf") if corr_best > 0 else None
    flagged = abs(lag_frames) * FRAME >= OFFSET_MIN_LAG_SECONDS and ratio is not None and ratio >= OFFSET_MIN_RATIO
    auc0 = activity_auc_of(energy, indicator)
    auc_at = activity_auc_of(energy, shift_indicator(indicator, lag_frames)) if flagged else None
    confirmed = bool(flagged and auc0 is not None and auc_at is not None and auc_at - auc0 >= OFFSET_MIN_AUC_GAIN)
    return {"offset_s": round(lag_frames * FRAME, 2), "offset_ratio": round(ratio, 3) if ratio not in (None, float("inf")) else ratio,
            "offset_corr0": round(corr0, 4), "offset_auc0": round(auc0, 3) if auc0 is not None else None,
            "offset_auc_at": round(auc_at, 3) if auc_at is not None else None,
            "offset_flagged": bool(flagged), "offset_confirmed": confirmed}


def timings_digest(lines: list[list[dict]]) -> str:
    """Content key of the timed units the scores depend on (lead and background letters and
    masks), so a cached score is reused whenever the labels are byte-identical, whatever the
    normalizer or builder version string says."""
    import hashlib
    parts = [f"{u['start']:.3f},{u['end']:.3f},{u.get('text') or ''},{int(bool(u.get('ignore')))},{u.get('vocal_role') or 'lead'}"
             for line in lines for u in line]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]


def score_file(path: Path, lines: list[list[dict]]) -> dict:
    features, energy = frame_features(decode(path))
    return {**score_song(features, energy, lines), **offset_search(energy, lines)}


# ---------------------------------------------------------------------------------------------
# Metric validation: perturbations should lower the scores.

def shifted(lines: list[list[dict]], seconds: float) -> list[list[dict]]:
    return [[{**u, "start": u["start"] + seconds, "end": u["end"] + seconds} for u in line] for line in lines]


def shuffled_lines(lines: list[list[dict]], seed: int = 0) -> list[list[dict]]:
    """Keep every line's time slot but move another line's labels into it."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(lines))
    result = []
    for slot, source in zip(lines, (lines[i] for i in order)):
        if not slot or not source:
            result.append(slot)
            continue
        slot_start, slot_end = min(u["start"] for u in slot), max(u["end"] for u in slot)
        src_start, src_end = min(u["start"] for u in source), max(u["end"] for u in source)
        scale = (slot_end - slot_start) / max(src_end - src_start, 1e-3)
        result.append([{**u, "start": slot_start + (u["start"] - src_start) * scale,
                        "end": slot_start + (u["end"] - src_start) * scale} for u in source])
    return result


def perturb(args) -> None:
    rows = [json.loads(line) for line in Path(args.manifest).open() if line.strip()][:args.songs]
    root = Path(args.root)
    table = {"original": [], "shift_0.5s": [], "shift_1s": [], "shuffled_lines": []}
    for row in rows:
        features, energy = frame_features(decode(root / row["audio"]))
        lines = row["timings"]
        for name, variant in (("original", lines), ("shift_0.5s", shifted(lines, 0.5)),
                              ("shift_1s", shifted(lines, 1.0)), ("shuffled_lines", shuffled_lines(lines))):
            table[name].append(score_song(features, energy, variant))
    print(f"{'variant':15s} {'pcmi':>7s} {'wacs':>7s} {'auc':>7s} {'score':>7s}   (means over {len(rows)} songs)")
    for name, scores in table.items():
        def mean(key):
            values = [s[key] for s in scores if s.get(key) is not None]
            return f"{np.mean(values):7.3f}" if values else "    n/a"
        print(f"{name:15s} {mean('pcmi')} {mean('wacs')} {mean('activity_auc')} {mean('score')}")
    original = np.array([s["score"] for s in table["original"] if s["score"] is not None])
    worst = min(np.array([s["score"] for s in table[k] if s["score"] is not None]).mean() for k in table if k != "original")
    print(f"original score percentiles 5/25/50: {np.percentile(original, [5, 25, 50]).round(3).tolist()}; "
          f"worst perturbed mean {worst:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("perturb", help="score sampled manifest songs under timing perturbations")
    p.add_argument("manifest"); p.add_argument("--root", required=True); p.add_argument("--songs", type=int, default=30)
    p.set_defaults(run=perturb)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
