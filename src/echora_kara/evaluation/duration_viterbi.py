"""Decoder A: CTC forced alignment with a minimum duration per letter.

The plain CTC Viterbi (engine/ctc_segmentation.py) lets every letter take a single 20 ms frame,
so when the model's scores are flat the best path crams words ("is a" flashing past) or holds the
first of a run of repeated words and squeezes the rest. Here each letter becomes a chain of k states
the path must walk through, one frame each, before its last state may hold: every letter lasts at
least k frames. Blanks are unchanged. k = 1 is the ordinary CTC topology (checked against
torchaudio's forced_align).

Scored on the 80 test songs' vocal stems with cached emissions (as checkpoint_eval.py): word-start error,
words crammed under 60 ms, interjection errors, and runs of 3+ repeated interjections whose first
word is held while the rest are crammed.

usage: python -m echora_kara.evaluation.duration_viterbi --emissions stem_emissions --k 1 2 3 4
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

from echora_kara.evaluation import interval_decoder as proto
from echora_kara.evaluation.shuffle_align import song_words

INTERJECTION = re.compile(r"^(o+h*|o+o+h*|a+h+|whoa+|wo+a+h*|yeah|na|la|hey|uh|ooh+|mm+)$")


def min_duration_viterbi(emission, ids, k, device="cpu"):
    """Best CTC path where letter i occupies at least k[i] frames (k may be one int for all).
    Returns (start, end) frames per letter."""
    T, n = emission.shape[0], len(ids)
    ks = [k] * n if isinstance(k, int) else list(k)
    # States: blank, then for each letter k sub-states and a blank.
    labels, kind, letter_of = [proto.BLANK], [0], [-1]  # kind 0 blank, 1 first sub-state, 2 inner, 3 last
    for i, token in enumerate(ids):
        k = ks[i]
        for j in range(k):
            labels.append(token)
            kind.append(1 if j == 0 else 2)
            letter_of.append(i)
        if k == 1:
            kind[-1] = 4  # first and last at once
        else:
            kind[-1] = 3
        labels.append(proto.BLANK)
        kind.append(0)
        letter_of.append(-1)
    S = len(labels)
    labels_t = torch.tensor(labels, device=device)
    kind_t = torch.tensor(kind, device=device)
    em = torch.from_numpy(emission).to(device)
    can_stay = (kind_t == 0) | (kind_t == 3) | (kind_t == 4)
    # Skip a blank into a letter's first sub-state from the previous letter's last sub-state
    # (offset 2) when the two letters differ.
    can_skip = torch.zeros(S, dtype=torch.bool, device=device)
    first_states = [s for s in range(S) if kind[s] in (1, 4)]
    for s in first_states:
        if s - 2 >= 0 and kind[s - 2] in (3, 4) and labels[s - 2] != labels[s]:
            can_skip[s] = True
    neg = torch.tensor(float("-inf"), device=device)
    score = torch.full((S,), float("-inf"), device=device)
    score[0] = em[0, proto.BLANK]
    score[1] = em[0, labels[1]]
    back = torch.zeros((T, S), dtype=torch.uint8, device=device)
    for t in range(1, T):
        stay = torch.where(can_stay, score, neg)
        advance = torch.cat((neg.reshape(1), score[:-1]))
        skip = torch.where(can_skip, torch.cat((neg.repeat(2), score[:-2])), neg)
        best, move = torch.stack((stay, advance, skip)).max(0)
        score = best + em[t].index_select(0, labels_t)
        back[t] = move.to(torch.uint8)
    final = [S - 1, S - 2]
    state = max(final, key=lambda s: float(score[s]))
    if not np.isfinite(float(score[state])):
        raise ValueError("no path: too many letters for the frames")
    back = back.cpu().numpy()
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = state
        if t:
            state -= int(back[t, state])
    letter = np.array(letter_of)[path]
    spans = []
    for i in range(n):
        frames = np.nonzero(letter == i)[0]
        spans.append((int(frames[0]), int(frames[-1]) + 1))
    return spans


VOWELS = {proto_id for proto_id in (1, 5, 9, 15, 21, 25)}  # a e i o u y in the model vocabulary


def word_floor(words, frames, letter_frames=1):
    """Per-letter minimum frames so every word lasts at least `frames` (each letter at least
    `letter_frames`); extra frames go to vowels."""
    out = []
    for w in words:
        ks = [letter_frames] * len(w["ids"])
        extra = frames - sum(ks)
        slots = [i for i, t in enumerate(w["ids"]) if t in VOWELS] or [0]
        j = 0
        while extra > 0:
            ks[slots[j % len(slots)]] += 1
            extra -= 1
            j += 1
        out += ks
    return out


def adaptive_floor(words, plain_spans, ratio=0.6, token_cap_ms=100, letter_cap_ms=40):
    """Per-letter minimum frames scaled to each line's own pace in the plain pass: a word's floor is
    ratio x (line span / words in line), capped at token_cap_ms and at least one frame; letters get
    letter_cap_ms only on lines whose word floor reaches 80 ms (fast rap keeps one-frame letters)."""
    lines = {}
    for w, (s, e) in zip(words, plain_spans):
        lines.setdefault(w["line"], []).append((s, e))
    pace = {}
    for line, spans in lines.items():
        span_frames = max(e for _, e in spans) - min(s for s, _ in spans)
        pace[line] = span_frames / max(1, len(spans))
    out = []
    for w in words:
        floor = max(1, min(round(token_cap_ms / 20), int(ratio * pace[w["line"]])))
        letter = round(letter_cap_ms / 20) if floor * 20 >= 80 else 1
        out += word_floor([w], floor, letter)
    return out


def onset_floor(words, plain_spans, ratio=0.7, token_cap_ms=100, letter_cap_ms=40):
    """Like adaptive_floor, but a line's pace is the median gap between consecutive word starts in
    the plain pass: pauses and single held words do not count, a fast rap line still reads fast."""
    starts = {}
    for w, (s, _) in zip(words, plain_spans):
        starts.setdefault(w["line"], []).append(s)
    pace = {}
    for line, values in starts.items():
        gaps = [b - a for a, b in zip(values, values[1:]) if b > a]
        pace[line] = float(np.median(gaps)) if gaps else 1e9
    out = []
    for w in words:
        floor = max(1, min(round(token_cap_ms / 20), int(ratio * pace[w["line"]])))
        letter = round(letter_cap_ms / 20) if floor * 20 >= 80 else 1
        out += word_floor([w], floor, letter)
    return out


def sung_floor(words, plain_spans, ratio=0.8, pause_frames=20, token_cap_ms=100, letter_cap_ms=40):
    """Floor from each line's sung pace: (line span minus pauses longer than pause_frames between
    words) / words, as measured in the plain pass. Cramming can't shrink a line's span, and pauses
    no longer inflate it."""
    lines = {}
    for w, span in zip(words, plain_spans):
        lines.setdefault(w["line"], []).append(span)
    pace = {}
    for line, spans in lines.items():
        spans = sorted(spans)
        total = max(e for _, e in spans) - spans[0][0]
        pauses = sum(max(0, b[0] - a[1]) for a, b in zip(spans, spans[1:]) if b[0] - a[1] > pause_frames)
        pace[line] = max(1.0, total - pauses) / len(spans)
    out = []
    for w in words:
        floor = max(1, min(round(token_cap_ms / 20), int(ratio * pace[w["line"]])))
        letter = round(letter_cap_ms / 20) if floor * 20 >= 80 else 1
        out += word_floor([w], floor, letter)
    return out


def sung_floor2(words, plain_spans, ratio=0.9, pause_frames=20, token_cap_ms=100, letter_cap_ms=40):
    """sung_floor with the letter floor paced too: letters get letter_cap_ms only where ratio x the
    line's sung frames per letter reaches it, so fast rap keeps one-frame letters."""
    lines = {}
    for w, span in zip(words, plain_spans):
        lines.setdefault(w["line"], []).append((span, len(w["ids"])))
    token_pace, letter_pace = {}, {}
    for line, items in lines.items():
        spans = sorted(span for span, _ in items)
        total = max(e for _, e in spans) - spans[0][0]
        pauses = sum(max(0, b[0] - a[1]) for a, b in zip(spans, spans[1:]) if b[0] - a[1] > pause_frames)
        sung = max(1.0, total - pauses)
        token_pace[line] = sung / len(items)
        letter_pace[line] = sung / max(1, sum(n for _, n in items))
    letter_cap = round(letter_cap_ms / 20)
    out = []
    for w in words:
        floor = max(1, min(round(token_cap_ms / 20), int(ratio * token_pace[w["line"]])))
        letter = max(1, min(letter_cap, int(ratio * letter_pace[w["line"]])))
        out += word_floor([w], floor, letter)
    return out


def vocal_activity(stem, frames):
    """Per-frame singing activity in [0, 1] from the vocal stem: RMS over each frame's receptive field
    (400 samples centred at 320 k + 200), in dB, scaled between the song's 10th and 90th percentile."""
    out = np.zeros(frames, dtype=np.float32)
    for k in range(frames):
        piece = stem[k * 320:k * 320 + 400]
        out[k] = 10 * np.log10(np.mean(piece ** 2) + 1e-10) if piece.size else -100
    low, high = np.percentile(out, 10), np.percentile(out, 90)
    return np.clip((out - low) / max(1e-6, high - low), 0, 1)


def with_activity(emission, activity, strength):
    """Letters cost `strength` x (1 - activity) extra per frame; blanks are unchanged."""
    adjusted = emission.copy()
    penalty = strength * (1 - activity)
    mask = np.ones(emission.shape[1], dtype=bool)
    mask[proto.BLANK] = False
    adjusted[:, mask] -= penalty[:, None]
    return adjusted


def word_spans(words, letter_spans):
    out, cursor = [], 0
    for w in words:
        own = letter_spans[cursor:cursor + len(w["ids"])]
        cursor += len(w["ids"])
        out.append((own[0][0], own[-1][1]))
    return out


def crammed_and_held(words, placed):
    durations = [(e - s) * proto.FRAME for s, e in placed]
    crammed = sum(d < 0.06 for d in durations)
    held = runs = 0
    i = 0
    while i < len(words):
        j = i
        while (j < len(words) and words[j]["line"] == words[i]["line"]
               and INTERJECTION.match(re.sub(r"[^a-z]", "", words[j]["text"].lower()) or "-")):
            j += 1
        if j - i >= 3:
            runs += 1
            d = durations[i:j]
            rest = float(np.median(d[1:]))
            held += d[0] > 3 * rest and rest < 0.15 and d[0] > 0.6
        i = max(j, i + 1)
    return crammed, len(durations), held, runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="test.jsonl")
    parser.add_argument("--emissions", default="stem_emissions")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--word-ms", type=int, nargs="*", default=[], help="per-word floors to compare")
    parser.add_argument("--hybrid-ms", type=int, nargs="*", default=[],
                        help="per-word floors on top of 40 ms per letter")
    parser.add_argument("--adaptive", type=float, nargs="*", default=[], help="adaptive floor ratios")
    parser.add_argument("--onset", type=float, nargs="*", default=[], help="onset-paced floor ratios")
    parser.add_argument("--sung", type=float, nargs="*", default=[], help="sung-pace floor ratios")
    parser.add_argument("--sung2", type=float, nargs="*", default=[], help="sung-pace floors, letters paced too")
    parser.add_argument("--activity", type=float, nargs="*", default=[],
                        help="vocal-activity penalty strengths, on top of the paced floor x0.9")
    parser.add_argument("--stems", default="vocal_stems")
    parser.add_argument("--songs", type=int, default=80)
    parser.add_argument("--out", default="decoder_eval/duration.json")
    args = parser.parse_args()
    songs = [json.loads(line) for line in open(args.data)]
    songs = [s for s in songs if s.get("source_timing_granularity") in ("word", "syllable")]
    random.Random(7).shuffle(songs)
    songs = songs[:args.songs]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = {name: [] for name in ["torchaudio baseline", *[f"min {k * 20} ms/letter" for k in args.k],
                                  *[f"min {m} ms/word" for m in args.word_ms],
                                  *[f"40/letter + {m}/word" for m in args.hybrid_ms],
                                  *[f"adaptive x{r:g}" for r in args.adaptive],
                                  *[f"onset-paced x{r:g}" for r in args.onset],
                                  *[f"sung-pace x{r:g}" for r in args.sung],
                                  *[f"paced letters x{r:g}" for r in args.sung2],
                                  *[f"paced + activity {a:g}" for a in args.activity]]}
    extra = {name: [0, 0, 0, 0] for name in rows}
    started = time.time()
    for number, song in enumerate(songs, 1):
        words = song_words(song)
        emission = np.load(Path(args.emissions) / f"{song['id']}.npy").astype(np.float32)
        try:
            base = proto.baseline(emission, words, device)
        except Exception as error:
            print(f"skip {song['id']}: {error}")
            continue
        placed = {"torchaudio baseline": base}
        ids = [i for w in words for i in w["ids"]]
        for k in args.k:
            try:
                placed[f"min {k * 20} ms/letter"] = word_spans(words, min_duration_viterbi(emission, ids, k, device))
            except ValueError:
                placed[f"min {k * 20} ms/letter"] = base  # not enough frames for this floor
        for m in args.word_ms:
            try:
                placed[f"min {m} ms/word"] = word_spans(
                    words, min_duration_viterbi(emission, ids, word_floor(words, round(m / 20)), device))
            except ValueError:
                placed[f"min {m} ms/word"] = base
        for m in args.hybrid_ms:
            try:
                placed[f"40/letter + {m}/word"] = word_spans(
                    words, min_duration_viterbi(emission, ids, word_floor(words, round(m / 20), 2), device))
            except ValueError:
                placed[f"40/letter + {m}/word"] = base
        for r in args.adaptive:
            try:
                placed[f"adaptive x{r:g}"] = word_spans(
                    words, min_duration_viterbi(emission, ids, adaptive_floor(words, base, r), device))
            except ValueError:
                placed[f"adaptive x{r:g}"] = base
        for r in args.onset:
            try:
                placed[f"onset-paced x{r:g}"] = word_spans(
                    words, min_duration_viterbi(emission, ids, onset_floor(words, base, r), device))
            except ValueError:
                placed[f"onset-paced x{r:g}"] = base
        for r in args.sung:
            try:
                placed[f"sung-pace x{r:g}"] = word_spans(
                    words, min_duration_viterbi(emission, ids, sung_floor(words, base, r), device))
            except ValueError:
                placed[f"sung-pace x{r:g}"] = base
        for r in args.sung2:
            try:
                placed[f"paced letters x{r:g}"] = word_spans(
                    words, min_duration_viterbi(emission, ids, sung_floor2(words, base, r), device))
            except ValueError:
                placed[f"paced letters x{r:g}"] = base
        if args.activity:
            activity = vocal_activity(np.load(Path(args.stems) / f"{song['id']}.npy").astype(np.float32), emission.shape[0])
        for a in args.activity:
            adjusted = with_activity(emission, activity, a)
            try:
                plain = proto.baseline(adjusted, words, device)
                placed[f"paced + activity {a:g}"] = word_spans(
                    words, min_duration_viterbi(adjusted, ids, sung_floor2(words, plain, 0.9), device))
            except Exception:
                placed[f"paced + activity {a:g}"] = base
        for name, spans in placed.items():
            rows[name].append(proto.evaluate(words, spans))
            c, n, h, r = crammed_and_held(words, spans)
            extra[name] = [a + b for a, b in zip(extra[name], (c, n, h, r))]
        if number % 10 == 0:
            print(f"{number}/{len(songs)} songs, {time.time() - started:.0f}s", flush=True)
    report = {}
    for name, r in rows.items():
        summary = proto.summarize(r)
        c, n, h, runs = extra[name]
        summary.update(crammed_under_60ms=c, words=n, held_first_runs=h, interjection_runs=runs)
        report[name] = summary
        e, i = summary["errors"], summary["interjection_errors"]
        print(f"{name:22} start {e['median_ms']}/{e['mean_ms']} ms >500 {e['over_500ms']:5d} | interjections "
              f"{i['median_ms']}/{i['mean_ms']} ms | crammed <60 ms {c}/{n} ({c / max(1, n):.1%}) | "
              f"held-first runs {h}/{runs}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
