"""Align lead and backing words as two interleaved voices (shuffle product), no training.

After "Modeling Overlapped Speech with Shuffles" (Wiesner et al., arXiv 2603.17769): instead of one
CTC path through lead and backing words in lyric order, segments that contain backing words are
aligned through every interleaving of the two letter sequences. Each frame is blank, a lead letter
or a backing letter; each voice keeps its own letter order, with the usual CTC rules inside a voice
(a letter may span frames; a repeated letter needs a blank or the other voice in between). The
Viterbi path gives per-voice word times, so a backing word can sit inside the lead's words.
A cost per switch between voices keeps backing letters from scattering one frame at a time.

Uses a single-path model's cached frame log-probabilities on the vocal stems and the dataset's
backing marks, which the aligner knows from parentheses. Same scoring as checkpoint_eval.py.

usage: python -m echora_kara.evaluation.shuffle_align --emissions stem_emissions --costs 0,1,3,10
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from echora_kara.evaluation import interval_decoder as proto

LETTERS = {c: i for i, c in enumerate("|abcdefghijklmnopqrstuvwxyz'")}
PAD = round(0.3 / proto.FRAME)  # frames of slack around a segment, within its neighbours
NEG = -1e9


def shuffle_viterbi(emission, lead, backing, switch_cost):
    """Best interleaving of two letter sequences over emission frames.

    Returns, per sequence, the (first, last) frame of every letter. States, with i lead and j
    backing letters emitted so far: 0 blank after lead, 1 on lead letter i, 2 on backing letter j,
    3 blank after backing. Every change of voice (directly or across blanks) pays switch_cost.
    """
    T = emission.shape[0]
    M, N = len(lead), len(backing)
    blank = emission[:, proto.BLANK]
    a, b = np.array(lead, dtype=int), np.array(backing, dtype=int)
    # Direct letter-to-letter advance inside a voice needs a different letter (CTC repeat rule).
    a_direct = np.zeros(M + 1, dtype=bool)
    a_direct[2:] = a[1:] != a[:-1]
    b_direct = np.zeros(N + 1, dtype=bool)
    b_direct[2:] = b[1:] != b[:-1]
    score = np.full((4, M + 1, N + 1), NEG)
    score[0, 0, 0] = blank[0]
    if M:
        score[1, 1, 0] = emission[0, a[0]]
    if N:
        score[2, 0, 1] = emission[0, b[0]] - switch_cost
    # back[t, state, i, j] = (previous state, how many letters back in own voice: 0 or 1)
    back = np.zeros((T, 4, M + 1, N + 1), dtype=np.int8)
    for t in range(1, T):
        bl, la, ba, bb = score
        new = np.full_like(score, NEG)
        # blanks keep the last voice
        options = np.stack([bl, la])
        back[t, 0] = np.where(options.argmax(0) == 0, 0, 1)
        new[0] = options.max(0) + blank[t]
        options = np.stack([bb, ba])
        back[t, 3] = np.where(options.argmax(0) == 0, 3, 2)
        new[3] = options.max(0) + blank[t]
        # lead letter i (encoded: 1 stay, 10+s advance from state s at i-1)
        if M:
            options = np.stack(
                [
                    la[1:],
                    bl[:-1],
                    np.where(a_direct[1:, None], la[:-1], NEG),
                    ba[:-1] - switch_cost,
                    bb[:-1] - switch_cost,
                ]
            )
            codes = np.array([1, 10, 11, 12, 13], dtype=np.int8)
            back[t, 1, 1:] = codes[options.argmax(0)]
            new[1, 1:] = options.max(0) + emission[t, a][:, None]
        if N:
            options = np.stack(
                [
                    ba[:, 1:],
                    bb[:, :-1],
                    np.where(b_direct[None, 1:], ba[:, :-1], NEG),
                    la[:, :-1] - switch_cost,
                    bl[:, :-1] - switch_cost,
                ]
            )
            codes = np.array([2, 13, 12, 11, 10], dtype=np.int8)
            back[t, 2, :, 1:] = codes[options.argmax(0)]
            new[2, :, 1:] = options.max(0) + emission[t, b][None, :]
        score = new
    finals = score[:, M, N]
    state = int(np.argmax(finals))
    if finals[state] <= NEG / 2:
        raise ValueError("no path: too many letters for the frames")
    lead_frames = [[None, None] for _ in range(M)]
    back_frames = [[None, None] for _ in range(N)]
    i, j = M, N
    for t in range(T - 1, -1, -1):
        if state in (1, 2):
            span = lead_frames[i - 1] if state == 1 else back_frames[j - 1]
            span[0] = t
            if span[1] is None:
                span[1] = t
        if t == 0:
            break
        code = int(back[t, state, i, j])
        if code < 10:  # stay in a letter, or a blank continuing from (same i, j)
            state = code
        else:
            if state == 1:
                i -= 1
            else:
                j -= 1
            state = {10: 0, 11: 1, 12: 2, 13: 3}[code]
    return lead_frames, back_frames


def segments(words):
    """Groups of line numbers to align jointly: each line with backing words plus the next line."""
    lines = sorted({w["line"] for w in words})
    with_backing = {w["line"] for w in words if w["background"]}
    groups = []
    for index, line in enumerate(lines):
        if line not in with_backing:
            continue
        members = {line}
        if index + 1 < len(lines):
            members.add(lines[index + 1])
        if groups and groups[-1] & members:
            groups[-1] |= members
        else:
            groups.append(members)
    return groups


def align_song(emission, words, base, switch_cost):
    """Re-place the words of every backing segment with ``shuffle_viterbi`` inside the segment's single-path span
    (±PAD frames, clamped to its neighbours); returns (start_frame, end_frame) per word, keeping ``base`` for words
    outside segments and for segments with no path."""
    placed = list(base)
    for group in segments(words):
        inside = [k for k, w in enumerate(words) if w["line"] in group]
        lead = [k for k in inside if not words[k]["background"]]
        backing = [k for k in inside if words[k]["background"]]
        if not backing:
            continue
        outside_before = [base[k][1] for k, w in enumerate(words) if k < inside[0]]
        outside_after = [base[k][0] for k, w in enumerate(words) if k > inside[-1]]
        lo = min(base[k][0] for k in inside) - PAD
        hi = max(base[k][1] for k in inside) + PAD
        lo = max(lo, max(outside_before, default=0), 0)
        hi = min(hi, min(outside_after, default=emission.shape[0]), emission.shape[0])
        letters_lead = [i for k in lead for i in words[k]["ids"]]
        letters_back = [i for k in backing for i in words[k]["ids"]]
        try:
            lead_frames, back_frames = shuffle_viterbi(emission[lo:hi], letters_lead, letters_back, switch_cost)
        except ValueError:
            continue
        for keys, frames in ((lead, lead_frames), (backing, back_frames)):
            cursor = 0
            for k in keys:
                own = frames[cursor : cursor + len(words[k]["ids"])]
                cursor += len(words[k]["ids"])
                placed[k] = (lo + own[0][0], lo + own[-1][1] + 1)
    return placed


def song_words(song):
    """Timed words, including the backing words the dataset blanked where they overlap the lead
    (normalization_error overlapping_background_vocal): their letters are rebuilt from the source
    text with the model's vocabulary. Words with letters outside it (non-Latin scripts) stay out."""
    words = []
    for t in song["alignment_tokens"]:
        ids = [int(i) for i in t.get("ids") or []]
        restored = False
        if not ids and t.get("normalization_error") == "overlapping_background_vocal":
            text = "".join(c for c in (t.get("source_text") or "").lower() if c.isalpha() or c == "'")
            if text and all(c in LETTERS for c in text):
                ids, restored = [LETTERS[c] for c in text], True
        if not ids or (t.get("ignore") and not restored):
            continue
        words.append(
            {
                "start": float(t["start"]),
                "end": float(t["end"]),
                "ids": ids,
                "text": t.get("text") or t.get("source_text", ""),
                "line": t["line"],
                "background": t.get("vocal_role") == "background",
                "restored": restored,
            }
        )
    return words


def backing_metrics(words, placed):
    """Lead/backing overlaps only: reference pairs recovered, backing-word start errors."""
    pred = [(proto.frame_time(s), proto.frame_time(e)) for s, e in placed]
    pairs = [p for p in proto.reference_pairs(words) if words[p[0]]["background"] != words[p[1]]["background"]]
    recovered = sum(min(pred[a][1], pred[b][1]) - max(pred[a][0], pred[b][0]) >= proto.RECOVERED for a, b in pairs)
    errors = [abs(pred[k][0] - w["start"]) for k, w in enumerate(words) if w["background"]]
    lead = [abs(pred[k][0] - w["start"]) for k, w in enumerate(words) if not w["background"]]
    return {"pairs": len(pairs), "recovered": recovered, "backing_errors": errors, "lead_errors": lead}


def main():
    """Compare the single path with shuffle alignment at each switch cost on cached test-song emissions; prints
    all-word and lead/backing summaries and writes --out/report.json."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="test.jsonl")
    parser.add_argument("--songs", type=int, default=80)
    parser.add_argument("--emissions", default="stem_emissions")
    parser.add_argument("--costs", default="0,1,3,10")
    parser.add_argument("--out", default="shuffle_align")
    args = parser.parse_args()
    songs = [json.loads(line) for line in open(args.data)]
    songs = [s for s in songs if s.get("source_timing_granularity") in ("word", "syllable")]
    random.Random(7).shuffle(songs)
    songs = songs[: args.songs]
    costs = [float(c) for c in args.costs.split(",")]
    rows = {"v6 single path": [], **{f"shuffle, switch cost {c:g}": [] for c in costs}}
    backing_rows = {name: [] for name in rows}  # the songs that have backing words
    started = time.time()
    for number, song in enumerate(songs, 1):
        words = song_words(song)
        emission = np.load(Path(args.emissions) / f"{song['id']}.npy").astype(np.float32)
        try:
            base = proto.baseline(emission, words, "cpu")
        except Exception as error:
            print(f"skip {song['id']}: {error}")
            continue
        has_backing = any(w["background"] for w in words)
        result = proto.evaluate(words, base)
        rows["v6 single path"].append(result)
        if has_backing:
            backing_rows["v6 single path"].append(backing_metrics(words, base))
        for c in costs:
            placed = align_song(emission, words, base, c) if has_backing else base
            rows[f"shuffle, switch cost {c:g}"].append(proto.evaluate(words, placed))
            if has_backing:
                backing_rows[f"shuffle, switch cost {c:g}"].append(backing_metrics(words, placed))
        if number % 10 == 0:
            print(f"{number}/{len(songs)} songs, {time.time() - started:.0f}s", flush=True)

    def backing_summary(rs):
        back = np.array([e for r in rs for e in r["backing_errors"]]) * 1000
        lead = np.array([e for r in rs for e in r["lead_errors"]]) * 1000
        pairs = sum(r["pairs"] for r in rs)
        return {
            "songs": len(rs),
            "lead_backing_pairs": pairs,
            "recovered": round(sum(r["recovered"] for r in rs) / max(1, pairs), 3),
            "backing_words": int(back.size),
            "backing_median_ms": round(float(np.median(back)), 1),
            "backing_mean_ms": round(float(back.mean()), 1),
            "backing_over_500": int((back > 500).sum()),
            "lead_median_ms": round(float(np.median(lead)), 1),
            "lead_mean_ms": round(float(lead.mean()), 1),
            "lead_over_500": int((lead > 500).sum()),
        }

    report = {
        "all songs": {n: proto.summarize(r) for n, r in rows.items()},
        "songs with backing": {n: backing_summary(r) for n, r in backing_rows.items()},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print("== all songs (all words; overlaps here are lead lines overlapping each other)")
    for name, value in report["all songs"].items():
        e = value["errors"]
        print(
            f"{name:26} start {e['median_ms']}/{e['mean_ms']} ms >500ms {e['over_500ms']:5d} | "
            f"invented overlaps {value['invented_overlap_rate']:.1%}"
        )
    print("== songs with backing words: lead/backing overlaps")
    for name, v in report["songs with backing"].items():
        print(
            f"{name:26} recovered {v['recovered']:.0%} of {v['lead_backing_pairs']} | backing start "
            f"{v['backing_median_ms']}/{v['backing_mean_ms']} ms >500 {v['backing_over_500']}/{v['backing_words']} | "
            f"lead {v['lead_median_ms']}/{v['lead_mean_ms']} ms >500 {v['lead_over_500']}"
        )


if __name__ == "__main__":
    main()
