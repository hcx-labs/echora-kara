"""Judge an overlap-head checkpoint on what matters: word timing and simultaneous words, on 80 test songs.

The same songs as interval_decoder.py (the baseline model's frame outputs are cached there). In the
report, ``v6`` labels the baseline model (interval_decoder.REPO) and ``v7`` the checkpoint. Three
decodings are compared:

* v6 single path: one CTC path through the baseline's softmax outputs (single-path production);
* v7 single path: the same decoding on the checkpoint's CTC head;
* v7 overlap decoder: every word gets its own interval scored on the overlap head (log-odds of
  its letters, filled in order without blanks, as in training), and the interval dynamic program
  lets a word overlap the previous one at a per-frame cost.

usage: python -m echora_kara.evaluation.checkpoint_eval <run>/checkpoint-1000
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from echora_kara.evaluation import interval_decoder as proto
from echora_kara.train.train import ForcedAligner


def outputs(model, audio, device):
    """CTC log-probabilities and overlap-head log-odds per frame, chunked as inference does."""
    chunk, overlap = 45 * proto.SAMPLE_RATE, 2 * proto.SAMPLE_RATE
    starts = [s for s in range(0, len(audio), chunk - overlap) if s == 0 or len(audio) - s > overlap]
    ctc_parts, odds_parts = [], []
    for index, start in enumerate(starts):
        piece = audio[start:start + chunk]
        piece = (piece - piece.mean()) / (piece.std() + 1e-7)
        with torch.inference_mode():
            result = model(torch.from_numpy(piece)[None].to(device))
            ctc = torch.log_softmax(result.logits[0].float(), dim=-1).cpu().numpy()
            odds = (result.overlap_logits[0].float().cpu().numpy() if getattr(result, "overlap_logits", None) is not None
                    else None)  # logit = log-odds
        frames = round(ctc.shape[0] * overlap / max(1, len(piece)))
        left = frames // 2 if index else 0
        right = frames - frames // 2 if index + 1 < len(starts) else 0
        stop = ctc.shape[0] - right if right else None
        ctc_parts.append(ctc[left:stop])
        odds_parts.append(odds[left:stop] if odds is not None else None)
    return np.concatenate(ctc_parts), (np.concatenate(odds_parts) if odds_parts[0] is not None else None)


def interval_scores(odds, ids, starts):
    """Best in-order, blank-free filling of [start, start+length) by the word's letters, scored as
    the summed log-odds of the letter assigned to each frame. Array (len(starts), MAX_LEN)."""
    total, n = odds.shape[0], len(ids)
    labels = np.array(ids)
    scores = np.full((len(starts), proto.MAX_LEN), -np.inf, dtype=np.float32)
    v = np.full((len(starts), n), -np.inf, dtype=np.float32)
    v[:, 0] = odds[np.minimum(starts, total - 1), labels[0]]
    for offset in range(proto.MAX_LEN):
        frames = starts + offset
        valid = frames < total
        if offset:
            emit = odds[np.minimum(frames, total - 1)][:, labels]
            advance = np.full_like(v, -np.inf)
            advance[:, 1:] = v[:, :-1]
            v = np.maximum(v, advance) + emit
            v[~valid] = -np.inf
        scores[:, offset] = np.where(valid & (offset + 1 >= n), v[:, -1], -np.inf)
    return scores


def candidates(odds, word, base):
    total = odds.shape[0]
    starts = np.arange(max(0, base[0] - proto.SEARCH), min(total - 1, base[0] + proto.SEARCH) + 1)
    scores = interval_scores(odds, word["ids"], starts)
    flat = np.argsort(scores, axis=None)[::-1][:proto.CANDIDATES]
    found = {(int(starts[i // proto.MAX_LEN]), int(starts[i // proto.MAX_LEN] + i % proto.MAX_LEN + 1)): float(scores.flat[i])
             for i in flat if np.isfinite(scores.flat[i])}
    b_start, b_end = base
    if starts[0] <= b_start <= starts[-1] and 0 < b_end - b_start <= proto.MAX_LEN:
        value = scores[b_start - starts[0], b_end - b_start - 1]
        found.setdefault((b_start, b_end), float(value) if np.isfinite(value) else -1e3)
    return [(s, e, sc) for (s, e), sc in found.items()] or [(b_start, b_end, -1e3)]


def presence(odds, words):
    """Per word and frame: how confidently the overlap head hears any of the word's letters
    (max log-odds over its letters), as prefix sums for O(1) interval means."""
    sums = []
    for w in words:
        values = odds[:, sorted(set(w["ids"]))].max(axis=1)
        sums.append(np.concatenate([[0.0], np.cumsum(values)]))
    return sums


def gated_decode(words, cands, prefix, cost, threshold):
    """Like proto.decode, but two words may overlap only where the overlap head hears both of them
    (mean presence over the overlapped frames above ``threshold`` for each); otherwise forbidden."""
    best = [np.array([c[2] for c in cands[0]])]
    back = [None]
    for w in range(1, len(words)):
        prev, cur = cands[w - 1], cands[w]
        p_start = np.array([c[0] for c in prev])[:, None]
        p_end = np.array([c[1] for c in prev])[:, None]
        c_start = np.array([c[0] for c in cur])[None, :]
        overlap = np.maximum(0, p_end - c_start)
        lo = np.broadcast_to(c_start, overlap.shape)
        hi = lo + overlap
        mean_prev = np.where(overlap > 0, (prefix[w - 1][hi] - prefix[w - 1][lo]) / np.maximum(overlap, 1), 0)
        mean_cur = np.where(overlap > 0, (prefix[w][hi] - prefix[w][lo]) / np.maximum(overlap, 1), 0)
        heard = (overlap == 0) | ((mean_prev > threshold) & (mean_cur > threshold))
        trans = best[-1][:, None] - cost * overlap
        trans = np.where((p_start <= c_start) & heard, trans, -np.inf)
        choice = trans.argmax(axis=0)
        value = trans[choice, np.arange(len(cur))]
        if not np.isfinite(value).any():
            value = best[-1].max() + np.zeros(len(cur))
        best.append(value + np.array([c[2] for c in cur]))
        back.append(choice)
    pick = int(best[-1].argmax())
    result = [None] * len(words)
    for w in range(len(words) - 1, -1, -1):
        result[w] = cands[w][pick][:2]
        if back[w] is not None:
            pick = int(back[w][pick])
    return result


GATES = ((0.5, 0.0), (0.5, 1.0), (0.5, 2.0), (0.5, 3.0), (1.0, 2.0))
DISTINCT_GATES = ((0.5, -1.0), (0.5, 0.0), (0.5, 1.0), (0.5, 2.0))


def distinct_presence(odds, words):
    """Per adjacent pair (w-1, w): prefix sums of each word's presence over only the letters the
    other word lacks. Shared letters (often vowels) light up for either word, so they cannot show
    that both are sung. A word with no distinctive letter cannot be shown to overlap."""
    pairs = [None]
    for w in range(1, len(words)):
        a, b = set(words[w - 1]["ids"]), set(words[w]["ids"])
        sums = []
        for own, other in ((a, b), (b, a)):
            letters = sorted(own - other)
            values = odds[:, letters].max(axis=1) if letters else np.full(len(odds), -1e3)
            sums.append(np.concatenate([[0.0], np.cumsum(values)]))
        pairs.append(sums)
    return pairs


def distinct_decode(words, cands, pairs, cost, threshold):
    """gated_decode with presence measured on each word's distinctive letters."""
    shifted = [None] * len(words)
    for w in range(1, len(words)):
        shifted[w] = pairs[w]
    best = [np.array([c[2] for c in cands[0]])]
    back = [None]
    for w in range(1, len(words)):
        prev, cur = cands[w - 1], cands[w]
        p_start = np.array([c[0] for c in prev])[:, None]
        p_end = np.array([c[1] for c in prev])[:, None]
        c_start = np.array([c[0] for c in cur])[None, :]
        overlap = np.maximum(0, p_end - c_start)
        lo = np.broadcast_to(c_start, overlap.shape)
        hi = lo + overlap
        prev_sum, cur_sum = shifted[w]
        mean_prev = np.where(overlap > 0, (prev_sum[hi] - prev_sum[lo]) / np.maximum(overlap, 1), 0)
        mean_cur = np.where(overlap > 0, (cur_sum[hi] - cur_sum[lo]) / np.maximum(overlap, 1), 0)
        heard = (overlap == 0) | ((mean_prev > threshold) & (mean_cur > threshold))
        trans = np.where((p_start <= c_start) & heard, best[-1][:, None] - cost * overlap, -np.inf)
        choice = trans.argmax(axis=0)
        value = trans[choice, np.arange(len(cur))]
        if not np.isfinite(value).any():
            value = best[-1].max() + np.zeros(len(cur))
        best.append(value + np.array([c[2] for c in cur]))
        back.append(choice)
    pick = int(best[-1].argmax())
    result = [None] * len(words)
    for w in range(len(words) - 1, -1, -1):
        result[w] = cands[w][pick][:2]
        if back[w] is not None:
            pick = int(back[w][pick])
    return result


def paired(v6_rows, v7_rows):
    """Per-song comparison of the two single paths, robust to a few songs with huge errors."""
    if len(v6_rows) != len(v7_rows) or not v6_rows:
        return {}
    v6_mean = np.array([np.mean(r["errors"]) for r in v6_rows]) * 1000
    v7_mean = np.array([np.mean(r["errors"]) for r in v7_rows]) * 1000
    v6_median = np.array([np.median(r["errors"]) for r in v6_rows]) * 1000
    v7_median = np.array([np.median(r["errors"]) for r in v7_rows]) * 1000
    v6_long = np.array([np.mean(np.array(r["errors"]) > 0.5) for r in v6_rows])
    v7_long = np.array([np.mean(np.array(r["errors"]) > 0.5) for r in v7_rows])
    v6_int = [np.mean(r["interjection_errors"]) for r in v6_rows if r["interjection_errors"]]
    v7_int = [np.mean(r["interjection_errors"]) for r in v7_rows if r["interjection_errors"]]
    difference = v7_mean - v6_mean
    return {
        "songs": len(v6_rows),
        "v7_better_mean": int((difference < 0).sum()), "v7_worse_mean": int((difference > 0).sum()),
        "median_song_mean_change_ms": round(float(np.median(difference)), 1),
        "median_song_median_change_ms": round(float(np.median(v7_median - v6_median)), 1),
        "v7_better_median": int((v7_median < v6_median).sum()),
        "long_miss_share_v6": round(float(v6_long.mean()), 4), "long_miss_share_v7": round(float(v7_long.mean()), 4),
        "v7_fewer_long_misses": int((v7_long < v6_long).sum()), "v7_more_long_misses": int((v7_long > v6_long).sum()),
        "interjection_songs": len(v6_int),
        "v7_better_interjections": int(sum(b < a for a, b in zip(v6_int, v7_int))),
        "median_song_interjection_change_ms": round(float(np.median(np.array(v7_int) - np.array(v6_int))) * 1000, 1)
        if v6_int else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--data", default="test.jsonl")
    parser.add_argument("--songs", type=int, default=80, help="0: every test song with word timing")
    parser.add_argument("--v6-cache", default="overlap_proto/emissions")
    parser.add_argument("--costs", default="0.5,1,2,4,8")
    parser.add_argument("--out", default="checkpoint_eval")
    parser.add_argument("--stems", default=None,
                        help="Directory of separated vocals (<id>.npy): evaluate on what production aligns")
    args = parser.parse_args()
    songs = [json.loads(line) for line in open(args.data)]
    songs = [s for s in songs if s.get("source_timing_granularity") in ("word", "syllable")]
    random.Random(7).shuffle(songs)
    songs = songs[:args.songs] if args.songs else songs
    device = "cuda"
    model = ForcedAligner.from_pretrained(args.checkpoint).to(device).eval()
    has_overlap = model.overlap_head is not None  # control runs have none: single path only
    costs = [float(c) for c in args.costs.split(",")]
    rows = {"v6 single path": [], "v7 single path": [], **{f"v7 overlap decoder, cost {c:g}": [] for c in costs},
            **{f"v7 gated decoder, cost {c:g}, heard > {t:g}": [] for c, t in GATES},
            **{f"v7 distinct-letter gate, cost {c:g}, heard > {t:g}": [] for c, t in DISTINCT_GATES}}
    started = time.time()
    v6_model = None
    for number, song in enumerate(songs, 1):
        words = [
            {"start": float(t["start"]), "end": float(t["end"]), "ids": [int(i) for i in t["ids"]],
             "text": t.get("text", ""), "line": t["line"], "background": t.get("vocal_role") == "background"}
            for t in song["alignment_tokens"] if t.get("ids") and not t.get("ignore")
        ]
        if args.stems:
            stem = Path(args.stems) / f"{song['id']}.npy"
            if not stem.exists():
                print(f"skip {song['id']}: no vocal stem")
                continue
            audio = np.load(stem).astype(np.float32)
        else:
            audio = proto.decode_audio(Path(song["audio"]))
        ctc, odds = outputs(model, audio, device)
        cached = Path(args.v6_cache) / f"{song['id']}.npy"
        if not cached.exists():
            if v6_model is None:
                from transformers import Wav2Vec2ForCTC
                v6_model = Wav2Vec2ForCTC.from_pretrained(proto.REPO, revision=proto.REVISION).to(device).eval()
            cached.parent.mkdir(parents=True, exist_ok=True)
            np.save(cached, proto.emissions(v6_model, audio, device).astype(np.float16))
        try:
            v7_base = proto.baseline(ctc, words, device)
            if cached.exists():
                rows["v6 single path"].append(proto.evaluate(words, proto.baseline(np.load(cached).astype(np.float32), words, device)))
        except Exception as error:
            print(f"skip {song['id']}: {error}")
            continue
        rows["v7 single path"].append(proto.evaluate(words, v7_base))
        if not has_overlap:
            continue
        cands = [candidates(odds, w, b) for w, b in zip(words, v7_base)]
        for c in costs:
            rows[f"v7 overlap decoder, cost {c:g}"].append(proto.evaluate(words, proto.decode(words, cands, c)))
        prefix = presence(odds, words)
        for c, t in GATES:
            rows[f"v7 gated decoder, cost {c:g}, heard > {t:g}"].append(
                proto.evaluate(words, gated_decode(words, cands, prefix, c, t)))
        pairs = distinct_presence(odds, words)
        for c, t in DISTINCT_GATES:
            rows[f"v7 distinct-letter gate, cost {c:g}, heard > {t:g}"].append(
                proto.evaluate(words, distinct_decode(words, cands, pairs, c, t)))
        if number % 10 == 0:
            print(f"{number}/{len(songs)} songs, {time.time() - started:.0f}s", flush=True)
    report = {name: proto.summarize(r) for name, r in rows.items() if r}
    report["paired"] = paired(rows["v6 single path"], rows["v7 single path"])
    report["checkpoint"] = args.checkpoint
    report["songs"] = len(rows["v7 single path"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{Path(args.checkpoint).name}.json").write_text(json.dumps(report, indent=2))
    print("paired baseline vs checkpoint single path:", json.dumps(report["paired"]))
    for name, value in report.items():
        if isinstance(value, dict) and "errors" in value:
            e, o, i = value["errors"], value["overlap_errors"], value["interjection_errors"]
            print(f"{name:40} start {e['median_ms']}/{e['mean_ms']} ms >500ms {e['over_500ms']:5d} | overlap words "
                  f"{o['median_ms']} ms | interjections {i['median_ms']}/{i['mean_ms']} ms | recovered "
                  f"{value['overlap_recall']:.0%} invented {value['invented_overlap_rate']:.1%}")


if __name__ == "__main__":
    main()
