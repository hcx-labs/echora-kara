"""Can an overlap-allowing interval decoder recover simultaneous words from a single-path model's outputs?

No training. For test-split songs with measured word timing, run the baseline model (REPO) once,
then place every word two ways on the same frame log-probabilities:

* baseline: one CTC path through the whole song (torchaudio forced_align), the single-path decoding;
* interval: each word gets its own interval [start, end), scored as the log-likelihood ratio of
  its letters (a local CTC path inside the interval) against blank over the same frames. A dynamic
  program over words keeps starts in lyric order and lets a word overlap the previous one at a
  cost per overlapped frame: high inside a line, lower across lines, lowest for background words
  (parenthesized in the lyrics, so known at inference).

Reports word-start error (all words, words that overlap in the reference, interjections), how many
reference overlaps are recovered and how many overlaps are invented, for each overlap cost.

usage: python -m echora_kara.evaluation.interval_decoder --songs 80 --out overlap_proto
"""

import argparse
import json
import random
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC

REPO = "hcX02/echora-mms-300m-multilingual-lyrics-forced-aligner"
REVISION = "1f241323a4f5d961160ae7c7c88e5c7d27f32f59"
BLANK = 29
SAMPLE_RATE = 16000
FRAME = 320 / SAMPLE_RATE
OFFSET = 199.5 / SAMPLE_RATE  # receptive-field centre of frame 0, as in training
INTERJECTION = re.compile(r"^(o+h+|u+h+|a+h+|h?u+h|wh?oa+h?|woo+h?|o+|a+)$")
SEARCH = round(1.5 / FRAME)  # candidate starts within ±1.5 s of the baseline start
MAX_LEN = round(4.0 / FRAME)
CANDIDATES = 40
OVERLAP_MIN = 0.2  # a reference pair counts as simultaneous from 200 ms
RECOVERED = 0.08  # a predicted pair counts as overlapping from 80 ms (4 frames)


def frame_time(frame):
    """Time in seconds of a stitched frame index (receptive-field centre, 20 ms grid)."""
    return OFFSET + frame * FRAME


def decode_audio(path):
    """Decode any audio file with ffmpeg to mono float32 samples at 16 kHz, shape (samples,)."""
    raw = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "f32le",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def emissions(model, audio, device):
    """Log-probabilities per frame, in 45 s chunks with 2 s shared context (as inference does)."""
    chunk, overlap = 45 * SAMPLE_RATE, 2 * SAMPLE_RATE
    step = chunk - overlap
    starts = [s for s in range(0, len(audio), step) if s == 0 or len(audio) - s > overlap]
    parts = []
    for index, start in enumerate(starts):
        piece = audio[start : start + chunk]
        piece = (piece - piece.mean()) / (piece.std() + 1e-7)
        with torch.inference_mode():
            logits = model(torch.from_numpy(piece)[None].to(device)).logits[0]
            local = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        frames = round(local.shape[0] * overlap / max(1, len(piece)))
        left = frames // 2 if index else 0
        right = frames - frames // 2 if index + 1 < len(starts) else 0
        parts.append(local[left : local.shape[0] - right if right else None])
    return np.concatenate(parts)


def baseline(emission, words, device):
    """One CTC path through all words; returns (start_frame, end_frame) per word."""
    targets = [i for w in words for i in w["ids"]]
    path, scores = torchaudio.functional.forced_align(
        torch.from_numpy(emission)[None].to(device), torch.tensor([targets], device=device), blank=BLANK
    )
    spans = torchaudio.functional.merge_tokens(path[0].cpu(), scores[0].exp().cpu(), blank=BLANK)
    assert len(spans) == len(targets)
    result, cursor = [], 0
    for w in words:
        own = spans[cursor : cursor + len(w["ids"])]
        cursor += len(w["ids"])
        result.append((own[0].start, own[-1].end))
    return result


def interval_scores(emission, ids, starts):
    """LLR of the word's letters vs blank for every (start, length): array (len(starts), MAX_LEN)."""
    labels = []
    for k, char in enumerate(ids):
        labels.append(char)
        if k + 1 < len(ids):
            labels.append(BLANK)
    labels = np.array(labels)
    n = len(labels)
    is_char = labels != BLANK
    skip = np.zeros(n, dtype=bool)  # char state i may be entered from char state i-2 (skipping a blank)
    for i in range(2, n, 2):
        skip[i] = labels[i] != labels[i - 2]
    total = emission.shape[0]
    blank_prefix = np.concatenate([[0.0], np.cumsum(emission[:, BLANK])])
    scores = np.full((len(starts), MAX_LEN), -np.inf, dtype=np.float32)
    v = np.full((len(starts), n), -np.inf, dtype=np.float32)
    first_rows = emission[np.minimum(starts, total - 1)]
    v[:, 0] = first_rows[:, labels[0]]
    for offset in range(MAX_LEN):
        frames = starts + offset
        valid = frames < total
        if offset:
            rows = emission[np.minimum(frames, total - 1)]
            emit = rows[:, labels]
            stay = v
            step = np.full_like(v, -np.inf)
            step[:, 1:] = v[:, :-1]
            jump = np.full_like(v, -np.inf)
            jump[:, 2:] = v[:, :-2]
            jump[:, ~skip] = -np.inf
            v = np.maximum(np.maximum(stay, step), jump) + emit
            v[~valid] = -np.inf
        end = np.minimum(frames + 1, total)
        llr = v[:, -1] - (blank_prefix[end] - blank_prefix[np.minimum(starts, total)])
        scores[:, offset] = np.where(valid & (offset + 1 >= int(is_char.sum())), llr, -np.inf)
    return scores


def candidates(emission, word, base):
    """Up to CANDIDATES best (start_frame, end_frame, score) intervals for ``word`` with starts within ±1.5 s of
    its baseline start; the baseline interval itself is always included when it fits, and -inf scores become -1000."""
    total = emission.shape[0]
    starts = np.arange(max(0, base[0] - SEARCH), min(total - 1, base[0] + SEARCH) + 1)
    scores = interval_scores(emission, word["ids"], starts)
    flat = np.argsort(scores, axis=None)[::-1][:CANDIDATES]
    found = {
        (int(starts[i // MAX_LEN]), int(starts[i // MAX_LEN] + i % MAX_LEN + 1)): float(scores.flat[i])
        for i in flat
        if np.isfinite(scores.flat[i])
    }
    b_start, b_end = base
    if b_start in set(starts.tolist()) and 0 < b_end - b_start <= MAX_LEN:
        found.setdefault((b_start, b_end), float(scores[b_start - starts[0], b_end - b_start - 1]))
    return [(s, e, sc if np.isfinite(sc) else -1e3) for (s, e), sc in found.items()]


def decode(words, cands, cost):
    """Best interval per word: starts in lyric order, overlap with the previous word costs per frame."""
    best = [np.array([c[2] for c in cands[0]])]
    back = [None]
    for w in range(1, len(words)):
        prev, cur = cands[w - 1], cands[w]
        rate = cost * (
            0.3
            if words[w]["background"] or words[w - 1]["background"]
            else 3.0
            if words[w]["line"] == words[w - 1]["line"]
            else 1.0
        )
        p_start = np.array([c[0] for c in prev])[:, None]
        p_end = np.array([c[1] for c in prev])[:, None]
        c_start = np.array([c[0] for c in cur])[None, :]
        trans = best[-1][:, None] - rate * np.maximum(0, p_end - c_start)
        trans = np.where(p_start <= c_start, trans, -np.inf)
        choice = trans.argmax(axis=0)
        value = trans[choice, np.arange(len(cur))]
        if not np.isfinite(value).any():  # no ordered option: fall back to the best unconstrained one
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


def reference_pairs(words):
    """Index pairs (a < b) of reference words that are sung simultaneously for at least OVERLAP_MIN seconds."""
    order = sorted(range(len(words)), key=lambda i: words[i]["start"])
    pairs = set()
    for a_index, a in enumerate(order):
        for b in order[a_index + 1 :]:
            if words[b]["start"] >= words[a]["end"]:
                break
            if min(words[a]["end"], words[b]["end"]) - words[b]["start"] >= OVERLAP_MIN:
                pairs.add((min(a, b), max(a, b)))
    return pairs


def evaluate(words, placed):
    """Score one song's placed (start_frame, end_frame) intervals against the reference word times: start errors in
    seconds (all, overlapping, interjections) and counts of reference overlaps recovered and overlaps invented between
    words that are sequential in the reference."""
    pred = [(frame_time(s), frame_time(e)) for s, e in placed]
    errors = [abs(p[0] - w["start"]) for p, w in zip(pred, words)]
    simultaneous = reference_pairs(words)
    in_pair = {i for pair in simultaneous for i in pair}
    recovered = sum(min(pred[a][1], pred[b][1]) - max(pred[a][0], pred[b][0]) >= RECOVERED for a, b in simultaneous)
    invented = checked = 0
    for a in range(len(words) - 1):
        if words[a + 1]["start"] >= words[a]["end"]:  # sequential in the reference
            checked += 1
            invented += min(pred[a][1], pred[a + 1][1]) - max(pred[a][0], pred[a + 1][0]) >= RECOVERED
    return {
        "errors": errors,
        "overlap_errors": [errors[i] for i in in_pair],
        "interjection_errors": [e for e, w in zip(errors, words) if INTERJECTION.match(w["text"])],
        "pairs": len(simultaneous),
        "recovered": recovered,
        "sequential": checked,
        "invented": invented,
    }


def summarize(rows):
    """Aggregate per-song ``evaluate`` results: start-error stats in ms per error kind, overlap recall and the rate of
    invented overlaps."""
    out = {}
    for key in ("errors", "overlap_errors", "interjection_errors"):
        values = np.array([v for r in rows for v in r[key]]) * 1000
        out[key] = {
            "n": int(values.size),
            "median_ms": round(float(np.median(values)), 1) if values.size else None,
            "mean_ms": round(float(values.mean()), 1) if values.size else None,
            "over_500ms": int((values > 500).sum()),
        }
    pairs = sum(r["pairs"] for r in rows)
    sequential = sum(r["sequential"] for r in rows)
    out["overlap_recall"] = round(sum(r["recovered"] for r in rows) / max(1, pairs), 3)
    out["invented_overlap_rate"] = round(sum(r["invented"] for r in rows) / max(1, sequential), 4)
    out["reference_pairs"] = pairs
    return out


def main():
    """Run the baseline model on the test songs (caching emissions under --out/emissions), compare the single-path
    baseline with the interval decoder at each overlap cost, and write --out/report.json."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="test.jsonl")
    parser.add_argument("--songs", type=int, default=80)
    parser.add_argument("--out", default="overlap_proto")
    parser.add_argument("--costs", default="0.25,0.5,1,2,4,8")
    args = parser.parse_args()
    out = Path(args.out)
    (out / "emissions").mkdir(parents=True, exist_ok=True)
    songs = [json.loads(line) for line in open(args.data)]
    songs = [s for s in songs if s.get("source_timing_granularity") in ("word", "syllable")]
    random.Random(7).shuffle(songs)
    songs = songs[: args.songs]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Wav2Vec2ForCTC.from_pretrained(REPO, revision=REVISION).to(device).eval()
    costs = [float(c) for c in args.costs.split(",")]
    rows = {"baseline": [], **{f"interval cost {c:g}": [] for c in costs}}
    started = time.time()
    for number, song in enumerate(songs, 1):
        words = [
            {
                "start": float(t["start"]),
                "end": float(t["end"]),
                "ids": [int(i) for i in t["ids"]],
                "text": t.get("text", ""),
                "line": t["line"],
                "background": t.get("vocal_role") == "background",
            }
            for t in song["alignment_tokens"]
            if t.get("ids") and not t.get("ignore")
        ]
        cache = out / "emissions" / f"{song['id']}.npy"
        if cache.exists():
            emission = np.load(cache).astype(np.float32)
        else:
            emission = emissions(model, decode_audio(Path(song["audio"])), device)
            np.save(cache, emission.astype(np.float16))
        try:
            base = baseline(emission, words, device)
        except Exception as error:  # path impossible (too many letters for the frames)
            print(f"skip {song['id']}: {error}")
            continue
        rows["baseline"].append(evaluate(words, base))
        cands = [candidates(emission, w, b) for w, b in zip(words, base)]
        for c in costs:
            rows[f"interval cost {c:g}"].append(evaluate(words, decode(words, cands, c)))
        print(
            f"{number}/{len(songs)} {song['artist']} – {song['title']} ({len(words)} words, {time.time() - started:.0f}s)",
            flush=True,
        )
    report = {name: summarize(r) for name, r in rows.items()}
    report["songs"] = len(rows["baseline"])
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
