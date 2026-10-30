"""The 80-song word-timing harness on the VALIDATION split (checkpoint selection and timing
calibration happen here; the test split is reported once per released model, never used to pick).

Same songs policy as checkpoint_eval.py / duration_viterbi.py (word- or syllable-timed songs, shuffled
with seed 7, first 80) but drawn from validation.jsonl of a stem dataset, whose audio is already the
Roformer vocal stem production hears. Same metrics: word-start error (median / mean / > 500 ms),
interjections, lead/backing overlaps (backing_metrics), crammed and held-first runs, plus per-language
medians and the per-song paired comparison against a baseline model (--baseline).

Only stream 1 (the CTC head, ``.logits``) is scored: it is what production decodes.
Frame log-probabilities are cached per model as float16 under --cache/<name>/<id>.npy (the raw material
for backing post-processing and decoder tuning), so re-scoring costs no GPU.

usage:
  python -m echora_kara.evaluation.validation <model dir> --name base --production
  python -m echora_kara.evaluation.validation <run>/checkpoint-1000 --baseline base --bf16
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from echora_kara.evaluation import checkpoint_eval as ev
from echora_kara.evaluation import interval_decoder as proto
from echora_kara.evaluation.duration_viterbi import (
    crammed_and_held,
    min_duration_viterbi,
    sung_floor2,
    vocal_activity,
    with_activity,
    word_spans,
)
from echora_kara.evaluation.shuffle_align import backing_metrics

PACE, ACTIVITY = 0.9, 2.0  # production decoder constants


def song_words(song):
    """Timed lead and backing words of a manifest song (tokens with ids and not ignored): start / end seconds, token
    ids, text, line number and background flag."""
    tokens = [t for t in song["alignment_tokens"] if t.get("ids") and not t.get("ignore")]
    return [
        {
            "start": float(t["start"]),
            "end": float(t["end"]),
            "ids": [int(i) for i in t["ids"]],
            "text": t.get("text", ""),
            "line": t["line"],
            "background": t.get("vocal_role") == "background",
        }
        for t in tokens
    ]


def load_model(path, device, bf16):
    """Load a ``train.ForcedAligner`` checkpoint on ``device`` in eval mode, in bfloat16 when ``bf16``."""
    from echora_kara.train.train import ForcedAligner  # handles checkpoints with extra heads

    model = ForcedAligner.from_pretrained(path, dtype=torch.bfloat16 if bf16 else torch.float32)
    return model.to(device).eval()


def emissions(model, audio, device):
    """proto.emissions with the input cast to the model's dtype (bf16 inference for the 1B)."""
    dtype = next(model.parameters()).dtype
    chunk, overlap = 45 * proto.SAMPLE_RATE, 2 * proto.SAMPLE_RATE
    starts = [s for s in range(0, len(audio), chunk - overlap) if s == 0 or len(audio) - s > overlap]
    parts = []
    for index, start in enumerate(starts):
        piece = audio[start : start + chunk]
        piece = (piece - piece.mean()) / (piece.std() + 1e-7)
        with torch.inference_mode():
            logits = model(torch.from_numpy(piece)[None].to(device, dtype)).logits[0]
            local = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        frames = round(local.shape[0] * overlap / max(1, len(piece)))
        left = frames // 2 if index else 0
        right = frames - frames // 2 if index + 1 < len(starts) else 0
        parts.append(local[left : local.shape[0] - right if right else None])
    return np.concatenate(parts)


def production(emission, audio, words, device):
    """The production decoder: vocal-activity penalty, then the paced per-letter floor."""
    adjusted = with_activity(emission, vocal_activity(audio, emission.shape[0]), ACTIVITY)
    plain = proto.baseline(adjusted, words, device)
    ids = [i for w in words for i in w["ids"]]
    return word_spans(words, min_duration_viterbi(adjusted, ids, sung_floor2(words, plain, PACE), device))


def summarize(rows, languages):
    """``proto.summarize`` plus crammed / held-first counts and per-language start-error stats in ms; ``languages``
    holds one language per row (song)."""
    out = proto.summarize(rows)
    crammed, words, held, runs = [sum(x) for x in zip(*(r["extra"] for r in rows))]
    out.update(crammed_under_60ms=crammed, words=words, held_first_runs=held, interjection_runs=runs)
    per_language = {}
    for language in sorted(set(languages)):
        errors = np.array([e for r, l in zip(rows, languages) if l == language for e in r["errors"]]) * 1000
        per_language[language] = {
            "songs": languages.count(language),
            "n": int(errors.size),
            "median_ms": round(float(np.median(errors)), 1),
            "mean_ms": round(float(errors.mean()), 1),
            "over_500ms": int((errors > 500).sum()),
        }
    out["per_language"] = per_language
    return out


def backing_summary(rs):
    """Aggregate per-song ``backing_metrics``: lead/backing overlap recall and backing / lead start-error stats (ms)."""
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


def main():
    """Score a checkpoint's single path (and, with --production, the production decoder) on the validation songs,
    caching its emissions; optionally pair it against a cached baseline. Prints a summary and writes
    --out/<name>.json. Needs CUDA when emissions must be computed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="model directory (ForcedAligner / Wav2Vec2ForCTC)")
    parser.add_argument("--name", default=None, help="cache / report name (default: <run>-<checkpoint>)")
    parser.add_argument("--data", default="validation.jsonl")
    parser.add_argument("--root", default=".")
    parser.add_argument("--songs", type=int, default=80)
    parser.add_argument("--cache", default="validation_eval/emissions")
    parser.add_argument("--out", default="validation_eval")
    parser.add_argument("--baseline", default="none", help="cached emissions to pair against ('none' to skip)")
    parser.add_argument("--production", action="store_true", help="also decode as production does (slower)")
    parser.add_argument("--bf16", action="store_true", help="run the model in bfloat16 (as the 1B trains)")
    args = parser.parse_args()
    songs = [json.loads(line) for line in open(Path(args.root) / args.data)]
    songs = [s for s in songs if s.get("source_timing_granularity") in ("word", "syllable")]
    random.Random(7).shuffle(songs)
    songs = songs[: args.songs] if args.songs else songs
    path = Path(args.checkpoint)
    name = args.name or (f"{path.parent.name}-{path.name}" if path.name.startswith("checkpoint-") else path.name)
    baseline = None if args.baseline in ("none", name) else args.baseline
    cache = Path(args.cache) / name
    cache.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    model = None
    rows = {"single path": [], "baseline single path": [], "production": [], "baseline production": []}
    backing = {key: [] for key in rows}
    languages = []
    started = time.time()
    for number, song in enumerate(songs, 1):
        words = song_words(song)
        cached = cache / f"{song['id']}.npy"
        audio = None
        if not cached.exists() or args.production:
            audio = proto.decode_audio(Path(args.root) / song["audio"])
        if cached.exists():
            emission = np.load(cached).astype(np.float32)
        else:
            if model is None:
                model = load_model(args.checkpoint, device, args.bf16)
            emission = emissions(model, audio, device)
            np.save(cached, emission.astype(np.float16))
        placed = {}
        try:
            placed["single path"] = proto.baseline(emission, words, device)
            if baseline:
                base_emission = np.load(Path(args.cache) / baseline / f"{song['id']}.npy").astype(np.float32)
                placed["baseline single path"] = proto.baseline(base_emission, words, device)
            if args.production:
                placed["production"] = production(emission, audio, words, device)
                if baseline:
                    placed["baseline production"] = production(base_emission, audio, words, device)
        except Exception as error:  # no CTC path (too many letters for the frames): skip for every row
            print(f"skip {song['id']}: {error}", flush=True)
            continue
        languages.append(song["language"])
        for key, spans in placed.items():
            result = proto.evaluate(words, spans)
            result["extra"] = crammed_and_held(words, spans)
            rows[key].append(result)
            if any(w["background"] for w in words):
                backing[key].append(backing_metrics(words, spans))
        if number % 10 == 0:
            print(f"{number}/{len(songs)} songs, {time.time() - started:.0f}s", flush=True)
    report = {key: summarize(r, languages) for key, r in rows.items() if r}
    report["lead_backing"] = {key: backing_summary(r) for key, r in backing.items() if r}
    if baseline:
        report["paired"] = ev.paired(rows["baseline single path"], rows["single path"])
        if args.production:
            report["paired_production"] = ev.paired(rows["baseline production"], rows["production"])
    report.update(
        checkpoint=args.checkpoint,
        name=name,
        baseline=baseline,
        split=args.data,
        songs=len(rows["single path"]),
        bf16=args.bf16,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(report, indent=2))
    print(f"== {name} vs {baseline} on {report['songs']} validation songs")
    for key in ("single path", "baseline single path", "production", "baseline production"):
        value = report.get(key)
        if not value:
            continue
        e, i = value["errors"], value["interjection_errors"]
        print(
            f"{key:24} start {e['median_ms']}/{e['mean_ms']} ms >500 {e['over_500ms']:5d} | interjections "
            f"{i['median_ms']}/{i['mean_ms']} ms >500 {i['over_500ms']} | crammed <60 ms {value['crammed_under_60ms']} | "
            f"held-first {value['held_first_runs']}/{value['interjection_runs']} | "
            + " ".join(f"{l}:{v['median_ms']}" for l, v in value["per_language"].items())
        )
        b = report["lead_backing"].get(key)
        if b:
            print(
                f"{'  backing':24} recovered {b['recovered']:.0%} of {b['lead_backing_pairs']} | backing start "
                f"{b['backing_median_ms']}/{b['backing_mean_ms']} ms >500 {b['backing_over_500']}/{b['backing_words']}"
            )
    if baseline:
        print("paired (baseline -> this):", json.dumps(report["paired"]))


if __name__ == "__main__":
    main()
