"""The validation harness with FA-Bench style metrics, on cached emissions (CPU is enough).

Same 80 VALIDATION songs, same production decoder ("decoder A": vocal-activity penalty 2, paced floor
x0.9) as validation.py. New here:

* F1@20/50 ms (forced alignment places exactly one start per reference word, so F1 at a tolerance
  is the share of words within it), signed bias (predicted - reference; mean and median) overall,
  per language and per word-onset class (stop_vl p/t/k, stop_vd b/d/g, fricative, nasal,
  approximant, vowel, interjection), and the per-song paired comparison on F1@50.
* Two clocks. ``legacy``: frame k at 12.47 ms + 20 k ms (it drifts 20 ms
  per 45 s chunk because each chunk yields one frame fewer than nominal). ``exact``: each
  frame at its true receptive-field centre (ctc_segmentation.stitched_frame_times). The legacy row
  keeps older reports comparable; everything new is judged on the exact clock.
* A full-mix row next to the stem row (``--audio mix``, for models trained on mixes). Mix
  emissions are cached under <cache>/<name>-mix/ and computed with the model when missing
  (``--device cpu`` is fine for the 300M; the 1B takes a few hours).
* Offset calibration (``--s2``): the production timing (k x duration/frames + 45 ms) against a constant
  offset, an onset-class offset table, and sub-frame start refinement, each calibrated on one
  half of the validation songs and scored on the other (two folds, never the test split).
* Edge head (``--edge-weight W``): if the checkpoint has an edge head, its logits are cached under
  <cache>/<name>-edge/ (or <name>-mix-edge/) and the duration-floor Viterbi pays the SOFA-style
  entry/stay costs with weight W; W = 0 is the control on the same emissions.
* Backbones: a ``wav2vec2-bert`` checkpoint (``train.ForcedAlignerBert``)
  is fed stacked fbank frames through its own processor and placed on the 279.5-sample clock; the
  backbone is recorded next to the cached emissions (``<cache>/<tag>/backbone.json``) so cached
  sets keep their exact clock without the model.
* Echo lines and interjections, reported separately (parenthesised echo lines like "(so lonely)"
  and repeated "oh-whoa" runs tend to start 0.3-2 s early): ``background`` (the token's
  ``vocal_role`` background or a parenthesised source span), ``repeated_interjection`` (an
  interjection whose neighbour in the line is the same word), each with the share of words more
  than 300 ms early / late.
* ``f1_at_300ms`` is the "correct onsets (< 0.3 s)" metric used for JamendoLyrics MultiLang
  (``--data <jamendo manifest>.jsonl --songs 0``).

usage (CPU):
  python -m echora_kara.evaluation.decode base --baseline none
  python -m echora_kara.evaluation.decode <run>-checkpoint-1000 --baseline base --s2
  python -m echora_kara.evaluation.decode base --checkpoint <model dir> --audio mix --device cpu
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from echora_kara.evaluation import validation as base
from echora_kara.evaluation import interval_decoder as proto

# Shared context between 45 s chunks (2 s by default); cached emissions of other overlaps get a -ovN tag.
OVERLAP_SECONDS = int(os.environ.get("ECHORA_KARA_CHUNK_OVERLAP_SECONDS", "2"))
DURATION_BAND_SECONDS = float(os.environ.get("ECHORA_KARA_DURATION_BAND_SECONDS", "15"))  # decoder C start window
# Decoder C2: one-sided duration cost beyond mu + k sigma (k from the env) plus decoder A's paced floors per token
DURATION_ONE_SIDED = (
    float(os.environ["ECHORA_KARA_DURATION_ONE_SIDED"]) if os.environ.get("ECHORA_KARA_DURATION_ONE_SIDED") else None
)
from echora_kara.engine.ctc_segmentation import (
    ONSET_CLASSES,
    align_with_duration_model,
    align_with_minimum_durations,
    align_with_source_priors,
    onset_class,
    paced_minimum_frames,
    refine_group_starts,
    stitched_frame_times,
    vocal_activity,
    with_vocal_activity,
)
from torchaudio.functional import TokenSpan  # noqa: E402

PACE, ACTIVITY, LETTER_CAP, GROUP_CAP, PAUSE = 0.9, 2.0, 2, 5, 20  # production decoder constants
VOWELS = {1, 5, 9, 15, 21, 25}
TOLERANCES_MS = (20, 50, 300)
PRODUCTION_OFFSET_MS = 45.0
EARLY_LATE_MS = 300.0  # the "0.3-2 s early" echo-line symptom
PARENTHESES = re.compile(r"[()\[\]（）]")


def mix_path(song):
    """The full mix of a stem-dataset song (audio-vocals/<pool>/<id>.flac -> audio/<pool>/<id>.m4a)."""
    if song.get("mix_audio"):
        return song["mix_audio"]
    stem = Path(song["audio"])
    return str(Path("audio") / stem.relative_to(stem.parts[0]).with_suffix(".m4a"))


def backbone_of(model):
    """``wav2vec2`` or ``wav2vec2-bert``: the backbone recorded on the model's frame clock (``wav2vec2`` if none)."""
    return getattr(getattr(model, "frame_clock", None), "backbone", "wav2vec2")


def load_model(checkpoint, device, bf16, fusion="on", fusion_model=None):
    """The aligner class picked from the checkpoint config (MMS/omniASR or w2v-BERT) plus, for
    w2v-BERT, its feature extractor. A fusion checkpoint also gets its frozen MuQ branch
    (``fusion="on"``) or runs as the plain encoder (``"gate0"``, the ablation)."""
    from echora_kara.train.train import load_aligner

    model = (
        load_aligner(
            checkpoint, fusion=fusion, fusion_model=fusion_model, dtype=torch.bfloat16 if bf16 else torch.float32
        )
        .to(device)
        .eval()
    )
    extractor = None
    if backbone_of(model) == "wav2vec2-bert":
        from transformers import SeamlessM4TFeatureExtractor

        extractor = SeamlessM4TFeatureExtractor.from_pretrained(checkpoint)
    return model, extractor


def model_outputs(model, audio, device, extractor=None, background=None):
    """Stitched log-probabilities and, when the model has an edge head, stitched edge logits.

    ``background`` (a list) receives the stitched log-probabilities of the background head when the
    checkpoint has one, for separate-path decoding of backing lines."""
    dtype = next(model.parameters()).dtype
    chunk, overlap = 45 * proto.SAMPLE_RATE, OVERLAP_SECONDS * proto.SAMPLE_RATE
    starts = [s for s in range(0, len(audio), chunk - overlap) if s == 0 or len(audio) - s > overlap]
    parts, edges = [], []
    for index, start in enumerate(starts):
        piece = audio[start : start + chunk]
        with torch.inference_mode():
            if extractor is not None:
                # w2v-BERT: stacked fbank pairs; the attention mask marks the valid frames.
                inputs = extractor(piece, sampling_rate=proto.SAMPLE_RATE, return_tensors="pt")
                valid = int(inputs["attention_mask"].sum())
                out = model(
                    input_features=inputs["input_features"].to(device, dtype),
                    attention_mask=inputs["attention_mask"].to(device),
                )
                out.logits = out.logits[:, :valid]
                if getattr(out, "edge_logits", None) is not None:
                    out.edge_logits = out.edge_logits[:, :valid]
                if getattr(out, "background_logits", None) is not None:
                    out.background_logits = out.background_logits[:, :valid]
            else:
                extra = {}
                if getattr(model, "fusion", None) is not None:
                    # The MuQ branch hears the raw chunk (the model resamples it to 24 kHz).
                    extra = {
                        "fusion_input_values": torch.from_numpy(piece)[None].to(device),
                        "fusion_lengths": torch.tensor([len(piece)]),
                    }
                piece = (piece - piece.mean()) / (piece.std() + 1e-7)
                out = model(torch.from_numpy(piece)[None].to(device, dtype), **extra)
            local = torch.log_softmax(out.logits[0].float(), dim=-1).cpu().numpy()
            edge = getattr(out, "edge_logits", None)
            edge = edge[0].float().cpu().numpy() if edge is not None else None
            backing = getattr(out, "background_logits", None)
            backing = torch.log_softmax(backing[0].float(), dim=-1).cpu().numpy() if backing is not None else None
        frames = round(local.shape[0] * overlap / max(1, len(piece)))
        left = frames // 2 if index else 0
        right = frames - frames // 2 if index + 1 < len(starts) else 0
        stop = local.shape[0] - right if right else None
        parts.append(local[left:stop])
        if edge is not None:
            edges.append(edge[left:stop])
        if backing is not None and background is not None:
            background.append(backing[left:stop])
    return np.concatenate(parts), (np.concatenate(edges) if edges else None)


def decode(
    emission,
    audio,
    words,
    device,
    edge=None,
    edge_weight=0.0,
    blank_free=False,
    duration=None,
    language=None,
    gap_cost=0.0,
):
    """Production decoder on the vendor code path: activity penalty, plain pass, paced floor.
    Returns (token spans per word, adjusted emission tensor)."""
    em = torch.from_numpy(emission).to(device)
    activity = vocal_activity(torch.from_numpy(audio), em.shape[0])
    adjusted = with_vocal_activity(em, activity, proto.BLANK, ACTIVITY)
    groups = [w["ids"] for w in words]
    plain = align_with_source_priors(
        adjusted, groups, blank=proto.BLANK, frame_shift_seconds=proto.FRAME, source_priors=[]
    )
    lines, ranges = {}, []
    for index, w in enumerate(words):
        lines.setdefault(w["line"], [index, index + 1])[1] = index + 1
    ranges = [tuple(v) for v in lines.values()]
    minimums = paced_minimum_frames(
        groups,
        plain,
        ranges,
        ratio=PACE,
        letter_cap=LETTER_CAP,
        group_cap=GROUP_CAP,
        pause_frames=PAUSE,
        vowel_ids=VOWELS,
    )
    if duration is not None:  # C: (model, weight) replaces the paced floor; C2 (one-sided) keeps it per token
        token_minimums = None
        if DURATION_ONE_SIDED is not None:
            token_minimums, offset = [], 0
            for group in groups:
                token_minimums.append(sum(minimums[offset : offset + len(group)]))
                offset += len(group)
        spans = align_with_duration_model(
            adjusted,
            groups,
            [w["text"] for w in words],
            ranges,
            plain,
            blank=proto.BLANK,
            frame_shift_seconds=proto.FRAME,
            model=duration[0],
            language=language,
            weight=duration[1],
            band_seconds=DURATION_BAND_SECONDS,
            one_sided_sigma=DURATION_ONE_SIDED,
            token_minimum_frames=token_minimums,
        )
        return spans, em
    spans = align_with_minimum_durations(
        adjusted,
        groups,
        blank=proto.BLANK,
        minimum_frames=minimums,
        edge_logits=torch.from_numpy(edge) if edge is not None else None,
        edge_weight=edge_weight,
        blank_free_tokens=blank_free,
        line_ranges=ranges,
        in_line_gap_cost=gap_cost,
    )
    return spans, em


def start_times(spans, clock, fractions=None):
    """Word-start seconds per word on a clock: 'legacy' (12.47 ms + 20 k ms), a tensor of exact
    frame times, or ('production', shift) = k x shift (the shipped decoder before its offset)."""
    out = []
    for index, group in enumerate(spans):
        frame = group[0].start
        fraction = fractions[index] if fractions is not None else 0.0
        if clock == "legacy":
            out.append(proto.frame_time(frame) + fraction * proto.FRAME)
        elif isinstance(clock, tuple):
            out.append((frame + fraction) * clock[1])
        else:
            out.append(float(clock[frame]) + fraction * proto.FRAME)
    return np.array(out)


def classes_of(words):
    """Word-onset class (``ctc_segmentation.onset_class``) of each word's text."""
    return [onset_class(w["text"]) for w in words]


def word_flags(song, words):
    """Per word (same filter as validation.song_words): ``background`` (background vocal
    role or a parenthesised source span: echo lines like "(so lonely)") and
    ``repeated_interjection`` (an interjection repeated by its neighbour in the same line,
    e.g. "oh-whoa oh-whoa")."""
    tokens = [t for t in song["alignment_tokens"] if t.get("ids") and not t.get("ignore")]
    assert len(tokens) == len(words)
    background = np.array(
        [
            t.get("vocal_role") == "background"
            or bool(PARENTHESES.search(str(t.get("source_text") or t.get("display_text") or "")))
            for t in tokens
        ]
    )
    interjection = np.array(
        [bool(proto.INTERJECTION.match(w["text"])) or onset_class(w["text"]) == "interjection" for w in words]
    )
    repeated = np.zeros(len(words), dtype=bool)
    for index, w in enumerate(words):
        if not interjection[index]:
            continue
        for other in (index - 1, index + 1):
            if 0 <= other < len(words) and words[other]["line"] == w["line"] and words[other]["text"] == w["text"]:
                repeated[index] = True
    return background, repeated


def stats(signed_ms):
    """Error statistics of signed start errors in ms (predicted - reference): absolute median / mean, count over
    500 ms, F1 at each tolerance in TOLERANCES_MS, signed bias and counts more than 300 ms early / late."""
    signed_ms = np.asarray(signed_ms, dtype=np.float64)
    absolute = np.abs(signed_ms)
    if not signed_ms.size:
        return {"n": 0}
    return {
        "n": int(signed_ms.size),
        "median_ms": round(float(np.median(absolute)), 1),
        "mean_ms": round(float(absolute.mean()), 1),
        "over_500ms": int((absolute > 500).sum()),
        **{f"f1_at_{t}ms": round(float((absolute <= t).mean()), 4) for t in TOLERANCES_MS},
        "bias_mean_ms": round(float(signed_ms.mean()), 1),
        "bias_median_ms": round(float(np.median(signed_ms)), 1),
        "early_over_300ms": int((signed_ms < -EARLY_LATE_MS).sum()),
        "late_over_300ms": int((signed_ms > EARLY_LATE_MS).sum()),
    }


def summarize(rows):
    """rows: list of dicts with signed (ms), language, classes, interjection flags, per song."""
    signed = np.concatenate([r["signed"] for r in rows])
    out = {"errors": stats(signed)}
    out["interjection_errors"] = stats(np.concatenate([r["signed"][r["interjection"]] for r in rows]))
    for key in ("background", "repeated_interjection"):
        if all(key in r for r in rows):
            out[f"{key}_errors"] = stats(np.concatenate([r["signed"][r[key]] for r in rows]))
            out[f"lead_without_{key}_errors"] = stats(np.concatenate([r["signed"][~r[key]] for r in rows]))
    out["per_language"] = {}
    for language in sorted({r["language"] for r in rows}):
        own = [r for r in rows if r["language"] == language]
        out["per_language"][language] = {"songs": len(own), **stats(np.concatenate([r["signed"] for r in own]))}
    out["per_onset_class"] = {}
    for name in ONSET_CLASSES:
        values = np.concatenate([r["signed"][r["classes"] == name] for r in rows])
        if values.size:
            out["per_onset_class"][name] = stats(values)
    out["songs"] = len(rows)
    out["song_f1_at_50ms"] = [round(float((np.abs(r["signed"]) <= 50).mean()), 4) for r in rows]
    out["song_median_ms"] = [round(float(np.median(np.abs(r["signed"]))), 1) for r in rows]
    return out


def paired(a, b):
    """Per-song comparison a -> b on F1@50 ms and the song median (robust to the > 500 ms tail)."""
    fa, fb = np.array(a["song_f1_at_50ms"]), np.array(b["song_f1_at_50ms"])
    ma, mb = np.array(a["song_median_ms"]), np.array(b["song_median_ms"])
    return {
        "songs": int(fa.size),
        "better_f1_50": int((fb > fa).sum()),
        "worse_f1_50": int((fb < fa).sum()),
        "median_song_f1_50_change": round(float(np.median(fb - fa)), 4),
        "better_median": int((mb < ma).sum()),
        "worse_median": int((mb > ma).sum()),
        "median_song_median_change_ms": round(float(np.median(mb - ma)), 1),
    }


def calibrate(rows, key, by):
    """Offsets (ms) that zero the median signed error of ``rows`` under residual ``key``: one
    constant (``by`` None), one per onset class ("class") or one per language ("language");
    groups with fewer than 30 words (classes) or 2 songs (languages) fall back to the constant."""
    residual = np.concatenate([r[key] for r in rows])
    constant = float(np.median(residual))
    table = {"*": constant}
    if by == "class":
        classes = np.concatenate([r["classes"] for r in rows])
        for name in ONSET_CLASSES:
            own = residual[classes == name]
            table[name] = float(np.median(own)) if own.size >= 30 else constant
    elif by == "language":
        for language in sorted({r["language"] for r in rows}):
            own = [r for r in rows if r["language"] == language]
            table[language] = float(np.median(np.concatenate([r[key] for r in own]))) if len(own) >= 2 else constant
    return table


def apply_offsets(row, key, table, by):
    """Residuals ``row[key]`` (ms) minus the calibrated offset of each word from a ``calibrate`` table (per onset
    class, per language or the constant ``"*"``)."""
    keys = row["classes"] if by == "class" else [row["language"]] * row["n"] if by == "language" else []
    offsets = np.array([table.get(k, table["*"]) for k in keys]) if len(keys) else table["*"]
    return row[key] - offsets  # residual = predicted - reference, so subtracting the offset shifts predictions later


def main():
    """Score a cached emission set (computing missing emissions from --checkpoint) with the production decoder and its
    variants on the validation songs, on the legacy, exact and production clocks; optionally pair it against a
    baseline set and calibrate offsets (--s2). Prints a summary and writes --out/<name>[-mix|-both].json."""
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="cached emission set under --cache (e.g. <run>-checkpoint-1000)")
    parser.add_argument("--checkpoint", default=None, help="model directory, needed only to compute missing emissions")
    parser.add_argument("--data", default="validation.jsonl")
    parser.add_argument("--root", default=".")
    parser.add_argument("--songs", type=int, default=80)
    parser.add_argument("--cache", default="validation_eval/emissions")
    parser.add_argument("--out", default="validation_eval/decode")
    parser.add_argument("--baseline", default="none", help="cached emission set to pair against ('none')")
    parser.add_argument(
        "--audio",
        choices=("stem", "mix", "both"),
        default="stem",
        help="stem: the Roformer vocal stem (cached emissions); mix: the full mix; both: paired on songs with a mix",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--s2", action="store_true", help="offset calibration, report key 's2' (two folds over the validation songs)"
    )
    parser.add_argument(
        "--edge-weight",
        type=float,
        nargs="*",
        default=[],
        help="edge head: decode again with the cached edge logits at these weights (0 is the control)",
    )
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--duration-model", default=None, help="decoder C: a duration-model JSON (rows '<kind> C w=W')")
    parser.add_argument(
        "--duration-weight",
        type=float,
        nargs="*",
        default=[0.0, 1.0, 3.0],
        help="decoder C weights scored (0: blank-free, floor-free semi-Markov)",
    )
    parser.add_argument(
        "--gap-cost",
        type=float,
        nargs="*",
        default=[],
        help="decoder A plus a cost per blank frame inside a line (rows '<kind> A g=G')",
    )
    parser.add_argument(
        "--blank-free",
        action="store_true",
        help="decoder A: also score each row with no blank between letters of a word (rows '<kind> A')",
    )
    parser.add_argument(
        "--any-granularity",
        action="store_true",
        help="keep rows without source_timing_granularity (compact manifests drop the field; "
        "use only when every song is word-timed)",
    )
    parser.add_argument(
        "--backbone",
        choices=("auto", "wav2vec2", "wav2vec2-bert"),
        default="auto",
        help="exact clock of the cached emissions when no --checkpoint is loaded (auto: the "
        "backbone.json written next to the cache, else wav2vec2)",
    )
    parser.add_argument(
        "--fusion",
        choices=("on", "gate0"),
        default="on",
        help="fusion checkpoints: on = with the frozen MuQ branch (as trained); gate0 = the "
        "adapter forced to zero, i.e. the plain encoder (cache it under another name, e.g. "
        "<tag>-gate0). Ignored for checkpoints without a fusion record.",
    )
    parser.add_argument("--fusion-model", default=None, help="MuQ hub id / snapshot overriding the checkpoint's")
    parser.add_argument(
        "--ids",
        default=None,
        help="file with one song id per line: score exactly these songs in this order instead of "
        "the Random(7) draw (e.g. a held-out list without songs that have whole-song offsets)",
    )
    args = parser.parse_args()
    duration_model = json.loads(Path(args.duration_model).read_text()) if args.duration_model else None
    if args.threads:
        torch.set_num_threads(args.threads)
    songs = [json.loads(line) for line in open(Path(args.root) / args.data)]
    songs = [
        s
        for s in songs
        if s.get("source_timing_granularity") in ("word", "syllable")
        or (args.any_granularity and "source_timing_granularity" not in s)
    ]
    if args.ids:
        wanted = [line.strip() for line in open(args.ids) if line.strip() and not line.startswith("#")]
        by_id = {s["id"]: s for s in songs}
        missing = [i for i in wanted if i not in by_id]
        if missing:
            raise SystemExit(f"--ids: {len(missing)} ids are not in {args.data}: {missing[:5]}")
        songs = [by_id[i] for i in wanted]
    else:
        random.Random(7).shuffle(songs)
    songs = songs[: args.songs] if args.songs else songs
    kinds = ["stem", "mix"] if args.audio == "both" else [args.audio]
    baseline = None if args.baseline in ("none", args.name) else args.baseline
    device = args.device
    model, extractor = None, None
    backbones = {}  # cache tag -> backbone of its emissions
    results = {}  # row name -> list of per-song dicts
    skipped = []
    started = time.time()
    for number, song in enumerate(songs, 1):
        words = base.song_words(song)
        classes = np.array(classes_of(words))
        reference = np.array([w["start"] for w in words])
        interjection = np.array([bool(proto.INTERJECTION.match(w["text"])) for w in words])
        background, repeated = word_flags(song, words)
        per_song = {}
        for kind in kinds:
            audio_rel = song["audio"] if kind == "stem" else mix_path(song)
            audio_path = Path(args.root) / audio_rel
            if not audio_path.exists():
                continue
            sets = [(args.name, False)] + ([(baseline, True)] if baseline else [])
            for set_name, is_baseline in sets:
                tag = set_name if kind == "stem" else f"{set_name}-mix"
                tag += f"-ov{OVERLAP_SECONDS}" if OVERLAP_SECONDS != 2 else ""
                cached = Path(args.cache) / tag / f"{song['id']}.npy"
                edge_cached = Path(args.cache) / f"{tag}-edge" / f"{song['id']}.npy"
                audio = proto.decode_audio(audio_path)
                sidecar = Path(args.cache) / tag / "backbone.json"
                if tag not in backbones:
                    backbones[tag] = (
                        json.loads(sidecar.read_text())["backbone"]
                        if sidecar.exists()
                        else args.backbone
                        if args.backbone != "auto"
                        else "wav2vec2"
                    )
                if cached.exists():
                    emission = np.load(cached).astype(np.float32)
                    edge = np.load(edge_cached).astype(np.float32) if edge_cached.exists() else None
                else:
                    if is_baseline or not args.checkpoint:
                        continue
                    if model is None:
                        model, extractor = load_model(
                            args.checkpoint, device, args.bf16, args.fusion, args.fusion_model
                        )
                        backbones[tag] = backbone_of(model)
                    backing = []
                    emission, edge = model_outputs(model, audio, device, extractor, backing)
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    np.save(cached, emission.astype(np.float16))
                    if backing:
                        # Background-head log-probabilities (checkpoints with that head), same clock as the emission.
                        backing_cached = Path(args.cache) / f"{tag}-background" / f"{song['id']}.npy"
                        backing_cached.parent.mkdir(parents=True, exist_ok=True)
                        np.save(backing_cached, np.concatenate(backing).astype(np.float16))
                    if not sidecar.exists():
                        record = {"backbone": backbone_of(model), "checkpoint": args.checkpoint}
                        if getattr(model, "fusion", None) is not None:
                            record["fusion"] = args.fusion
                        sidecar.write_text(json.dumps(record) + "\n")
                    if edge is not None:
                        edge_cached.parent.mkdir(parents=True, exist_ok=True)
                        np.save(edge_cached, edge.astype(np.float16))
                exact = stitched_frame_times(len(audio), backbone=backbones[tag], overlap_seconds=OVERLAP_SECONDS)
                if exact.shape[0] != emission.shape[0]:
                    print(
                        f"clock mismatch {song['id']} {tag}: {exact.shape[0]} times, {emission.shape[0]} frames",
                        flush=True,
                    )
                    exact = None
                shift = len(audio) / proto.SAMPLE_RATE / emission.shape[0]
                weights = (
                    [0.0] + [w for w in args.edge_weight if w > 0] if edge is not None and not is_baseline else [0.0]
                )
                variants = [(False, None)] + ([(True, None)] if args.blank_free and not is_baseline else [])
                variants = [(b, d, 0.0) for b, d in variants]
                if duration_model is not None and not is_baseline:
                    variants += [(True, w, 0.0) for w in args.duration_weight]
                if not is_baseline:
                    variants += [(True, None, g) for g in args.gap_cost]
                for weight, (blank_free, duration_weight, gap_cost) in [(w, v) for w in weights for v in variants]:
                    if (duration_weight is not None or gap_cost) and weight:
                        continue
                    try:
                        spans, em = decode(
                            emission,
                            audio,
                            words,
                            device,
                            edge,
                            weight,
                            blank_free,
                            (duration_model, duration_weight) if duration_weight is not None else None,
                            song["language"],
                            gap_cost,
                        )
                    except Exception as error:
                        skipped.append(f"{song['id']} {tag}: {error}")
                        continue
                    prefix = "baseline " if is_baseline else ""
                    label = (
                        f"{prefix}{kind}"
                        + (f" edge w={weight:g}" if weight else "")
                        + (
                            f" C{'2' if DURATION_ONE_SIDED is not None else ''} w={duration_weight:g}"
                            if duration_weight is not None
                            else f" A g={gap_cost:g}"
                            if gap_cost
                            else " A"
                            if blank_free
                            else ""
                        )
                    )
                    row = {
                        "id": song["id"],
                        "language": song["language"],
                        "classes": classes,
                        "interjection": interjection,
                        "background": background,
                        "repeated_interjection": repeated,
                        "n": len(words),
                    }
                    row["legacy"] = (start_times(spans, "legacy") - reference) * 1000
                    row["production45"] = (
                        start_times(spans, ("production", shift)) + PRODUCTION_OFFSET_MS / 1000 - reference
                    ) * 1000
                    if exact is not None:
                        fractions = refine_group_starts(em, spans, blank=proto.BLANK)
                        row["exact"] = (start_times(spans, exact) - reference) * 1000
                        row["exact_refined"] = (start_times(spans, exact, fractions) - reference) * 1000
                    per_song[label] = row
        if "stem" in per_song or "mix" in per_song:
            for label, row in per_song.items():
                results.setdefault(label, []).append(row)
        if number % 10 == 0:
            print(f"{number}/{len(songs)} songs, {time.time() - started:.0f}s", flush=True)
    if args.audio == "both":  # pair stem and mix on the songs that have both
        ids = {r["id"] for r in results.get("mix", [])}
        for label in list(results):
            if label.startswith("stem") or label.startswith("baseline stem"):
                results[label] = [r for r in results[label] if r["id"] in ids]
    report = {
        "name": args.name,
        "baseline": baseline,
        "audio": args.audio,
        "songs": len(results.get(kinds[0], [])),
        "skipped": skipped,
        "rows": {},
    }
    for label, rows in results.items():
        for clock in ("legacy", "exact", "exact_refined", "production45"):
            if all(clock in r for r in rows):
                report["rows"][f"{label} | {clock}"] = summarize([{**r, "signed": r[clock]} for r in rows])
    if baseline:
        report["paired"] = {}
        for label in results:
            if label.startswith("baseline"):
                continue
            other = "baseline " + label.split(" edge")[0]
            if other in results:
                for clock in ("legacy", "exact"):
                    a, b = report["rows"].get(f"{other} | {clock}"), report["rows"].get(f"{label} | {clock}")
                    if a and b and a["songs"] == b["songs"]:
                        report["paired"][f"{label} | {clock}"] = paired(a, b)
    if args.s2:
        report["s2"] = {}
        for label, rows in results.items():
            if label.startswith("baseline") or not all("exact" in r for r in rows):
                continue
            folds = [[r for i, r in enumerate(rows) if i % 2 == f] for f in (0, 1)]
            variants = {
                "production +45 ms (legacy clock)": None,
                "exact clock + constant": ("exact", None),
                "exact clock + onset-class table": ("exact", "class"),
                "exact clock + per-language table": ("exact", "language"),
                "exact clock + sub-frame + constant": ("exact_refined", None),
                "exact clock + sub-frame + onset-class table": ("exact_refined", "class"),
                "exact clock + sub-frame + per-language table": ("exact_refined", "language"),
            }
            s2 = {}
            for name, spec in variants.items():
                scored, tables = [], []
                for fold in (0, 1):
                    train, test = folds[1 - fold], folds[fold]
                    if spec is None:
                        scored += [{**r, "signed": r["production45"]} for r in test]
                        continue
                    key, by = spec
                    table = calibrate(train, key, by)
                    tables.append({k: round(v, 1) for k, v in table.items()})
                    scored += [{**r, "signed": apply_offsets(r, key, table, by)} for r in test]
                summary = summarize(scored)
                s2[name] = {
                    "errors": summary["errors"],
                    "per_language": {
                        l: {k: v[k] for k in ("median_ms", "bias_median_ms", "f1_at_50ms")}
                        for l, v in summary["per_language"].items()
                    },
                    "per_onset_class": {
                        c: {k: v[k] for k in ("n", "median_ms", "bias_median_ms", "f1_at_50ms")}
                        for c, v in summary["per_onset_class"].items()
                    },
                    "fold_offsets_ms": tables,
                    "song_f1_at_50ms": summary["song_f1_at_50ms"],
                    "song_median_ms": summary["song_median_ms"],
                }
            report["s2"][label] = s2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = {"stem": "", "mix": "-mix", "both": "-both"}[args.audio]
    (out / f"{args.name}{suffix}.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"== {args.name} ({args.audio}) vs {baseline} on {report['songs']} validation songs; skipped {len(skipped)}")
    for label, value in report["rows"].items():
        e = value["errors"]
        print(
            f"{label:42} median {e['median_ms']:6.1f} mean {e['mean_ms']:6.1f} >500 {e['over_500ms']:4d} "
            f"F1@20 {e['f1_at_20ms']:.3f} F1@50 {e['f1_at_50ms']:.3f} bias mean/med {e['bias_mean_ms']:+6.1f}/{e['bias_median_ms']:+6.1f} | "
            + " ".join(f"{l}:{v['bias_median_ms']:+.0f}" for l, v in value["per_language"].items())
        )
    for label, value in report["rows"].items():
        if label.endswith("| exact"):
            print(
                f"  onset classes ({label}): "
                + " ".join(f"{c}:{v['bias_median_ms']:+.0f}({v['n']})" for c, v in value["per_onset_class"].items())
            )
            for key in ("background", "repeated_interjection", "interjection"):
                e = value.get(f"{key}_errors", {})
                if e.get("n"):
                    print(
                        f"  {key:22} ({label}): n {e['n']:5d} median {e['median_ms']:6.1f} mean {e['mean_ms']:6.1f} "
                        f"bias med {e['bias_median_ms']:+6.1f} early>300 {e['early_over_300ms']} late>300 {e['late_over_300ms']} "
                        f"F1@50 {e['f1_at_50ms']:.3f} <300ms {e['f1_at_300ms']:.3f}"
                    )
            e = value["errors"]
            print(
                f"  all words ({label}): <300 ms {e['f1_at_300ms']:.3f} (JamendoLyrics onset metric), "
                f"early>300 {e['early_over_300ms']} late>300 {e['late_over_300ms']}"
            )
    for label, value in report.get("paired", {}).items():
        print(f"paired {label}: {json.dumps(value)}")
    for label, s2 in report.get("s2", {}).items():
        print(f"== offset calibration ({label}), two folds over the validation songs")
        for name, value in s2.items():
            e = value["errors"]
            print(
                f"{name:46} median {e['median_ms']:6.1f} mean {e['mean_ms']:6.1f} F1@20 {e['f1_at_20ms']:.3f} F1@50 {e['f1_at_50ms']:.3f} "
                f"bias med {e['bias_median_ms']:+5.1f} | offsets {value['fold_offsets_ms']}"
            )


if __name__ == "__main__":
    main()
