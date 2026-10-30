#!/usr/bin/env python3
"""Per-song vocal-activity arrays on the 20 ms label clock, for the manifest builder's tail trim.

The aligner trains on ORIGINAL MIXES and must need no separation at inference, so activity is an
*offline label-construction* input only: ``build_manifest.py --label-policy v2 --activity-dir STORE``
shortens each fragment's tail where the singer is not active.

Input layout under ``--root``: ``audio/<id>.m4a`` (full mixes) and, where available,
``vocals/<id>.flac`` (a separated vocal stem). Sources, in order of trust:

- ``stem_energy`` (CPU): the vocal stem's loudness in dB per frame scaled between the song's 10th and
  90th percentile, exactly ``ctc_segmentation.vocal_activity``; trimmed frames are labelled blank
  (the stem says nobody sings).
- ``vad_head`` (GPU): a w2v-BERT model's auxiliary VAD head (``aux_heads.safetensors`` in the model
  directory) run on the full MIX in 45 s chunks as ``Wav2Vec2ForcedAligner.infer`` does. An estimate,
  so frames it trims are labelled *ignore*, not blank. It is not reliable enough for the tail trim (most
  frames it would trim are sung according to the stem), so the builder trims only with ``stem-energy``;
  the VAD arrays serve decoding and evaluation.

Store (``--store``): ``<store>/<source>/<id>.npy``, float16, one value per 20 ms label frame (frame t
covers samples [320 t, 320 t + 400) at 16 kHz exactly as the trainer's labels), with
``<source>/index.json`` (model id/revision, frame clock, parameters, value semantics). Sources:

- ``stem-energy``: the stem's per-frame loudness scaled between its 10th and 90th percentile, [0, 1].
- ``vad-<encoder>-<revision>-auxhead-<revision>``: the raw VAD-head sigmoid on the mix, [0, 1]
  (unscaled; ``scale_percentiles`` gives the per-song scaling).

A directory of ``.npz`` arrays (uint8 activity, method, version, source) is read as well.
Every command is incremental: a recording whose array exists is skipped unless ``--force``.

``compare`` reports, on songs that have both, the agreement of two sources (AUC of each against
the timed fragments, frame agreement at the trim threshold, and how often the tail-trim decision
coincides).

  python -m echora_kara.data.vocal_activity stems --root DATA --store ACTIVITY --workers 2
  python -m echora_kara.data.vocal_activity vad --output MANIFEST_DIR --root DATA --store ACTIVITY \\
      --model w2v-bert-2.0-music-lyrics-ctc --device cuda
  python -m echora_kara.data.vocal_activity compare --output MANIFEST_DIR --a STEM_NPZ --b VAD_NPZ
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from collections import Counter

import numpy as np

VERSION = "activity_v1"
SAMPLE_RATE = 16000
HOP, WINDOW = 320, 400
METHODS = ("stem_energy", "vad_head")
# Activity below this (after the 10/90 percentile scaling) counts as "not singing" for the trim.
TRIM_THRESHOLD = 0.35

STEM_SOURCE = "stem-energy"
FRAME_CLOCK = {
    "sample_rate": SAMPLE_RATE,
    "hop_samples": HOP,
    "window_samples": WINDOW,
    "frame": "frame t covers samples [320 t, 320 t + 400) of the 16 kHz mono decode (the trainer's label clock)",
}


def vad_source_id(model: Path) -> str:
    """Store name of a VAD-head source: encoder + head identity (CONVERSION.json / HF revisions)."""
    model = Path(model)
    encoder, head = model.name, "aux"
    conversion = model / "CONVERSION.json"
    if conversion.exists():
        record = json.loads(conversion.read_text())
        encoder = f"{Path(record.get('source_dir', model.name)).name}-{record.get('source_sha256', '')[:8]}"
    import hashlib

    aux = model / "aux_heads.safetensors"
    if aux.exists():
        head = hashlib.sha256(aux.read_bytes()).hexdigest()[:8]
    return f"vad-{encoder}-auxhead-{head}"


def store_dir(source: str, store: Path) -> Path:
    """Directory of one source in the store, created if missing."""
    path = Path(store) / source
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_store(source: str, isrc: str, values: np.ndarray, store: Path) -> None:
    """Write one recording's activity as a float16 ``.npy`` in the store (atomically, via a temporary file)."""
    path = store_dir(source, store) / f"{isrc}.npy"
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, np.asarray(values, dtype=np.float16))
    temporary.replace(path)


def load_store(source: str, isrc: str, store: Path) -> np.ndarray | None:
    """One recording's stored activity as float32, or None when the source has no array for it."""
    path = Path(store) / source / f"{isrc}.npy"
    return np.load(path).astype(np.float32) if path.exists() else None


def update_index(source: str, record: dict, store: Path) -> None:
    """Merge ``record`` into the source's ``index.json`` and refresh its version, frame clock, song count and time."""
    directory = store_dir(source, store)
    path = directory / "index.json"
    index = json.loads(path.read_text()) if path.exists() else {}
    index.update(record)
    index.update(
        source=source,
        version=VERSION,
        frame_clock=FRAME_CLOCK,
        dtype="float16",
        songs=sum(1 for _ in directory.glob("*.npy")),
        updated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    path.write_text(json.dumps(index, indent=2) + "\n")


def decode(path: Path) -> np.ndarray:
    """Decode an audio file with ffmpeg (one thread) to 16 kHz mono float32 samples."""
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "1",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def scale_percentiles(values: np.ndarray) -> np.ndarray:
    """Values scaled so the 10th percentile maps to 0 and the 90th to 1, clipped to [0, 1]."""
    low, high = np.percentile(values, 10), np.percentile(values, 90)
    return np.clip((values - low) / max(high - low, 1e-6), 0.0, 1.0)


def stem_energy_activity(waveform: np.ndarray) -> np.ndarray:
    """Loudness (dB) per label frame of a vocal stem, scaled between the 10th and 90th percentile."""
    count = max(1, (len(waveform) - WINDOW) // HOP + 1)
    padded = np.pad(waveform, (0, max(0, count * HOP + WINDOW - len(waveform))))
    index = np.arange(WINDOW)[None, :] + HOP * np.arange(count)[:, None]
    loudness = 10 * np.log10((padded[index] ** 2).mean(1) + 1e-10)
    return scale_percentiles(loudness).astype(np.float32)


def to_uint8(activity: np.ndarray) -> np.ndarray:
    """Activity in [0, 1] quantised to uint8 (0-255) for ``.npz`` caches."""
    return np.clip(np.round(activity * 255), 0, 255).astype(np.uint8)


def load_activity_from(directory: Path, isrc: str) -> dict | None:
    """The manifest builder's activity for one song: {"activity": float32 [0,1], "method": str, ...}
    from the store's ``stem-energy`` array when ``directory`` is a store root (has a ``stem-energy``
    folder), else from an ``.npz`` directory; None when the song has no array."""
    directory = Path(directory)
    if (directory / STEM_SOURCE).is_dir():
        values = load_store(STEM_SOURCE, isrc, directory)
        return (
            None
            if values is None
            else {"activity": values, "method": "stem_energy", "version": VERSION, "source": STEM_SOURCE}
        )
    path = directory / f"{isrc}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {
            "activity": data["activity"].astype(np.float32) / 255.0,
            "method": str(data["method"]),
            "version": str(data["version"]),
            "source": str(data["source"]),
        }


# ---------------------------------------------------------------------------------------------
# stem_energy


def _stem_worker(task: tuple[str, str, str]) -> tuple[str, str]:
    isrc, stem, store = task
    try:
        write_store(STEM_SOURCE, isrc, stem_energy_activity(decode(Path(stem))), Path(store))
        return isrc, "ok"
    except Exception as error:  # noqa: BLE001 - reported, never fatal
        return isrc, f"error {type(error).__name__}: {error}"


def stems(args) -> None:
    """Compute ``stem-energy`` arrays for every ``<root>/vocals/*.flac`` not yet in the store (or all with
    ``--force``), in a process pool, then update the source's index."""
    from multiprocessing import Pool

    vocals = args.root / "vocals"
    wanted = set(line.strip() for line in args.ids.open() if line.strip()) if args.ids else None
    todo = []
    for path in sorted(vocals.glob("*.flac")):
        isrc = path.stem.upper()
        if wanted is not None and isrc not in wanted:
            continue
        existing = store_dir(STEM_SOURCE, args.store) / f"{isrc}.npy"
        if existing.exists() and not args.force:
            continue
        todo.append((isrc, str(path), str(args.store)))
    print(
        f"stems: {len(todo)} to compute ({sum(1 for _ in store_dir(STEM_SOURCE, args.store).glob('*.npy'))} cached)",
        flush=True,
    )
    started = time.time()
    counts = {}
    with Pool(args.workers) as pool:
        for number, (isrc, status) in enumerate(pool.imap_unordered(_stem_worker, todo, chunksize=16), 1):
            counts[status.split(" ")[0]] = counts.get(status.split(" ")[0], 0) + 1
            if status.startswith("error"):
                print(isrc, status, flush=True)
            if number % 2000 == 0:
                print(f"  {number}/{len(todo)} {counts} ({time.time() - started:.0f}s)", flush=True)
    update_index(
        STEM_SOURCE,
        {
            "method": "stem_energy",
            "input": "<root>/vocals/<id>.flac (vocal stem, 16 kHz mono decode)",
            "values": "per-frame loudness (dB of the 400-sample window) scaled between the song's 10th and 90th percentile, clipped to [0, 1]",
            "function": "vocal_activity.stem_energy_activity (= ctc_segmentation.vocal_activity)",
        },
        args.store,
    )
    print(json.dumps({"method": "stem_energy", **counts, "seconds": round(time.time() - started)}))


def migrate(args) -> None:
    """Copy an ``.npz`` directory (uint8) into the store as float16 (skips existing)."""
    counts = Counter()
    for path in sorted(Path(args.source).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            method = str(data["method"])
            if method != "stem_energy":
                counts[f"skipped_{method}"] += 1
                continue
            target = store_dir(STEM_SOURCE, args.store) / f"{path.stem}.npy"
            if target.exists() and not args.force:
                counts["exists"] += 1
                continue
            write_store(STEM_SOURCE, path.stem, data["activity"].astype(np.float32) / 255.0, args.store)
            counts["migrated"] += 1
    update_index(STEM_SOURCE, {"method": "stem_energy", "migrated_from_uint8": True}, args.store)
    print(json.dumps(counts))


# ---------------------------------------------------------------------------------------------
# vad_head (needs torch)

CHUNK_SECONDS, OVERLAP_SECONDS = 45, 2


def vad_chunks(waveform: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """45 s chunks with 2 s overlap (as Wav2Vec2ForcedAligner.infer)."""
    chunk_samples, overlap_samples = CHUNK_SECONDS * SAMPLE_RATE, OVERLAP_SECONDS * SAMPLE_RATE
    step = chunk_samples - overlap_samples
    starts = list(range(0, max(1, len(waveform) - overlap_samples), step)) or [0]
    return [(start, waveform[start : start + chunk_samples]) for start in starts]


def vad_features(waveform: np.ndarray, extractor) -> list[dict]:
    """Feature-extractor inputs per chunk (CPU; runs in the loader threads)."""
    return [
        {"length": len(chunk), **extractor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="np")}
        for _, chunk in vad_chunks(waveform)
    ]


def vad_head_activity(
    features: list[dict], model, heads, device: str, dtype=None, batch: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """(scaled activity, raw sigmoid) per 20 ms frame from the w2v-BERT VAD head on the mix.

    Chunks of one song are run ``batch`` at a time (padded, attention-masked); half of each 2 s
    overlap is kept on each side, as the aligner does. ``dtype`` (e.g. torch.bfloat16) runs the
    encoder under autocast.
    """
    import torch

    overlap_samples = OVERLAP_SECONDS * SAMPLE_RATE
    logits_per_chunk = []
    for first in range(0, len(features), batch):
        group = features[first : first + batch]
        width = max(f["input_features"].shape[1] for f in group)
        values = np.zeros((len(group), width, group[0]["input_features"].shape[2]), dtype=np.float32)
        mask = np.zeros((len(group), width), dtype=np.int64)
        for row, f in enumerate(group):
            n = f["input_features"].shape[1]
            values[row, :n] = f["input_features"][0]
            mask[row, :n] = f["attention_mask"][0]
        inputs = {
            "input_features": torch.from_numpy(values).to(device),
            "attention_mask": torch.from_numpy(mask).to(device),
        }
        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=dtype is not None),
        ):
            hidden = model(**inputs).last_hidden_state
            logits = heads["vad"](hidden.float())[..., 0].float().cpu().numpy()
        for row, f in enumerate(group):
            logits_per_chunk.append(logits[row, : int(f["attention_mask"].sum())])
    parts = []
    for index, (f, logits) in enumerate(zip(features, logits_per_chunk)):
        overlap_frames = round(len(logits) * overlap_samples / max(1, f["length"]))
        trim_left = overlap_frames // 2 if index else 0
        trim_right = overlap_frames - overlap_frames // 2 if index + 1 < len(features) else 0
        parts.append(logits[trim_left : len(logits) - trim_right if trim_right else None])
    raw = 1 / (1 + np.exp(-np.concatenate(parts)))
    return scale_percentiles(raw).astype(np.float32), raw.astype(np.float32)


def vad_index_record(args) -> dict:
    """Index metadata of a VAD-head source (model, chunking, precision, value semantics) for ``update_index``."""
    return {
        "method": "vad_head",
        "model": str(args.model),
        "encoder": vad_source_id(args.model),
        "head": "aux_heads.safetensors 'vad' in the model directory",
        "input": "<root>/audio/<id>.m4a (full mix, 16 kHz mono)",
        "values": "raw sigmoid of the VAD logit per frame (unscaled; scale_percentiles for the per-song scaling)",
        "chunking": {"chunk_seconds": CHUNK_SECONDS, "overlap_seconds": OVERLAP_SECONDS, "batch": args.batch},
        "precision": args.dtype,
        "note": "not used for the tail trim; kept for decoding and evaluation",
    }


def vad(args) -> None:
    """Run the VAD head on the full mixes of the manifest's train/validation songs (or ``--ids``) on ``--device``.

    Writes the raw sigmoid per song to the store, or scaled uint8 ``.npz`` arrays to ``--activity-dir``; songs that
    already have an array are skipped unless ``--force``. Decoding and feature extraction run in threads ahead of
    the GPU."""
    import torch
    from safetensors import safe_open
    from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

    model = Wav2Vec2BertModel.from_pretrained(args.model).to(args.device).eval()
    extractor = AutoFeatureExtractor.from_pretrained(args.model)
    with safe_open(str(Path(args.model) / "aux_heads.safetensors"), framework="pt", device="cpu") as handle:
        state = {k: handle.get_tensor(k) for k in handle.keys()}
    heads = torch.nn.ModuleDict({"vad": torch.nn.Linear(state["vad.weight"].shape[1], 1)})
    heads["vad"].load_state_dict({"weight": state["vad.weight"], "bias": state["vad.bias"]})
    heads.to(args.device).eval()
    if args.threads:
        torch.set_num_threads(args.threads)
    source = vad_source_id(args.model)
    if not args.activity_dir and not args.store:
        raise SystemExit("vad needs --store or --activity-dir")
    target = Path(args.activity_dir) if args.activity_dir else store_dir(source, args.store)
    target.mkdir(parents=True, exist_ok=True)
    manifest_ids = []
    if args.ids:
        manifest_ids = [line.strip() for line in args.ids.open() if line.strip()]
    else:
        for split in ("train", "validation"):
            path = args.output / f"{split}.jsonl"
            source = (
                path
                if path.exists()
                else (Path(args.manifest_fallback) / f"{split}.jsonl" if args.manifest_fallback else None)
            )
            if source and source.exists():
                manifest_ids += [json.loads(line)["id"] for line in source.open() if line.strip()]
    # Never overwrite an existing array (a stem_energy array is the more trusted source) unless --force.
    suffix = ".npz" if args.activity_dir else ".npy"
    todo = [isrc for isrc in dict.fromkeys(manifest_ids) if args.force or not (target / f"{isrc}{suffix}").exists()]
    if args.only_without_stem:
        todo = [isrc for isrc in todo if not (args.root / "vocals" / f"{isrc}.flac").exists()]
    todo = [isrc for isrc in todo if (args.root / "audio" / f"{isrc}.m4a").exists()]
    if args.limit:
        todo = todo[: args.limit]
    print(f"vad_head: {len(todo)} songs on {args.device} with {args.model} -> {target}", flush=True)
    from concurrent.futures import ThreadPoolExecutor

    def load(isrc):
        try:
            waveform = decode(args.root / "audio" / f"{isrc}.m4a")
            return isrc, (len(waveform), vad_features(waveform, extractor)), None
        except Exception as error:  # noqa: BLE001
            return isrc, None, f"{type(error).__name__}: {error}"

    dtype = {"float32": None, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    started = time.time()
    audio_seconds = 0.0
    errors = 0
    with ThreadPoolExecutor(max_workers=args.decode_workers) as pool:

        def bounded(items, depth):
            """pool.map with at most ``depth`` songs decoded ahead (features stay in memory)."""
            from collections import deque

            pending = deque()
            for item in items:
                pending.append(pool.submit(load, item))
                if len(pending) >= depth:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()

        for number, (isrc, waveform, error) in enumerate(bounded(todo, 2 * args.decode_workers), 1):
            if error is None:
                try:
                    samples, features = waveform
                    activity, raw = vad_head_activity(features, model, heads, args.device, dtype, args.batch)
                except Exception as failure:  # noqa: BLE001
                    error = f"{type(failure).__name__}: {failure}"
            if error is not None:
                errors += 1
                print(f"  {isrc} error {error[:200]}", flush=True)
                continue
            audio_seconds += samples / SAMPLE_RATE
            if args.activity_dir:
                np.savez_compressed(
                    target / f"{isrc}.npz",
                    activity=to_uint8(activity),
                    method=np.array("vad_head"),
                    version=np.array(VERSION),
                    source=np.array(str(args.model)),
                )
            else:
                write_store(source, isrc, raw, args.store)
            if not args.activity_dir and number % 500 == 0:
                update_index(source, vad_index_record(args), args.store)
            if args.raw_dir:
                Path(args.raw_dir).mkdir(parents=True, exist_ok=True)
                np.save(Path(args.raw_dir) / f"{isrc}.npy", raw)
            if number % 50 == 0 or number == len(todo):
                elapsed = time.time() - started
                print(
                    f"  {number}/{len(todo)} ({elapsed:.0f}s, {number / elapsed:.2f} songs/s, "
                    f"{audio_seconds / max(elapsed, 1e-6):.0f}x real time, {errors} errors)",
                    flush=True,
                )
    if not args.activity_dir:
        update_index(source, vad_index_record(args), args.store)
    print(
        json.dumps(
            {
                "method": "vad_head",
                "songs": len(todo) - errors,
                "errors": errors,
                "seconds": round(time.time() - started),
                "audio_hours": round(audio_seconds / 3600, 2),
            }
        ),
        flush=True,
    )


# ---------------------------------------------------------------------------------------------
# compare: stem_energy vs vad_head on songs that have both


def fragment_frames(row: dict) -> tuple[np.ndarray, np.ndarray]:
    """(sung, graded) per frame from the row's lead source fragments; edges ±1 frame ungraded."""
    frames = int(row["duration"] / 0.02) + 1
    sung = np.zeros(frames, dtype=bool)
    edge = np.zeros(frames, dtype=bool)
    for token in row["alignment_tokens"]:
        if token.get("ignore") or token.get("vocal_role") == "background":
            continue
        span = (token.get("source_spans") or [token])[0]
        first, stop = int(round(span["start"] / 0.02)), int(round(span["end"] / 0.02))
        sung[max(0, first) : min(frames, stop)] = True
        for frame in (first - 1, first, stop - 1, stop):
            if 0 <= frame < frames:
                edge[frame] = True
    return sung, ~edge


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """ROC AUC of ``positive`` over ``negative`` scores by rank sum; NaN when either is empty."""
    if not len(positive) or not len(negative):
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([positive, negative]))) + 1
    return float(
        (ranks[: len(positive)].sum() - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative))
    )


def compare(args) -> None:
    """Agreement of two activity sources on the songs that have both arrays."""
    from echora_kara.data import tail_trim

    rows = {}
    for split in ("train", "validation"):
        for source in (
            args.output / f"{split}.jsonl",
            Path(args.manifest_fallback or "/nonexistent") / f"{split}.jsonl",
        ):
            if source.exists():
                for line in source.open():
                    row = json.loads(line)
                    rows[row["id"]] = row
                break
    a_dir, b_dir = Path(args.a), Path(args.b)
    report = []
    for path in sorted(a_dir.glob("*.npz")):
        other = b_dir / path.name
        isrc = path.stem
        if not other.exists() or isrc not in rows:
            continue
        with np.load(path) as da, np.load(other) as db:
            a = da["activity"].astype(np.float32) / 255
            b = db["activity"].astype(np.float32) / 255
            method_a, method_b = str(da["method"]), str(db["method"])
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        sung, graded = fragment_frames(rows[isrc])
        sung, graded = sung[:n], graded[:n]
        row = rows[isrc]
        trims_a = tail_trim.plan_trims(row["alignment_tokens"], a, tail_trim.TrimConfig())
        trims_b = tail_trim.plan_trims(row["alignment_tokens"], b, tail_trim.TrimConfig())
        keys = set(trims_a) | set(trims_b)
        same = sum(1 for k in keys if abs(trims_a.get(k, 0.0) - trims_b.get(k, 0.0)) <= 0.06)
        report.append(
            {
                "id": isrc,
                "auc_a": auc(a[sung & graded], a[~sung & graded]),
                "auc_b": auc(b[sung & graded], b[~sung & graded]),
                "agree_at_threshold": float(((a >= TRIM_THRESHOLD) == (b >= TRIM_THRESHOLD))[graded].mean()),
                "corr": float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan"),
                "trims_a": len(trims_a),
                "trims_b": len(trims_b),
                "trim_decisions_within_60ms": same / max(1, len(keys)),
                "methods": (method_a, method_b),
            }
        )
    if not report:
        print("no songs with both arrays")
        return

    def mean(key):
        values = [r[key] for r in report if r[key] == r[key]]
        return round(float(np.mean(values)), 3) if values else None

    summary = {
        "songs": len(report),
        "methods": report[0]["methods"],
        "mean_auc_a": mean("auc_a"),
        "mean_auc_b": mean("auc_b"),
        "mean_frame_agreement_at_threshold": mean("agree_at_threshold"),
        "mean_correlation": mean("corr"),
        "mean_trim_decisions_within_60ms": mean("trim_decisions_within_60ms"),
        "trims_per_song": (mean("trims_a"), mean("trims_b")),
    }
    print(json.dumps(summary, indent=1))
    if args.report:
        Path(args.report).write_text(json.dumps({"summary": summary, "songs": report}, indent=1))


def main() -> None:
    """Command-line entry point for the ``stems``, ``vad``, ``migrate`` and ``compare`` commands."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("stems", help="stem_energy arrays for every vocal stem (CPU)")
    p.add_argument("--store", type=Path, required=True, help="activity store root")
    p.add_argument("--root", type=Path, required=True, help="data root with vocals/<id>.flac")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--ids", type=Path, help="only these recording ids (one per line)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(run=stems)
    p = sub.add_parser("vad", help="vad_head arrays on the mixes (needs torch)")
    p.add_argument("--output", type=Path, required=True, help="manifest directory whose songs are processed")
    p.add_argument("--store", type=Path, help="activity store root")
    p.add_argument("--root", type=Path, required=True, help="data root with audio/<id>.m4a")
    p.add_argument("--model", type=Path, required=True, help="w2v-BERT directory with aux_heads.safetensors")
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--limit", type=int)
    p.add_argument("--ids", type=Path)
    p.add_argument(
        "--manifest-fallback", help="manifest directory to take the song list from when --output has none yet"
    )
    p.add_argument("--only-without-stem", action="store_true", help="skip songs that have a vocal stem (default: all)")
    p.add_argument("--raw-dir", help="also keep the raw sigmoid per song (for the comparison)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--activity-dir", help="write here instead of the manifest's activity cache (comparison samples)")
    p.add_argument("--decode-workers", type=int, default=6)
    p.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="bfloat16",
        help="encoder precision under autocast",
    )
    p.add_argument("--batch", type=int, default=4, help="45 s chunks per forward pass")
    p.set_defaults(run=vad)
    p = sub.add_parser("migrate", help="copy an .npz stem directory into the store")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(run=migrate)
    p = sub.add_parser("compare", help="agreement of two activity caches on songs present in both")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--a", required=True, help="directory of .npz arrays (e.g. the stem_energy cache)")
    p.add_argument("--b", required=True, help="directory of .npz arrays (e.g. a vad_head sample)")
    p.add_argument("--manifest-fallback")
    p.add_argument("--report")
    p.set_defaults(run=compare)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
