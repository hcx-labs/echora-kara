#!/usr/bin/env python3
"""Final physical-audio QC for timed-lyrics training manifests.

Input manifests hold rows with ``timings`` and an ``audio`` path relative to ``--audio-root``. This
pass probes the actual audio duration with ffprobe, excludes corrupt/mismatched files, and writes
curated manifests without copying audio. build_manifest.py reuses ``duration``.
"""
import argparse
import json
import math
import hashlib
from collections import Counter
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from echora_kara.data.integrity import TRAIN_AUDIO_STATUSES, TRAIN_LYRIC_STATUSES, VERIFIED_SPLITS, identity_problem

TOLERANCE_SECONDS = 0.05
# A different edit, live take or extended mix usually differs by seconds.
MAX_CATALOG_DURATION_DIFF_SECONDS = 2.0
# Verified recordings already match by ISRC; their files routinely differ
# from catalog length by a few seconds of padding. Only gross gaps reject.
MAX_VERIFIED_DURATION_DIFF_SECONDS = 30.0
WORKERS = 8


def duration(path: Path) -> float | None:
    try:
        output = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
            text=True, timeout=30,
        )
        value = float(output.strip())
        return value if math.isfinite(value) and value > 0 else None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return None


def probe_audio(path: Path) -> tuple[float | None, str | None]:
    actual = duration(path)
    if actual is None:
        return None, None
    try:
        with path.open('rb') as audio:
            digest = hashlib.file_digest(audio, 'sha256').hexdigest()
    except OSError:
        return None, None
    return actual, digest


def valid(row: dict, actual_duration: float | None) -> tuple[bool, str]:
    if actual_duration is None or not math.isfinite(actual_duration) or actual_duration <= 0:
        return False, "ffprobe_failed"
    catalog = row.get("duration")
    limit = MAX_VERIFIED_DURATION_DIFF_SECONDS if row.get("audio_match_status") == "exact_isrc" \
        else MAX_CATALOG_DURATION_DIFF_SECONDS
    if not isinstance(catalog, (int, float)) or not math.isfinite(catalog) or \
            abs(actual_duration - catalog) > limit:
        return False, "catalog_duration_mismatch"
    units = [unit for line in row["timings"] for unit in line]
    if any(not math.isfinite(unit[k]) for unit in units for k in ("start", "end")):
        return False, "nonfinite_timing"
    if any(unit["start"] < 0 or unit["end"] <= unit["start"] for unit in units):
        return False, "invalid_interval"
    usable = [unit for unit in units if unit.get("text") and not unit.get("ignore")]
    if not usable:
        return False, "no_supervision"
    if max(unit["end"] for unit in units) > actual_duration + TOLERANCE_SECONDS:
        return False, "lyrics_after_audio"
    if max(unit["end"] for unit in usable) - min(unit["start"] for unit in usable) < actual_duration * 0.25:
        return False, "coverage_under_25pct"
    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True, help="root of the rows' audio paths")
    args = parser.parse_args()
    source, output_dir, audio_root = args.source.resolve(), args.output.resolve(), args.audio_root.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for split in ("train", "validation", "test"):
        with (source / f"{split}.jsonl").open(encoding="utf-8") as manifest:
            for raw in manifest:
                if raw.strip():
                    all_rows.append((split, json.loads(raw)))
    seen = {field: {} for field in ("id", "audio", "artist_key")}
    for split, row in all_rows:
        for field, assignments in seen.items():
            value = row[field]
            if value in assignments and (field in {"id", "audio"} or assignments[value] != split):
                raise ValueError(f"Duplicate or cross-split {field}: {value}")
            assignments[value] = split
    results = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(probe_audio, audio_root / row["audio"]): (split, row) for split, row in all_rows}
        for future in as_completed(futures):
            split, row = futures[future]
            actual, digest = future.result()
            ok, reason = valid(row, actual)
            if ok and (problem := identity_problem(row, split)):
                ok, reason = False, problem
            if ok and row.get('audio_match_status') == 'exact_isrc' and row.get('audio_sha256') != digest:
                ok, reason = False, 'audio_provenance_mismatch'
            results[row["id"]] = (ok, reason, actual, digest)
    # Only kept rows compete; a rejected copy must not remove a good one.
    digests = Counter(result[3] for result in results.values() if result[0])
    kept = {"train": [], "validation": [], "test": []}
    rejected = {}
    for split, row in all_rows:
        ok, reason, actual, digest = results[row["id"]]
        if ok and digests[digest] > 1:
            ok, reason = False, 'duplicate_audio_content'
        if ok:
            row['audio_sha256'] = digest
            row["catalog_duration"] = row["duration"]
            row["duration"] = actual
            row["duration_source"] = "ffprobe_format"
            # Only sub-50ms container rounding is clipped. Larger overruns reject.
            for line in row["timings"]:
                for unit in line:
                    unit["end"] = min(unit["end"], actual)
            if any(unit["end"] <= unit["start"] for line in row["timings"] for unit in line):
                rejected["interval_outside_audio"] = rejected.get("interval_outside_audio", 0) + 1
                continue
            row["lyric_end"] = max(unit["end"] for line in row["timings"] for unit in line)
            kept[split].append(row)
        else:
            rejected[reason] = rejected.get(reason, 0) + 1
    for split, rows in kept.items():
        with (output_dir / f"{split}.jsonl").open("w") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source": str(source),
        "splits": {name: len(rows) for name, rows in kept.items()},
        "total": sum(map(len, kept.values())),
        "rejected": rejected,
        "audio_duration_tolerance_seconds": TOLERANCE_SECONDS,
        "max_catalog_duration_diff_seconds": {"unverified": MAX_CATALOG_DURATION_DIFF_SECONDS,
                                              "exact_isrc": MAX_VERIFIED_DURATION_DIFF_SECONDS},
        "identity_policy": {"exact_isrc_required": list(VERIFIED_SPLITS),
                            "train_audio_statuses": sorted(TRAIN_AUDIO_STATUSES),
                            "train_lyric_statuses": sorted(TRAIN_LYRIC_STATUSES)},
        "duplicate_detection": "encoded file SHA-256; different encodings require fingerprint/manual QC",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
