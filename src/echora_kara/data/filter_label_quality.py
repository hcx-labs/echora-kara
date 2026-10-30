#!/usr/bin/env python3
"""Drop songs whose label-quality score is below a threshold from a compact karaoke manifest.

Keeps the existing artist-disjoint split (no re-split), filters train and validation, copies test
unchanged (the held-out test set is never curated). Songs without a score are kept.
A threshold of 0.05 removes about 5 % of the songs, mostly ones whose timings belong to another edit
of the recording.

usage: python -m echora_kara.data.filter_label_quality --source DIR/compact --output DIR/compact_q005 --min 0.05
"""

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    """Write the filtered manifest to ``--output``: train/validation without low-score songs, ``dropped_<split>.txt``
    id lists, test and vocab copied verbatim, and a ``summary.json`` of the counts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min", type=float, required=True, help="minimum label_quality_score kept")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {args.output}")
    args.output.mkdir(parents=True)

    summary = {"source": str(args.source), "min_label_quality": args.min, "splits": {}, "dropped": {}}
    for split in ("train", "validation"):
        kept = dropped = 0
        dropped_ids = []
        with (args.source / f"{split}.jsonl").open() as source, (args.output / f"{split}.jsonl").open("w") as sink:
            for line in source:
                score = json.loads(line).get("label_quality_score")
                if score is not None and score < args.min:
                    dropped += 1
                    dropped_ids.append(json.loads(line)["id"])
                    continue
                sink.write(line)
                kept += 1
        summary["splits"][split] = kept
        summary["dropped"][split] = dropped
        (args.output / f"dropped_{split}.txt").write_text("\n".join(dropped_ids) + "\n")
    for name in ("test.jsonl", "vocab.json"):
        shutil.copy(args.source / name, args.output / name)
    summary["splits"]["test"] = sum(1 for _ in (args.output / "test.jsonl").open())
    summary["total"] = sum(summary["splits"].values())  # dataset validation checks it
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
