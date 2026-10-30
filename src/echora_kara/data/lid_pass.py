#!/usr/bin/env python3
"""fastText lid.176 language identification of TTML lyric documents (CPU).

Reads ``<recording id>\\t<ttml path>`` lines on stdin and prints one JSON object per document:
``{"isrc", "doc_sha256", "lid_language", "lid_confidence", "lid_top3"}``. build_manifest.py calls
``identify_documents`` in-process.

  python -m echora_kara.data.lid_pass --model lid.176.ftz < list.tsv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from echora_kara.text.document_language import document_digest, lyric_text


def identify(model, ttml: str) -> dict:
    """fastText top-3 language prediction for a TTML document's lyric text; no language for under 20 characters."""
    text = lyric_text(ttml)
    if len(text) < 20:
        return {"lid_language": None, "lid_confidence": 0.0, "lid_top3": []}
    # The low-level call avoids fastText's numpy copy=False incompatibility with NumPy 2.
    predictions = model.f.predict(text + "\n", 3, 0.0, "strict")
    top = [(label.replace("__label__", ""), round(float(score), 4)) for score, label in predictions]
    return {"lid_language": top[0][0], "lid_confidence": top[0][1], "lid_top3": top}


def read_ttml(path: Path) -> str:
    """Document text exactly as stored (no newline translation: the digest is of these bytes)."""
    return Path(path).read_bytes().decode("utf-8")


def identify_documents(model_path, todo):
    """Yield one record per readable (recording id, TTML path) of ``todo``."""
    import fasttext

    model = fasttext.load_model(str(model_path))
    for isrc, path in todo:
        try:
            ttml = read_ttml(path)
        except (OSError, ValueError):
            continue
        yield {"isrc": isrc, "doc_sha256": document_digest(ttml), **identify(model, ttml)}


def main() -> None:
    """Print one language-id JSON record per ``<id>\t<ttml path>`` line read from stdin."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    todo = (line.rstrip("\n").split("\t", 1) for line in sys.stdin if line.strip())
    for record in identify_documents(args.model, todo):
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
