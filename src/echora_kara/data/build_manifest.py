#!/usr/bin/env python3
"""Build the karaoke forced-alignment manifest on full mixes, incrementally.

  python -m echora_kara.data.build_manifest --index recordings.jsonl --audio-root /data/audio \\
      --output /data/manifests/karaoke_v1 --lid-model lid.176.ftz

Input layout. ``--index`` is a JSONL file with one object per recording:

  id                  recording identifier (e.g. an ISRC); the manifest row id
  title, artist       credits; ``artist`` is split on ", ", " & ", " x ", " + " into credited artists
  duration            expected duration in seconds; the probed audio must agree with it
  lyrics              TTML path, relative to the index file (default ``lyrics/<id>.ttml``)
  audio               audio path, relative to ``--audio-root`` (default ``audio/<id>.m4a``); rows store
                      it, so the trainer runs with ``--root <audio-root>``
  audio_match_status  optional: ``exact_isrc`` (the file is verified to be this recording),
                      ``isrc_mismatch`` (a known wrong recording: excluded), else ``unverified_legacy``
  audio_sha256, audio_bytes
                      optional: hash and size of the verified file; ``exact_isrc`` is downgraded when the
                      file's hash differs, and a matching size skips re-hashing
  lyric_match_status  optional: ``exact_isrc`` when the lyrics were matched by this id
                      (default ``unverified_provider_lookup``)

Lyrics are word-timed TTML: one ``<p>`` per line, timed ``<span begin end>`` words or syllables,
``role="x-bg"`` for background vocals, ``xml:lang`` and an optional Jyutping ``<transliteration>``
for the language. A root ``timing`` attribute marks absolute media times at every level (see
word_timed.py). Elements are matched by local name, in any namespace.

Labels: the shared normalizer, 6-character units, background rules, with the per-document
language hint of ``document_language`` (``ja`` for Japanese, ``yue`` for Jyutping Cantonese,
``auto`` otherwise). Every per-song result (labels, audio probe, label-quality score, language id)
is cached under ``--cache-dir`` (default ``<output>/cache``) keyed by content hashes and tool
versions, so re-running after more audio arrives only processes the new songs; the artist→split
assignment is persisted in the output directory so songs keep their split across re-runs.

Splits: 95 % train / 5 % validation over connected components of credited artists. Validation
admits only exact-ISRC audio and lyric identity; songs failing that in a validation component are
dropped, never moved. ``--held-out-test`` names a frozen prepared test manifest: its songs *and*
artists are excluded from train/validation, and ``test.jsonl`` holds its songs whose audio exists,
labels verbatim, audio re-probed.

Label quality (label_quality.py) is computed on the mix for every song and stored as
``label_quality`` per row plus a distribution in summary.json. Nothing is filtered by it unless
``--min-label-quality`` is given.

Label round v2 (opt-in; the defaults build v1 labels):

  python -m echora_kara.data.build_manifest ... --label-rules v3 --label-policy v2 \\
      --reference-labels PREVIOUS_CACHE/labels --split-assignments-from PREVIOUS/split_assignments.json \\
      --activity-dir ACTIVITY_STORE --offset-gate drop --near-duplicates drop --workers 12

- ``--label-rules``: the shared normalizer's rules (label_rules.py). From v2 on, the hint comes from
  the *final* language (fastText override included), not from ``document_language`` alone.
- ``--label-policy v2``: label_policy.py (phrase-level background, parenthesised echoes, lead-lead
  overlap trimming, repeat metadata) and the tail trim from vocal_activity.py arrays (stem energy →
  trimmed frames blank). Rows get ``label_schema_version`` 7.
- ``--reference-labels``: the label-quality score and the whole-song offset search are scored on
  these (v1) timings, which the 0.05 threshold was calibrated on; songs without a reference are not
  built.
- ``--offset-gate drop``: drop songs whose whole-song offset is confirmed (label_quality.offset_search),
  and songs titled live/remaster/edit whose offset is flagged; ``record`` (default) only stores it.
- ``--near-duplicates drop``: one recording per (title, artist, duration ± 1.5 s) group.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path

from echora_kara.data.curate import duration as probe_duration
from echora_kara.data.integrity import artist_key, check_row, identity_problem, new_assignments
from echora_kara.data.lid_pass import read_ttml
from echora_kara.text.document_language import document_digest, document_language, normalizer_language

BUILDER_VERSION = "karaoke_manifest_v1"
BUILDER_VERSION_V2 = "karaoke_manifest_v2"
TITLE_FLAG = re.compile(r"\b(live|remaster(ed)?|edit)\b", re.I)
NEAR_DUPLICATE_SECONDS = 1.5
VOCAB = {"|": 0, **{chr(ord("a") + index): index + 1 for index in range(26)}, "'": 27, "[UNK]": 28, "[PAD]": 29}
MAX_LABEL_CHARS = 6
MIN_UNITS = 12
MIN_COVERAGE = 0.25
TOLERANCE_SECONDS = 0.05
MAX_CATALOG_DURATION_DIFF = {"exact_isrc": 30.0, "unverified_legacy": 2.0}
VALIDATION_SHARE = 0.05
ARTIST_SEPARATORS = re.compile(r", | & | x | X | \+ ")
UNVERIFIED_LYRIC = "unverified_provider_lookup"


# ---------------------------------------------------------------------------------------------
# Caches


def read_jsonl_cache(path: Path) -> dict:
    """Records of an append-only JSONL cache by their ``key`` field (later lines win); {} when the file is missing."""
    values = {}
    if path.exists():
        with path.open() as source:
            for line in source:
                if line.strip():
                    record = json.loads(line)
                    values[record["key"]] = record
    return values


# ---------------------------------------------------------------------------------------------
# Recordings and audio identity


def load_index(path: Path) -> tuple[dict[str, dict], int]:
    """Recordings of the JSONL index by id (lyrics path resolved), and the number of duplicate ids."""
    recordings, duplicates = {}, 0
    with path.open() as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            isrc = str(record["id"])
            if isrc in recordings:
                duplicates += 1
                continue
            record["lyrics"] = path.parent / (record.get("lyrics") or f"lyrics/{isrc}.ttml")
            record["audio"] = record.get("audio") or f"audio/{isrc}.m4a"
            recordings[isrc] = record
    return recordings, duplicates


def audio_path(isrc: str, recordings: dict) -> str:
    """Audio path of a recording relative to ``--audio-root``: the index entry's, else ``audio/<id>.m4a``."""
    return (recordings.get(isrc) or {}).get("audio") or f"audio/{isrc}.m4a"


def audio_identity(isrc: str, entry: dict) -> dict:
    """Identity record of one recording's audio file (how it was matched, which hash)."""
    status = entry.get("audio_match_status") or "unverified_legacy"
    identity = {
        "audio_match_status": status,
        "index_sha256": entry.get("audio_sha256"),
        "index_bytes": entry.get("audio_bytes"),
    }
    if status in ("exact_isrc", "isrc_mismatch"):
        identity.update(requested_isrc=isrc, matched_isrc=isrc)
    return identity


def probe_audio(path: Path, trusted_sha: str | None, trusted_bytes: int | None) -> dict:
    """ffprobe duration (seconds) and SHA-256 of an audio file, plus its size and mtime for cache invalidation.

    The index's ``trusted_sha`` is reused without hashing when ``trusted_bytes`` matches the file size. An unreadable
    file yields ``{"error": "ffprobe_failed", ...}`` instead of duration and hash."""
    stat = path.stat()
    actual = probe_duration(path)
    if actual is None:
        return {"error": "ffprobe_failed", "size": stat.st_size, "mtime": stat.st_mtime}
    if trusted_sha and trusted_bytes == stat.st_size:
        digest = trusted_sha
    else:
        with path.open("rb") as audio:
            digest = hashlib.file_digest(audio, "sha256").hexdigest()
    return {"duration": actual, "sha256": digest, "size": stat.st_size, "mtime": stat.st_mtime}


# ---------------------------------------------------------------------------------------------
# Labels (multiprocessing workers)

_WORKER = {}


def _init_worker(rules: str = "v1", policy: str = "v1"):
    from echora_kara.data import word_timed as builder
    from echora_kara.text.normalization import normalizer_version, training_language

    _WORKER.update(
        builder=builder,
        version=normalizer_version(rules),
        training_language=training_language,
        rules=rules,
        policy=policy,
    )


def labels_key(doc_sha: str, rules: str = "v1", policy: str = "v1", hint: str | None = None) -> str:
    """Labels-cache key of a document: its digest, normalizer version and builder version.

    v1 rules with policy v1 keep the historical three-part key; any v2 build adds the policy version and the
    normalizer ``hint`` language, so a changed final language invalidates the cached labels."""
    from echora_kara.text.normalization import normalizer_version

    if rules == "v1" and policy == "v1":
        return f"{doc_sha}:{normalizer_version('v1')}:{BUILDER_VERSION}"
    from echora_kara.data import label_policy as label_policy_v2

    return f"{doc_sha}:{normalizer_version(rules)}:{BUILDER_VERSION_V2}:{label_policy_v2.POLICY_VERSION if policy == 'v2' else 'policy_v1'}:{hint}"


def normalize_document(task: tuple[str, str]) -> tuple[str, dict]:
    """Worker: (id, TTML path[, final language]) → labels record written to the per-song cache file."""
    isrc, path, final_language = task if len(task) == 3 else (*task, None)
    builder = _WORKER["builder"]
    rules, policy = _WORKER.get("rules", "v1"), _WORKER.get("policy", "v1")
    ttml = read_ttml(Path(path))
    doc_sha = document_digest(ttml)
    language = document_language(ttml)
    builder.TTML_AUDIT.clear()
    if rules == "v1" and policy == "v1":
        hint = _WORKER["training_language"](normalizer_language(language["language"]))
        lines = builder.timing_lines(ttml, hint)
        key = f"{doc_sha}:{_WORKER['version']}:{BUILDER_VERSION}"
        extra = {}
    else:
        # v2: the hint follows the final language (fastText override included).
        hint = _WORKER["training_language"](normalizer_language(final_language or language["language"], rules), rules)
        lines = builder.timing_lines(ttml, hint, rules, policy)
        key = labels_key(doc_sha, rules, policy, hint)
        extra = {
            "label_rules": rules,
            "label_policy": policy,
            "hint_language": final_language or language["language"],
            "policy_stats": {
                k: v
                for k, v in builder.TTML_AUDIT.items()
                if k.startswith(("paren", "background_phr", "background_words", "lead_overlap"))
            },
        }
        if policy == "v2":
            from echora_kara.data import label_policy as label_policy_v2

            extra["line_repeat"] = label_policy_v2.repeat_metadata(lines)
    units = [unit for line in lines for unit in line]
    masks = Counter(unit["normalization_error"].split(":", 1)[0] for unit in units if unit.get("ignore"))
    usable = [unit for unit in units if unit.get("text") and not unit.get("ignore")]
    record = {
        "isrc": isrc,
        "key": key,
        "doc_sha256": doc_sha,
        **extra,
        "timing_granularity": builder.timing_granularity(ttml),
        "language": language["language"],
        "language_source": language["source"],
        "xml_lang": language["xml_lang"],
        "transliteration": language["transliteration"],
        "normalizer_language": hint,
        "normalizer": f"echora.fa_kara.process_haruhi_line:{hint}+{_WORKER['version']}",
        "timings": lines,
        "timing_units": len(units),
        "usable_units": len(usable),
        "mask_reasons": dict(masks),
        "ttml_audit": dict(builder.TTML_AUDIT),
        "lyric_start": round(min((u["start"] for u in units), default=0.0), 3),
        "lyric_end": round(max((u["end"] for u in units), default=0.0), 3),
        "usable_start": round(min((u["start"] for u in usable), default=0.0), 3),
        "usable_end": round(max((u["end"] for u in usable), default=0.0), 3),
        "nonfinite": any(not math.isfinite(u[k]) for u in units for k in ("start", "end")),
    }
    return isrc, record


def _v2_head(task: tuple[str, str]) -> tuple[str, dict]:
    """Worker: document digest and document_language head of one document (v2 hint)."""
    isrc, path = task
    ttml = read_ttml(Path(path))
    language = document_language(ttml)
    head = {"language": language["language"], "language_source": language["source"], "xml_lang": language["xml_lang"]}
    return isrc, {"doc_sha256": document_digest(ttml), **head}


def v2_label_tasks(documents: dict, caches: Path, args) -> list[tuple[str, str, str]]:
    """v2 label tasks (id, path, final language) for documents whose cached labels are stale.

    The final language is the one the row will carry: document_language, then the cached fastText
    decision (resolve_language). v1 computes the hint from document_language alone."""
    from echora_kara.text.normalization import training_language

    heads = {}
    with Pool(args.workers) as pool:
        for isrc, head in pool.imap_unordered(
            _v2_head, [(i, str(d["path"])) for i, d in documents.items()], chunksize=32
        ):
            heads[isrc] = head
    lid_cache_path = caches / "lid.jsonl"
    lid = {record["key"]: record for record in read_jsonl_cache(lid_cache_path).values()}
    missing = [(isrc, documents[isrc]["path"]) for isrc, head in heads.items() if head["doc_sha256"] not in lid]
    if missing and args.lid_model:
        print(f"language id for {len(missing)} documents before normalisation ({len(lid)} cached)", flush=True)
        with lid_cache_path.open("a") as sink:
            for record in run_lid(missing, args.lid_model):
                record = {"key": record["doc_sha256"], **record}
                lid[record["key"]] = record
                sink.write(json.dumps(record) + "\n")
    todo, hints = [], Counter()
    for isrc, doc in documents.items():
        head = heads[isrc]
        final = resolve_language(head, lid.get(head["doc_sha256"]))["language"]
        hint = training_language(normalizer_language(final, args.label_rules), args.label_rules)
        hints[hint] += 1
        cached = caches / "labels" / f"{isrc}.json"
        doc["labels_path"] = cached
        if cached.exists():
            try:
                with cached.open() as source:
                    prefix = source.read(4096)
                key_at = prefix.find('"key": "')
                cached_key = prefix[key_at + 8 : prefix.index('"', key_at + 8)] if key_at >= 0 else None
                if cached_key == labels_key(head["doc_sha256"], args.label_rules, args.label_policy, hint):
                    continue
            except (ValueError, OSError):
                pass
        todo.append((isrc, str(doc["path"]), final))
    print(f"v2 hints: {dict(hints.most_common())}", flush=True)
    return todo


def _reference_quality_key(task: tuple[str, str, str]) -> tuple[str, str | None]:
    """Worker: the quality-cache key of one song scored on its reference timings."""
    isrc, reference, sha = task
    from echora_kara.data import label_quality

    try:
        timings = json.loads(Path(reference).read_text())["timings"]
    except (OSError, ValueError, KeyError):
        return isrc, None
    return isrc, f"{sha}:{label_quality.timings_digest(timings)}:{label_quality.VERSION}"


def _quality_worker(task: tuple[str, str, str]) -> tuple[str, dict]:
    isrc, audio_path, labels_path = task
    from echora_kara.data import label_quality

    record = json.loads(Path(labels_path).read_text())
    try:
        score = label_quality.score_file(Path(audio_path), record["timings"])
    except Exception as error:  # noqa: BLE001 - recorded, never fatal for the build
        score = {
            "method": label_quality.VERSION,
            "error": f"{type(error).__name__}: {error}",
            "score": None,
            "lines": [],
        }
    return isrc, score


# ---------------------------------------------------------------------------------------------
# Language identification


def run_lid(todo: list[tuple[str, Path]], model: Path):
    """Yield lid_pass records (in-process); nothing when fastText is not installed."""
    try:
        import fasttext  # noqa: F401
    except ImportError:
        print("  LID skipped: fastText is not installed", file=sys.stderr, flush=True)
        return
    from echora_kara.data.lid_pass import identify_documents

    yield from identify_documents(model, todo)


def resolve_language(head: dict, lid: dict | None) -> dict:
    """xml:lang / script first; fastText fills ``und`` and overrides an implausible Latin-script tag."""
    language, source = head["language"], head["language_source"]
    detected, confidence = (lid or {}).get("lid_language"), float((lid or {}).get("lid_confidence") or 0.0)
    if source in ("script", "transliteration"):
        pass
    elif language == "und" and detected and confidence >= 0.5:
        language, source = detected, "fasttext"
    elif detected and detected != language and confidence >= 0.8:
        language, source = detected, "fasttext_override"
    return {
        "language": language,
        "language_source": source,
        "xml_lang": head["xml_lang"],
        "lid_language": detected,
        "lid_confidence": confidence,
    }


# ---------------------------------------------------------------------------------------------
# Artists and splits


def drop_near_duplicates(accepted: list[str], meta: dict, quality: dict) -> tuple[list[str], list[str]]:
    """One recording per (title, artist) whose durations agree within 1.5 s (remasters and
    re-releases under a second recording id). Kept: exact ISRC audio and lyrics first, then the
    higher label-quality score, then the smaller id."""
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for isrc in accepted:
        track = meta[isrc]["track"]
        groups[(artist_key(str(track.get("title") or "")), artist_key(str(track.get("artist") or "")))].append(isrc)
    dropped: set[str] = set()

    def rank(isrc):
        exact = meta[isrc]["identity"].get("audio_match_status") == "exact_isrc"
        lyric = meta[isrc]["lyric_match_status"] == "exact_isrc"
        return (not exact, not lyric, -((quality.get(isrc) or {}).get("score") or 0.0), isrc)

    for (title, _), isrcs in groups.items():
        if len(isrcs) < 2 or not title:
            continue
        isrcs = sorted(isrcs, key=lambda i: meta[i]["probe"]["duration"])
        cluster = [isrcs[0]]
        for isrc in isrcs[1:] + [None]:
            if (
                isrc is not None
                and meta[isrc]["probe"]["duration"] - meta[cluster[-1]]["probe"]["duration"] < NEAR_DUPLICATE_SECONDS
            ):
                cluster.append(isrc)
                continue
            if len(cluster) > 1:
                dropped.update(sorted(cluster, key=rank)[1:])
            cluster = [isrc] if isrc is not None else []
    return [isrc for isrc in accepted if isrc not in dropped], sorted(dropped)


def credited_artists(artist: str) -> list[str]:
    """Normalised credited artist keys of an artist string, split on ``ARTIST_SEPARATORS``, deduplicated in order."""
    names = [artist_key(name) for name in ARTIST_SEPARATORS.split(artist or "")]
    return [name for name in dict.fromkeys(names) if name]


class Components:
    """Union-find over artist keys; each component's root is its lexicographically smallest member."""

    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        """Root of ``item``'s component (registering it as a singleton if unseen), with path halving."""
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, items: list[str]) -> None:
        """Merge the components of all ``items`` into one."""
        for other in items[1:]:
            a, b = self.find(items[0]), self.find(other)
            if a != b:
                self.parent[max(a, b)] = min(a, b)


def split_hash(component: str, seed: str, share: float = VALIDATION_SHARE) -> str:
    """Deterministic split of an artist component: "validation" for ``share`` of the sha256 buckets of
    ``seed:component``, else "train"."""
    value = int(hashlib.sha256(f"{seed}:{component}".encode()).hexdigest()[:12], 16) % 10000
    return "validation" if value < share * 10000 else "train"


# ---------------------------------------------------------------------------------------------
# Rows


def slim_unit(unit: dict) -> dict:
    """Drop debug-only fields; everything the trainer, check_row and evaluators read stays."""
    return {k: v for k, v in unit.items() if k not in ("normalizer_items", "display_text", "normalizer")}


def finite(value):
    """JSON-safe copy: non-finite floats (an offset_ratio of inf when the zero-lag correlation is not
    positive) become None."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite(v) for v in value]
    return value


def prepared_row(
    isrc: str,
    meta: dict,
    labels: dict,
    probe: dict,
    identity: dict,
    split: str,
    artists: list[str],
    component: str,
    quality: dict | None,
    v2: dict | None = None,
) -> dict:
    """Full manifest row (label schema 6, or 7 when ``v2`` holds the finalised policy-v2 labels) for one song.

    Timings are clipped to the probed duration and flattened into ``alignment_tokens`` with vocabulary ids and their
    line index; identity, language, normalizer and label-quality fields are copied from the inputs."""
    timings = [[slim_unit(u) for u in line] for line in (v2["timings"] if v2 else labels["timings"])]
    for line in timings:
        for unit in line:
            unit["end"] = min(unit["end"], probe["duration"])
    flat, transcripts = [], []
    for line_index, line in enumerate(timings):
        transcripts.append("".join(unit["text"] for unit in line))
        for unit in line:
            flat.append({**unit, "line": line_index, "ids": [VOCAB[c] for c in unit["text"]]})
    track = meta["track"]
    return {
        "id": isrc,
        "audio": track["audio"],
        "audio_match_status": identity["audio_match_status"],
        **{k: identity[k] for k in ("requested_isrc", "matched_isrc") if identity.get(k)},
        "audio_sha256": probe["sha256"],
        "duration": probe["duration"],
        "duration_source": "ffprobe_format",
        "catalog_duration": track.get("duration"),
        "title": track.get("title"),
        "artist": track.get("artist"),
        "artists": artists,
        "artist_key": component,
        "split": split,
        "lyric_match_status": meta["lyric_match_status"],
        "source_timing_granularity": labels.get("timing_granularity") or "fragments",
        "measured_syllable_boundaries": False,
        "language": labels["language"],
        "language_source": labels["language_source"],
        "xml_lang": labels["xml_lang"],
        "lid_language": labels.get("lid_language"),
        "lid_confidence": labels.get("lid_confidence"),
        "normalizer_language": labels["normalizer_language"],
        "normalizer": labels["normalizer"],
        "timings": timings,
        "timing_units": labels["timing_units"],
        "lyric_start": labels["lyric_start"],
        "lyric_end": round(max(u["end"] for line in timings for u in line), 3),
        "alignment_tokens": flat,
        "line_transcripts": transcripts,
        "tokenizer": "NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn",
        "pad_token_id": VOCAB["[PAD]"],
        "label_schema_version": 6,
        "supervision": "source-timed pronunciation characters; within-fragment boundaries are weak estimates",
        "label_quality": finite(quality),
        **(v2_row_fields(v2, probe["duration"]) if v2 else {}),
    }


def v2_row_fields(v2: dict, duration: float) -> dict:
    """Schema-7 fields: label versions, tail-trim provenance, repeat metadata, background-head targets."""
    background = []
    for unit in v2["background"]:
        if unit["start"] >= duration:
            continue
        token = {
            k: unit[k]
            for k in (
                "start",
                "end",
                "text",
                "line",
                "lead_overlap",
                "phrase",
                "background_source",
                "background_category",
                "tail_trim",
                "provider_end",
                "ignore",
            )
            if k in unit
        }
        token["end"] = min(token["end"], duration)
        token["ids"] = [] if unit.get("ignore") else [VOCAB[c] for c in unit["text"]]
        token["source_spans"] = [
            {"start": span["start"], "end": min(span["end"], duration)} for span in unit.get("source_spans") or []
        ]
        if token["end"] > token["start"]:
            background.append(token)
    return {
        "label_schema_version": 7,
        "label_rules": v2["rules"],
        "label_policy": v2["policy"],
        "normalizer_language": v2["hint"],
        "label_policy_stats": v2["activity"],
        "line_repeat": v2["line_repeat"],
        "background_tokens": background,
    }


def held_out_rows(path: Path | None, audio_root: Path, probes: dict, recordings: dict) -> tuple[list[dict], Counter]:
    """The frozen held-out test songs on their mix audio, labels verbatim."""
    rows, skipped = [], Counter()
    if path is None:
        return rows, skipped
    with path.open() as source:
        for line in source:
            row = json.loads(line)
            relative = audio_path(row["id"], recordings)
            audio = audio_root / relative
            probe = probes.get(str(audio))
            if not audio.exists() or not probe or probe.get("error"):
                skipped["test_audio_missing"] += 1
                continue
            if abs(probe["duration"] - row["duration"]) > 0.5:
                skipped["test_duration_differs"] += 1
                continue
            # The held-out rows keep their artist strings; namespacing the key keeps check_row's
            # cross-split test from colliding with a train component that happens to be named
            # after a test artist's collaborator graph (its direct songs are excluded anyway).
            row.update(
                audio=relative,
                audio_sha256=probe["sha256"],
                duration=probe["duration"],
                mix_audio=None,
                audio_source="mix",
                held_out_from=path.parent.name,
                artists=credited_artists(row.get("artist", "")),
                label_quality=None,
                artist_key=f"held_out:{row['artist_key']}",
            )
            row.pop("mix_audio")
            if any(u["end"] > probe["duration"] + 1e-6 for u in row["alignment_tokens"]):
                skipped["test_lyrics_after_audio"] += 1
                continue
            rows.append(row)
    return rows, skipped


def compact_row(row: dict) -> dict:
    """Only what the trainer (train/targets, streams, overlap) and integrity.check_row read: identity,
    duration, per-token start/end/ids/ignore flags, the measured source fragment (region, edge-head
    starts) and the timings signature."""
    keep_token = (
        "start",
        "end",
        "text",
        "ids",
        "ignore",
        "soft_ignore",
        "vocal_role",
        "line",
        "timing_method",
        "overlap",
    )
    tokens = []
    for token in row["alignment_tokens"]:
        compact = {k: token[k] for k in keep_token if k in token}
        if token.get("source_spans"):
            compact["source_spans"] = [{"start": span["start"], "end": span["end"]} for span in token["source_spans"]]
        tokens.append(compact)
    timings = [
        [{k: unit[k] for k in ("start", "end", "text", "ignore") if k in unit} for unit in line]
        for line in row["timings"]
    ]
    quality = row.get("label_quality") or {}
    return {
        **{
            k: row[k]
            for k in (
                "id",
                "audio",
                "split",
                "artist_key",
                "audio_sha256",
                "audio_match_status",
                "lyric_match_status",
                "duration",
                "duration_source",
                "language",
                "label_schema_version",
                "pad_token_id",
                "line_transcripts",
            )
            if k in row
        },
        **{k: row[k] for k in ("requested_isrc", "matched_isrc") if row.get(k)},
        "timings": timings,
        "alignment_tokens": tokens,
        "label_quality_score": quality.get("score"),
        **({k: row[k] for k in ("label_rules", "label_policy", "line_repeat") if k in row}),
        **(
            {
                "background_tokens": [
                    {
                        k: t[k]
                        for k in (
                            "start",
                            "end",
                            "text",
                            "ids",
                            "ignore",
                            "line",
                            "lead_overlap",
                            "background_category",
                            "source_spans",
                        )
                        if k in t
                    }
                    for t in row["background_tokens"]
                ]
            }
            if "background_tokens" in row
            else {}
        ),
        **(
            {"offset_s": quality.get("offset_s"), "offset_flagged": quality.get("offset_flagged")}
            if "offset_s" in quality
            else {}
        ),
    }


# ---------------------------------------------------------------------------------------------
# Main build


def load_split_history(directory: Path) -> dict[str, str]:
    """Read actual song membership, not component names or JSON formatting."""
    history = {}
    for split in ("train", "validation", "test"):
        with (directory / f"{split}.jsonl").open() as source:
            for line in source:
                row = json.loads(line)
                isrc = row["id"]
                if row.get("split", split) != split:
                    raise ValueError(f"Historical split mismatch: {isrc} in {split}")
                if isrc in history and history[isrc] != split:
                    raise ValueError(f"Mixed historical membership: {isrc}")
                history[isrc] = split
    return history


def historical_component_split(
    component: str, songs: list[str], history: dict[str, str], assigned: str | None = None
) -> str:
    """Reject merged histories; new admissions are train-only, never new warm-start holdouts."""
    splits = {history[song] for song in songs if song in history}
    if len(splits) > 1 or "test" in splits:
        raise ValueError(f"Mixed or held-out history in component {component}: {sorted(splits)}")
    split = next(iter(splits), "train")
    if split == "validation" and any(song not in history for song in songs):
        raise ValueError(f"New admissions mixed with historical validation: {component}")
    if assigned is not None and assigned != split:
        raise ValueError(f"Historical split conflicts with assignment for {component}: {split} != {assigned}")
    return split


def quarantine_validation_admissions(by_component, history, assignments):
    """Keep a warm-start holdout fixed without training on its new artist peers.

    Unknown songs in a validation component cannot safely be assigned to train.
    Exclude them instead; genuine mixed historical membership still fails below.
    """
    kept, quarantined = {}, []
    for component, songs in by_component.items():
        validation = assignments.get(component) == "validation" or any(
            history.get(song) == "validation" for song in songs
        )
        excluded = {song for song in songs if validation and song not in history}
        quarantined.extend(excluded)
        remaining = [song for song in songs if song not in excluded]
        if remaining:
            kept[component] = remaining
    return kept, sorted(quarantined)


def check_heldout_history(songs, history: dict[str, str], split: str) -> None:
    """Raise ValueError when any of ``songs`` was in a different split than ``split`` in the historical build."""
    conflicts = sorted(song for song in songs if song in history and history[song] != split)
    if conflicts:
        raise ValueError(f"Historical songs moved to {split}: {conflicts[:10]}")


def load_activity_baseline(directory: Path) -> dict[str, str]:
    """Full prior rows are authoritative; compact rows omit activity provenance.

    Cross-check source totals against the summary to reject incomplete inventories.
    Read before any output is opened, including when rebuilding in place.
    """
    summary = json.loads((directory / "summary.json").read_text())
    if summary.get("label_policy") != "v2":
        raise ValueError("Activity baseline must use label policy v2")
    baseline, seen, sources = {}, set(), Counter()
    for split in ("train", "validation"):
        count = 0
        with (directory / f"{split}.jsonl").open() as source:
            for line in source:
                row = json.loads(line)
                song = row["id"]
                if song in seen or row.get("split") != split:
                    raise ValueError(f"Invalid activity baseline membership: {song}")
                seen.add(song)
                stats = row.get("label_policy_stats") or {}
                if "activity" not in stats:
                    raise ValueError(f"Missing full-row activity provenance: {song}")
                method = stats["activity"]
                if method is not None and (not isinstance(method, str) or not method or method == "None"):
                    raise ValueError(f"Invalid baseline activity source: {song}")
                sources[str(method)] += 1
                if method is not None:
                    baseline[song] = method
                count += 1
        if count != summary["splits"][split]:
            raise ValueError(f"Activity baseline row count mismatch: {split}")
    if dict(sources) != summary["tail_trim"]["songs_by_activity_source"]:
        raise ValueError("Activity baseline source counts disagree with summary")
    return baseline


def check_baseline_activity(song: str, activity, baseline: dict) -> None:
    """Raise ValueError when a song of the activity baseline lost its activity array or now loads it from another
    source."""
    if song in baseline and (activity is None or activity["method"] != baseline[song]):
        raise ValueError(f"Baseline activity lost or source changed: {song} (expected {baseline[song]})")


def activity_coverage(songs, directory: Path, loader, required: bool = False, baseline: dict | None = None) -> dict:
    """Use the real loader so shared-store precedence and .npz caches agree with finalization."""
    baseline = baseline or {}
    songs = set(songs)
    missing, sources = [], Counter()
    for song in sorted(songs):
        activity = loader(directory, song)
        check_baseline_activity(song, activity, baseline)
        if activity is None:
            missing.append(song)
        else:
            sources[activity["method"]] += 1
    report = {
        "directory": str(directory),
        "required": required,
        "baseline_covered": len(baseline),
        "baseline_retained": len(songs & baseline.keys()),
        "baseline_not_emitted": len(baseline.keys() - songs),
        "available": sum(sources.values()),
        "missing": len(missing),
        "missing_sample": missing[:20],
        "sources": dict(sources),
    }
    if required and missing:
        raise ValueError(f"Required activity missing: {json.dumps(report)}")
    return report


def main() -> None:
    """Build or incrementally update the manifest: full rows, ``compact/`` trainer rows, ``vocab.json``,
    ``split_assignments.json`` and ``summary.json`` under ``--output``.

    Per-song labels, audio probes, language ids and label-quality scores are cached under ``--cache-dir`` and reused
    when their content keys match; see the module docstring for the inputs and the v2 options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, required=True, help="recordings JSONL (see the input layout above)")
    parser.add_argument("--audio-root", type=Path, required=True, help="root of the index's audio paths")
    parser.add_argument("--output", type=Path, required=True, help="manifest directory (created; re-runs update it)")
    parser.add_argument("--cache-dir", type=Path, help="per-song caches (default: <output>/cache)")
    parser.add_argument(
        "--held-out-test",
        type=Path,
        help="frozen prepared test.jsonl: its songs and artists are excluded from train/validation",
    )
    parser.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 8) // 2))
    parser.add_argument("--limit", type=int, help="only the first N documents with audio (smoke runs)")
    parser.add_argument("--languages", nargs="*", help="restrict to these document languages")
    parser.add_argument("--no-label-quality", action="store_true", help="skip the label-quality score (faster)")
    parser.add_argument(
        "--min-label-quality",
        type=float,
        help="opt-in: drop songs whose label_quality.score is below this (default keeps all)",
    )
    parser.add_argument("--seed", default="karaoke", help="split hash seed prefix (keep fixed across re-runs)")
    parser.add_argument("--validation-share", type=float, default=VALIDATION_SHARE)
    parser.add_argument(
        "--search-seeds",
        type=int,
        default=2000,
        help="candidate split seeds searched on the first run (size and language balance)",
    )
    parser.add_argument("--lid-model", type=Path, help="fastText lid.176 model; without it no language id runs")
    # Label round v2. Defaults build v1 labels.
    parser.add_argument(
        "--label-rules",
        choices=("v1", "v2", "v2-orthographic", "v3", "v3-orthographic", "v3-sung"),
        default="v1",
        help="shared normalizer rules (label_rules.py); v2 must ship with a v2-trained model",
    )
    parser.add_argument(
        "--label-policy",
        choices=("v1", "v2"),
        default="v1",
        help="v2: label_policy + tail trim from the activity arrays; label schema 7",
    )
    parser.add_argument(
        "--activity-dir", type=Path, help="vocal_activity.py store or .npz arrays (default: <cache-dir>/activity)"
    )
    parser.add_argument(
        "--require-activity",
        action="store_true",
        help="strict guard: require loadable activity for every emitted train/validation song (policy v2)",
    )
    parser.add_argument(
        "--activity-baseline",
        type=Path,
        help="prior policy-v2 full manifest directory; preserve retained songs’ activity sources",
    )
    parser.add_argument(
        "--reference-labels",
        type=Path,
        help="labels cache whose timings label quality and the offset search are scored on; "
        "documents without a reference label are not built",
    )
    parser.add_argument(
        "--split-assignments-from",
        type=Path,
        help="copy this split_assignments.json into a new output (keeps an earlier build's artist split)",
    )
    parser.add_argument(
        "--offset-gate",
        choices=("record", "drop"),
        default="record",
        help="drop: songs with a confirmed whole-song offset, and live/remaster/edit titles with a flagged one",
    )
    parser.add_argument(
        "--near-duplicates",
        choices=("keep", "drop"),
        default="keep",
        help="drop: keep one recording per (title, artist, duration within 1.5 s) group",
    )
    args = parser.parse_args()
    if (args.require_activity or args.activity_baseline) and args.label_policy != "v2":
        parser.error("Activity guards require --label-policy v2")
    if args.lid_model and not args.lid_model.exists():
        parser.error(f"--lid-model not found: {args.lid_model}")
    if not args.lid_model:
        print("WARNING: no --lid-model; languages come from the documents alone", file=sys.stderr)
    activity_baseline = load_activity_baseline(args.activity_baseline) if args.activity_baseline else {}
    v2_build = args.label_rules != "v1" or args.label_policy != "v1"

    history = None
    history_root = args.split_assignments_from.parent if args.split_assignments_from else args.output
    if args.split_assignments_from or (history_root / "compact" / "train.jsonl").exists():
        history = load_split_history(history_root / "compact")

    started = time.time()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.split_assignments_from and not (output / "split_assignments.json").exists():
        (output / "split_assignments.json").write_text(args.split_assignments_from.read_text())
    caches = (args.cache_dir or output / "cache").resolve()
    (caches / "labels").mkdir(parents=True, exist_ok=True)
    (caches / "quality").mkdir(exist_ok=True)
    audio_root = args.audio_root.resolve()
    recordings, duplicate_ids = load_index(args.index.resolve())
    held_out_test = args.held_out_test.resolve() if args.held_out_test else None
    from echora_kara.text.normalization import NORMALIZER_VERSION

    # 1. Documents with audio present.
    documents: dict[str, dict] = {}
    rejected = Counter()
    if duplicate_ids:
        rejected["duplicate_isrc"] += duplicate_ids
    for isrc in sorted(recordings):
        entry = recordings[isrc]
        if not (audio_root / entry["audio"]).exists():
            rejected["audio_missing"] += 1
            continue
        if args.reference_labels and not (args.reference_labels / f"{isrc}.json").exists():
            rejected["no_reference_labels"] += 1
            continue
        if not entry["lyrics"].exists():
            rejected["lyrics_missing"] += 1
            continue
        documents[isrc] = {"path": entry["lyrics"]}
        if args.limit and len(documents) >= args.limit:
            break
    print(f"documents with audio: {len(documents)} (no audio yet: {rejected['audio_missing']})", flush=True)

    # 2. Labels (cached per document content + normalizer version).
    todo = []
    if v2_build:
        todo = v2_label_tasks(documents, caches, args)
    for isrc, doc in [] if v2_build else documents.items():
        cached = caches / "labels" / f"{isrc}.json"
        doc["labels_path"] = cached
        if cached.exists():
            try:
                head = json.loads(cached.read_text())
                doc_sha = document_digest(read_ttml(doc["path"]))
                if head.get("key") == labels_key(doc_sha):
                    continue
            except (ValueError, KeyError, OSError):
                pass
        todo.append((isrc, str(doc["path"])))
    print(
        f"normalizing {len(todo)} documents ({len(documents) - len(todo)} cached) with {args.workers} workers",
        flush=True,
    )
    if todo:
        with Pool(args.workers, initializer=_init_worker, initargs=(args.label_rules, args.label_policy)) as pool:
            for count, (isrc, record) in enumerate(pool.imap_unordered(normalize_document, todo, chunksize=4), 1):
                (caches / "labels" / f"{isrc}.json").write_text(json.dumps(record, ensure_ascii=False))
                if count % 2000 == 0:
                    print(f"  normalized {count}/{len(todo)} ({time.time() - started:.0f}s)", flush=True)

    # 2b. fastText language identification (cached by document content).
    lid_cache_path = caches / "lid.jsonl"
    lid = {record["key"]: record for record in read_jsonl_cache(lid_cache_path).values()}
    lid_todo = []
    for isrc, doc in documents.items():
        head = json.loads(doc["labels_path"].read_text())
        doc["doc_sha256"], doc["labels_head"] = head["doc_sha256"], {k: v for k, v in head.items() if k != "timings"}
        if head["doc_sha256"] not in lid:
            lid_todo.append((isrc, doc["path"]))
    if lid_todo and args.lid_model:
        print(f"language id for {len(lid_todo)} documents ({len(lid)} cached)", flush=True)
        with lid_cache_path.open("a") as sink:
            for record in run_lid(lid_todo, args.lid_model):
                record = {"key": record["doc_sha256"], **record}
                lid[record["key"]] = record
                sink.write(json.dumps(record) + "\n")

    # 3. Audio probes (cached by path, size, mtime).
    probe_cache_path = caches / "audio_probe.jsonl"
    probes = {record["key"]: record for record in read_jsonl_cache(probe_cache_path).values()}
    probe_todo = []
    for isrc in (
        list(documents) + [json.loads(l)["id"] for l in held_out_test.open()] if held_out_test else list(documents)
    ):
        audio = audio_root / audio_path(isrc, recordings)
        if not audio.exists():
            continue
        stat = audio.stat()
        cached = probes.get(str(audio))
        if cached and cached.get("size") == stat.st_size and abs(cached.get("mtime", -1) - stat.st_mtime) < 1e-6:
            continue
        entry = recordings.get(isrc, {})
        probe_todo.append((isrc, audio, entry.get("audio_sha256"), entry.get("audio_bytes")))
    print(f"probing {len(probe_todo)} audio files ({len(probes)} cached)", flush=True)
    if probe_todo:
        with ThreadPoolExecutor(max_workers=args.workers) as pool, probe_cache_path.open("a") as sink:
            futures = {pool.submit(probe_audio, audio, sha, size): audio for _, audio, sha, size in probe_todo}
            for count, future in enumerate(as_completed(futures), 1):
                record = {"key": str(futures[future]), **future.result()}
                probes[record["key"]] = record
                sink.write(json.dumps(record) + "\n")
                if count % 2000 == 0:
                    print(f"  probed {count}/{len(probe_todo)} ({time.time() - started:.0f}s)", flush=True)

    # 4. Curation and artist components.
    meta: dict[str, dict] = {}
    labels: dict[str, dict] = {}
    accepted: list[str] = []
    for isrc, doc in documents.items():
        track = recordings[isrc]
        head = dict(doc["labels_head"])
        record = {**head, **resolve_language(head, lid.get(doc["doc_sha256"]))}
        if args.languages and record["language"] not in args.languages:
            rejected["language_filtered"] += 1
            continue
        probe = probes.get(str(audio_root / track["audio"]))
        if not probe or probe.get("error"):
            rejected["ffprobe_failed"] += 1
            continue
        identity = audio_identity(isrc, track)
        if identity["audio_match_status"] == "exact_isrc" and identity.get("index_sha256") not in (
            None,
            probe["sha256"],
        ):
            identity = {"audio_match_status": "unverified_legacy"}
        if identity["audio_match_status"] == "isrc_mismatch":
            rejected["known_wrong_recording"] += 1
            continue
        if record["nonfinite"]:
            rejected["nonfinite_timing"] += 1
            continue
        if record["usable_units"] < MIN_UNITS:
            rejected["too_few_units"] += 1
            continue
        catalog = track.get("duration")
        limit = MAX_CATALOG_DURATION_DIFF.get(identity["audio_match_status"], 2.0)
        if (
            not isinstance(catalog, (int, float))
            or not math.isfinite(catalog)
            or abs(probe["duration"] - catalog) > limit
        ):
            rejected["catalog_duration_mismatch"] += 1
            continue
        if record["lyric_end"] > probe["duration"] + TOLERANCE_SECONDS:
            rejected["lyrics_after_audio"] += 1
            continue
        if record["usable_end"] - record["usable_start"] < probe["duration"] * MIN_COVERAGE:
            rejected["coverage_under_25pct"] += 1
            continue
        artists = credited_artists(track.get("artist") or "")
        if not artists:
            rejected["artist_missing"] += 1
            continue
        meta[isrc] = {
            "track": track,
            "identity": identity,
            "probe": probe,
            "artists": artists,
            "lyric_match_status": "exact_isrc" if track.get("lyric_match_status") == "exact_isrc" else UNVERIFIED_LYRIC,
        }
        labels[isrc] = record
        accepted.append(isrc)
    # Exact duplicate audio content: keep none (as curate.py).
    digests = Counter(meta[isrc]["probe"]["sha256"] for isrc in accepted)
    duplicates = [isrc for isrc in accepted if digests[meta[isrc]["probe"]["sha256"]] > 1]
    rejected["duplicate_audio_content"] += len(duplicates)
    accepted = [isrc for isrc in accepted if digests[meta[isrc]["probe"]["sha256"]] == 1]

    # 5. Label quality (cached by audio hash + labels key).
    quality: dict[str, dict] = {}
    if not args.no_label_quality:
        from echora_kara.data import label_quality

        quality_todo = []
        reference_keys = {}
        if args.reference_labels:
            # Scored on the reference timings, which the 0.05 threshold and the offset gate were
            # calibrated on.
            tasks = [
                (isrc, str(args.reference_labels / f"{isrc}.json"), meta[isrc]["probe"]["sha256"]) for isrc in accepted
            ]
            with Pool(args.workers) as pool:
                for isrc, key in pool.imap_unordered(_reference_quality_key, tasks, chunksize=16):
                    reference_keys[isrc] = key
        for isrc in accepted:
            cached = caches / "quality" / f"{isrc}.json"
            key = (
                reference_keys.get(isrc)
                if args.reference_labels
                else f"{meta[isrc]['probe']['sha256']}:{labels[isrc]['key']}:{label_quality.VERSION}"
            )
            if cached.exists():
                try:
                    record = json.loads(cached.read_text())
                    if record.get("key") == key and not record["quality"].get("error"):
                        quality[isrc] = record["quality"]
                        continue
                except (ValueError, OSError):
                    pass
            quality_todo.append(
                (
                    isrc,
                    str(audio_root / recordings[isrc]["audio"]),
                    str(args.reference_labels / f"{isrc}.json")
                    if args.reference_labels
                    else str(documents[isrc]["labels_path"]),
                    key,
                )
            )
        print(f"scoring label quality for {len(quality_todo)} songs ({len(quality)} cached)", flush=True)
        if quality_todo:
            with Pool(args.workers) as pool:
                tasks = [(isrc, audio, labels_path) for isrc, audio, labels_path, _ in quality_todo]
                keys = {isrc: key for isrc, _, _, key in quality_todo}
                for count, (isrc, score) in enumerate(pool.imap_unordered(_quality_worker, tasks, chunksize=2), 1):
                    (caches / "quality" / f"{isrc}.json").write_text(json.dumps({"key": keys[isrc], "quality": score}))
                    quality[isrc] = score
                    if count % 1000 == 0:
                        print(f"  scored {count}/{len(quality_todo)} ({time.time() - started:.0f}s)", flush=True)
        if args.min_label_quality is not None:
            low = [
                isrc
                for isrc in accepted
                if (quality.get(isrc) or {}).get("score") is not None
                and quality[isrc]["score"] < args.min_label_quality
            ]
            rejected["label_quality_below_threshold"] += len(low)
            accepted = [isrc for isrc in accepted if isrc not in set(low)]
    gate_ids: dict[str, list[str]] = defaultdict(list)
    if args.offset_gate == "drop":
        kept = []
        for isrc in accepted:
            score = quality.get(isrc) or {}
            if score.get("offset_confirmed"):
                gate_ids["offset_confirmed"].append(isrc)
            elif score.get("offset_flagged") and TITLE_FLAG.search(str(meta[isrc]["track"].get("title") or "")):
                gate_ids["offset_flagged_live_remaster_edit"].append(isrc)
            else:
                kept.append(isrc)
        for reason, isrcs in gate_ids.items():
            rejected[reason] += len(isrcs)
        accepted = kept
    if args.near_duplicates == "drop":
        accepted, duplicates_dropped = drop_near_duplicates(accepted, meta, quality)
        rejected["near_duplicate_recording"] += len(duplicates_dropped)
        gate_ids["near_duplicate_recording"] = duplicates_dropped

    # 6. Held-out test songs and artists; artist components; persisted split assignment.
    held_isrcs, held_artists = set(), set()
    if held_out_test:
        for line in held_out_test.open():
            row = json.loads(line)
            held_isrcs.add(row["id"])
            held_artists.update(credited_artists(row.get("artist", "")))
    components = Components()
    for isrc in accepted:
        components.union(meta[isrc]["artists"])
    (output / "compact").mkdir(exist_ok=True)
    for name in held_artists:
        components.find(name)
    # Songs crediting a held-out test artist are excluded. The rest of such an artist's component (its
    # collaborators' collaborators) stays trainable but never enters validation: the collaboration
    # graph tends to have one giant component (film music, K-pop, musical casts) that would otherwise
    # take a large share of the collection with it.
    held_components = {components.find(name) for name in held_artists}
    if history is not None:
        check_heldout_history(held_isrcs, history, "test")
    assignments_path = output / "split_assignments.json"
    assignments = json.loads(assignments_path.read_text()) if assignments_path.exists() else {}
    split_of: dict[str, str] = {}
    by_component: dict[str, list[str]] = defaultdict(list)
    for isrc in accepted:
        component = components.find(meta[isrc]["artists"][0])
        meta[isrc]["component"] = component
        if isrc in held_isrcs or held_artists.intersection(meta[isrc]["artists"]):
            rejected["held_out_test_song_or_artist"] += 1
            continue
        if component in held_components:
            # Train-only: the component of a held-out test artist's collaborators.
            rejected["held_out_component_kept_in_train"] += 1
        by_component[component].append(isrc)

    if history is not None:
        by_component, quarantined = quarantine_validation_admissions(by_component, history, assignments)
        rejected["new_admission_in_historical_validation_component"] += len(quarantined)
        gate_ids["new_admission_in_historical_validation_component"] = quarantined
        print(f"quarantined {len(quarantined)} new admissions in historical validation components", flush=True)

    verified_isrc = {
        isrc: identity_problem(
            {"id": isrc, **meta[isrc]["identity"], "lyric_match_status": meta[isrc]["lyric_match_status"]}, "validation"
        )
        is None
        for isrcs in by_component.values()
        for isrc in isrcs
    }

    def verified(isrc: str) -> bool:
        return verified_isrc[isrc]

    # Validation target: the share of *all* songs that end in train or validation, drawn only from the
    # eligible components. Seed search: squared size error,
    # per-language targets for languages with >= 100 songs, and a cost for unverified songs dropped
    # from validation. Components assigned on an earlier run keep their split; the seed persists.
    total_songs = len(split_of) + sum(map(len, by_component.values()))
    target = args.validation_share * total_songs
    language_of = {isrc: labels[isrc]["language"] for isrc in accepted}
    language_total = Counter(language_of[isrc] for isrcs in by_component.values() for isrc in isrcs)
    language_total.update(language_of[isrc] for isrc in split_of)
    major = [language for language, count in language_total.items() if count >= 100]
    fixed = {component: assignments[component] for component in by_component if component in assignments}
    fixed.update({component: "train" for component in by_component if component in held_components})
    if history is not None:
        for component, songs in by_component.items():
            fixed[component] = historical_component_split(component, songs, history, fixed.get(component))
    pending = sorted(component for component in by_component if component not in fixed)

    def evaluate(seed: int) -> float:
        size, dropped, per_language = 0, 0, Counter()
        for component, isrcs in by_component.items():
            split = fixed.get(component) or split_hash(component, f"{args.seed}:{seed}", args.validation_share)
            if split != "validation":
                continue
            for isrc in isrcs:
                if verified(isrc):
                    size += 1
                    per_language[language_of[isrc]] += 1
                else:
                    dropped += 1
        cost = 8 * (size - target) ** 2 / max(target, 1) + 0.1 * dropped
        for language in major:
            expected = language_total[language] * args.validation_share
            cost += (per_language[language] - expected) ** 2 / max(expected, 8)
        return cost

    if "_seed" in assignments:
        seed = int(assignments["_seed"])
    else:
        seed = min(range(max(1, args.search_seeds)), key=evaluate) if pending else 0
    for component in pending:
        fixed[component] = split_hash(component, f"{args.seed}:{seed}", args.validation_share)
    for component, isrcs in by_component.items():
        split = fixed[component]
        for isrc in isrcs:
            if split == "validation" and not verified(isrc):
                rejected["validation_identity_unverified"] += 1
                continue
            if split == "train" and identity_problem(
                {"id": isrc, **meta[isrc]["identity"], "lyric_match_status": "exact_isrc"}, "train"
            ):
                rejected["unknown_identity_status"] += 1
                continue
            split_of[isrc] = split
    activity_root = args.activity_dir or (caches / "activity")
    activity_report = None
    if args.label_policy == "v2":
        from echora_kara.data.vocal_activity import load_activity_from

        activity_report = activity_coverage(
            split_of, activity_root, load_activity_from, args.require_activity, activity_baseline
        )
        activity_report["baseline"] = str(args.activity_baseline) if args.activity_baseline else None
        print("activity coverage: " + json.dumps(activity_report), flush=True)
        if activity_report["missing"]:
            print(
                "WARNING: missing activity; tail trim disabled for those songs. "
                "Use --activity-baseline to preserve prior coverage or --require-activity for full coverage.",
                file=sys.stderr,
            )
    if history is not None:
        check_heldout_history(
            (song for song, split in split_of.items() if split == "validation"), history, "validation"
        )
    assignments = {
        "_seed": seed,
        **{component: split for component, split in fixed.items()},
        **{component: "train" for component in held_components},
    }
    assignments_path.write_text(json.dumps(assignments, ensure_ascii=False, indent=0, sort_keys=True))

    # 7. Write manifests and check every row structurally.
    checks = new_assignments()
    counts, languages, mask_reasons, errors = (
        Counter(),
        {s: Counter() for s in ("train", "validation", "test")},
        Counter(),
        [],
    )
    quality_scores = {s: [] for s in ("train", "validation")}
    seconds = Counter()
    trim_stats, trim_seconds = Counter(), Counter()
    background_counts = {s: Counter() for s in ("train", "validation")}
    policy_counts = {s: defaultdict(Counter) for s in ("train", "validation")}
    activity_root = args.activity_dir or (caches / "activity")
    for split in ("train", "validation"):
        with (
            (output / f"{split}.jsonl").open("w") as sink,
            (output / "compact" / f"{split}.jsonl").open("w") as compact_sink,
        ):
            for isrc in accepted:
                if split_of.get(isrc) != split:
                    continue
                record = {**json.loads(documents[isrc]["labels_path"].read_text()), **labels[isrc]}
                v2 = None
                if v2_build:
                    timings, background, trim = record["timings"], [], {"activity": None}
                    if args.label_policy == "v2":
                        from echora_kara.data import label_policy as label_policy_v2
                        from echora_kara.data.vocal_activity import load_activity_from

                        activity = load_activity_from(activity_root, isrc)
                        check_baseline_activity(isrc, activity, activity_baseline)
                        if args.require_activity and activity is None:
                            raise ValueError(f"Required activity disappeared after preflight: {isrc}")
                        timings, background, trim = label_policy_v2.finalize(timings, activity)
                        trim_stats[trim["activity"]] += 1
                        trim_seconds[trim["activity"]] += trim.get("tail_seconds_trimmed", 0.0)
                        for key in (
                            "background_phrases_sequential",
                            "background_phrases_simultaneous",
                            "background_units_sequential",
                            "background_units_simultaneous",
                            "lead_overlaps",
                            "lead_overlaps_trimmed",
                        ):
                            policy_counts[split][labels[isrc]["language"]][key] += trim.get(key, 0)
                    v2 = {
                        "timings": timings,
                        "background": background,
                        "activity": trim,
                        "rules": args.label_rules,
                        "policy": args.label_policy,
                        "hint": record["normalizer_language"],
                        "line_repeat": record.get("line_repeat"),
                    }
                row = prepared_row(
                    isrc,
                    meta[isrc],
                    record,
                    meta[isrc]["probe"],
                    meta[isrc]["identity"],
                    split,
                    meta[isrc]["artists"],
                    meta[isrc]["component"],
                    quality.get(isrc),
                    v2,
                )
                try:
                    check_row(row, split, checks, audio_root, prepared=True)
                except ValueError as error:
                    errors.append(str(error))
                    rejected["integrity_check_failed"] += 1
                    continue
                sink.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                compact_sink.write(json.dumps(compact_row(row), ensure_ascii=False, allow_nan=False) + "\n")
                counts[split] += 1
                languages[split][row["language"]] += 1
                for unit in row["alignment_tokens"]:
                    if unit.get("ignore"):
                        mask_reasons[unit.get("normalization_error", "unspecified").split(":")[0]] += 1
                    elif unit.get("vocal_role") != "background":
                        seconds[split] += unit["end"] - unit["start"]
                if (row.get("label_quality") or {}).get("score") is not None:
                    quality_scores[split].append(row["label_quality"]["score"])
                if "background_tokens" in row:
                    supervised = [t for t in row["background_tokens"] if not t.get("ignore")]
                    background_counts[split]["background_head_units"] += len(supervised)
                    background_counts[split]["background_head_units_under_lead"] += sum(
                        1 for t in supervised if t.get("lead_overlap", 0) > 0
                    )
                    background_counts[split]["main_head_background_soft_masked"] += sum(
                        1 for t in row["alignment_tokens"] if t.get("vocal_role") == "background" and t.get("ignore")
                    )
                    background_counts[split]["background_head_seconds"] += sum(
                        t["end"] - t["start"] for t in row["background_tokens"]
                    )
    test_rows, test_skipped = held_out_rows(held_out_test, audio_root, probes, recordings)
    with (output / "test.jsonl").open("w") as sink, (output / "compact" / "test.jsonl").open("w") as compact_sink:
        for row in test_rows:
            try:
                check_row(row, "test", checks, audio_root, prepared=True)
            except ValueError as error:
                errors.append(f"test: {error}")
                test_skipped["test_integrity_check_failed"] += 1
                continue
            sink.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            compact_sink.write(json.dumps(compact_row(row), ensure_ascii=False, allow_nan=False) + "\n")
            counts["test"] += 1
            languages["test"][row["language"]] += 1
    (output / "vocab.json").write_text(json.dumps(VOCAB, indent=2) + "\n")
    (output / "compact" / "vocab.json").write_text(json.dumps(VOCAB, indent=2) + "\n")

    def percentiles(values):
        if not values:
            return None
        values = sorted(values)
        return {
            f"p{p}": round(values[min(len(values) - 1, int(p / 100 * len(values)))], 4) for p in (5, 10, 25, 50, 75, 90)
        }

    summary = {
        "builder": BUILDER_VERSION,
        "normalizer": NORMALIZER_VERSION,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "audio_root": str(audio_root),
        "audio_layout": "<audio-root>/<index audio path> (full mixes)",
        "label_schema_version": 6,
        "splits": {s: counts[s] for s in ("train", "validation", "test")},
        "total": sum(counts.values()),
        "supervised_hours": {s: round(seconds[s] / 3600, 2) for s in ("train", "validation")},
        "languages": {s: dict(languages[s].most_common()) for s in languages},
        "index": str(args.index.resolve()),
        "documents_cached": len(recordings),
        "documents_with_audio": len(documents),
        "rejected": dict(rejected.most_common()),
        "test_skipped": dict(test_skipped),
        "mask_reasons": dict(mask_reasons.most_common()),
        "label_quality": {
            "method": None if args.no_label_quality else "label_quality_mfcc_v1",
            "filter_threshold": args.min_label_quality,
            "score_percentiles": {s: percentiles(v) for s, v in quality_scores.items()},
        },
        "split_method": f"95/5 by connected components of credited artists, sha256 seed {args.seed!r}; "
        "assignments persisted in split_assignments.json; songs crediting a held-out test artist excluded, "
        "their wider components train-only",
        "activity": activity_report,
        "split_history": str(history_root) if history is not None else None,
        "held_out": {
            "source": str(held_out_test) if held_out_test else None,
            "songs": len(held_isrcs),
            "artists": len(held_artists),
        },
        "constraints": {
            "max_label_chars": MAX_LABEL_CHARS,
            "min_units": MIN_UNITS,
            "min_lyric_coverage": MIN_COVERAGE,
            "max_catalog_duration_diff_seconds": MAX_CATALOG_DURATION_DIFF,
            "artist_disjoint": True,
        },
        "language_sources": dict(Counter(labels[isrc]["language_source"] for isrc in split_of)),
        "manifest_bytes": {name: (output / f"{name}.jsonl").stat().st_size for name in ("train", "validation", "test")},
        "compact_manifest_bytes": {
            name: (output / "compact" / f"{name}.jsonl").stat().st_size for name in ("train", "validation", "test")
        },
        "integrity_errors": errors[:50],
        "elapsed_seconds": round(time.time() - started),
    }
    if v2_build:
        from echora_kara.text.normalization import normalizer_version
        from echora_kara.text import label_rules as label_rules_module

        summary.update(
            {
                "builder": BUILDER_VERSION_V2,
                "normalizer": normalizer_version(args.label_rules),
                "label_schema_version": 7,
                "label_rules": args.label_rules,
                "label_policy": args.label_policy,
                "espeak_ng": label_rules_module.espeak_version(),
                "reference_labels": str(args.reference_labels) if args.reference_labels else None,
                "label_quality": {
                    **summary["label_quality"],
                    "method": "label_quality_mfcc_v2 on the reference labels"
                    if args.reference_labels
                    else "label_quality_mfcc_v2",
                },
                "gates": {
                    "offset_gate": args.offset_gate,
                    "near_duplicates": args.near_duplicates,
                    **{reason: len(isrcs) for reason, isrcs in gate_ids.items()},
                },
                "tail_trim": {
                    "songs_by_activity_source": {str(k): v for k, v in trim_stats.items()},
                    "seconds_trimmed_by_source": {str(k): round(v, 1) for k, v in trim_seconds.items()},
                },
                "background_head": {
                    split: {k: round(v, 1) for k, v in counts.items()} for split, counts in background_counts.items()
                },
                "label_policy_counts": {
                    split: {
                        language: dict(c) for language, c in sorted(by.items(), key=lambda kv: -sum(kv[1].values()))
                    }
                    for split, by in policy_counts.items()
                },
            }
        )
        (output / "gates").mkdir(exist_ok=True)
        for reason, isrcs in gate_ids.items():
            (output / "gates" / f"{reason}.txt").write_text("".join(f"{isrc}\n" for isrc in sorted(isrcs)))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (output / "compact" / "summary.json").write_text(
        json.dumps(
            {
                **summary,
                "note": "compact manifest for training: trainer-read fields only; full rows in the parent directory",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("languages", "mask_reasons")}, ensure_ascii=False, indent=1
        )
    )
    print("languages:", json.dumps(summary["languages"], ensure_ascii=False))


if __name__ == "__main__":
    main()
