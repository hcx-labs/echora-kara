"""Read-only structural checks shared by preparation and publication audits."""
import hashlib
import math
import unicodedata
from pathlib import Path

SPLITS = ("train", "validation", "test")
# Training tolerates a few legacy recording mismatches as label noise.
# Evaluation numbers select checkpoints, so they need proven identity.
VERIFIED_SPLITS = ("validation", "test")
TRAIN_AUDIO_STATUSES = {"exact_isrc", "unverified_legacy"}
TRAIN_LYRIC_STATUSES = {"exact_isrc", "unverified_provider_lookup"}
# 6: label policy v1. 7: the same fields plus optional per-unit provider_end (the source end before
# trimming) / tail_trim / overlap / phrase / background_source and per-row line_repeat, offset
# (label_quality) and near_duplicate_group; the trainer reads both identically.
LABEL_SCHEMA_VERSIONS = {6, 7}


def artist_key(artist: str) -> str:
    return unicodedata.normalize("NFKC", artist).casefold().strip()


def split_for_artist(artist: str) -> str:
    bucket = int(hashlib.sha256(artist_key(artist).encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 90 else "validation" if bucket < 95 else "test"


def identity_problem(row, split):
    """Why this row's recording identity is unacceptable for its split, or None."""
    audio, lyric = row.get('audio_match_status'), row.get('lyric_match_status')
    if audio == 'exact_isrc' and (row.get('requested_isrc') != row.get('id') or row.get('matched_isrc') != row.get('id')):
        return 'audio_provenance_mismatch'
    if audio == 'isrc_mismatch':
        return 'known_wrong_recording'
    if split in VERIFIED_SPLITS:
        if lyric != 'exact_isrc':
            return 'unverified_lyric_identity'
        if audio != 'exact_isrc':
            return 'unverified_audio_identity'
    elif audio not in TRAIN_AUDIO_STATUSES or lyric not in TRAIN_LYRIC_STATUSES:
        return 'unknown_identity_status'
    return None


def check_row(row, split, assignments, root=None, prepared=False):
    if row.get("split") != split:
        raise ValueError(f"{row.get('id')}: split field does not match manifest")
    problem = identity_problem(row, split)
    if problem:
        raise ValueError(f"{row.get('id')}: {problem}")
    digest = row.get('audio_sha256', '')
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError(f"{row.get('id')}: audio content hash missing")
    for field in ("id", "audio", "artist_key", "audio_sha256"):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"missing {field}")
        previous = assignments[field].get(value)
        if previous is not None and (field != "artist_key" or previous != split):
            raise ValueError(f"duplicate or cross-split {field}: {value}")
        assignments[field][value] = split
    duration = row.get("duration")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"{row['id']}: invalid physical duration")
    if row.get("duration_source") != "ffprobe_format":
        raise ValueError(f"{row['id']}: physical audio QC is missing")
    if root is not None:
        path = (Path(root) / row["audio"]).resolve()
        if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
            raise ValueError(f"{row['id']}: missing or unsafe audio path")
    source_units = [u for line in row['timings'] for u in line]
    units = row["alignment_tokens"] if prepared else source_units
    if prepared:
        if row.get('label_schema_version') not in LABEL_SCHEMA_VERSIONS or row.get('pad_token_id') != 29:
            raise ValueError(f"{row['id']}: obsolete prepared-label schema")
        signature = lambda u: (u['start'], u['end'], u['text'], bool(u.get('ignore')))
        if list(map(signature, units)) != list(map(signature, source_units)):
            raise ValueError(f"{row['id']}: prepared labels differ from curated timings")
        if row.get('line_transcripts') != [''.join(u['text'] for u in line) for line in row['timings']]:
            raise ValueError(f"{row['id']}: transcript differs from curated timings")
    usable = 0
    for unit in units:
        start, end = unit["start"], unit["end"]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (start, end)):
            raise ValueError(f"{row['id']}: nonfinite interval")
        if not 0 <= start < end <= duration:
            raise ValueError(f"{row['id']}: interval outside audio")
        if unit.get("ignore"):
            if unit.get("text") or (prepared and unit.get("ids")):
                raise ValueError(f"{row['id']}: mask contains supervised labels")
        else:
            text = unit.get("text", "")
            if not text or any(not 'a' <= c <= 'z' for c in text):
                raise ValueError(f"{row['id']}: unsafe pronunciation")
            if unit.get("timing_method") != "weak_characters_within_source_fragment":
                raise ValueError(f"{row['id']}: obsolete timing policy")
            if len(unit.get("source_spans", [])) != 1:
                raise ValueError(f"{row['id']}: merged source timing fragments")
            span = unit["source_spans"][0]
            if unit["start"] < span["start"] - 1e-8 or unit["end"] > span["end"] + 1e-8:
                raise ValueError(f"{row['id']}: source boundary not preserved")
            if prepared and unit["ids"] != [ord(c) - ord('a') + 1 for c in text]:
                raise ValueError(f"{row['id']}: vocabulary mismatch")
            usable += 1
    if not usable:
        raise ValueError(f"{row['id']}: no usable supervision")
    return usable


def new_assignments():
    return {field: {} for field in ("id", "audio", "artist_key", "audio_sha256")}
