"""Whole-song decoded-timing evaluation inside training.

At every evaluation the training model aligns a fixed set of validation songs the way production does:
45 s chunks with 2 s shared context on the full mix, the letter emission decoded by the harness decoder
(``evaluation.decode.decode``: vocal-activity penalty, paced floors, in-line gap cost) and, when
the model has a text head, the same decode with the head's token share as a bonus (beta x log P(token |
singing) added to each word's letters; ECHORA_KARA_TEXT_BONUS at inference). Word starts on the exact frame
clock against the reference starts give eval_decoded_{median_ms, mean_ms, over500, f1_50} (and
eval_decoded_bonus_*), the numbers the held-out bench reports. The background head gets a threshold-free
average precision (1 - P(blank) per frame against the reference background-token frames): eval_background_ap.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from echora_kara.evaluation import validation as base
from echora_kara.evaluation import decode as s3
from echora_kara.evaluation import interval_decoder as proto
from echora_kara.engine.ctc_segmentation import (
    align_with_minimum_durations,
    align_with_source_priors,
    paced_minimum_frames,
    stitched_frame_times,
    vocal_activity,
    with_vocal_activity)
from echora_kara.model import text_aligner as ta

BLANK = proto.BLANK


def _summary(errors_ms: np.ndarray, prefix: str) -> dict:
    absolute = np.abs(errors_ms)
    return {f"{prefix}median_ms": float(np.median(absolute)), f"{prefix}mean_ms": float(absolute.mean()),
            f"{prefix}over500": int((absolute > 500).sum()), f"{prefix}f1_50": float((absolute <= 50).mean())}


def _average_precision(scores: np.ndarray, truth: np.ndarray) -> float:
    if truth.sum() == 0:
        return float("nan")
    order = np.argsort(-scores)
    hits = truth[order].astype(np.float64)
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum())


class DecodedEval:
    def __init__(self, root: Path, manifest: Path, ids_file: Path, songs: int, song_text: dict | None,
                 bonus: float = 0.1, gap_cost: float = 0.2):
        wanted = [line.strip() for line in open(ids_file) if line.strip() and not line.startswith("#")]
        rows = {}
        with open(manifest) as source:
            for line in source:
                row = json.loads(line)
                if row["id"] in wanted:
                    rows[row["id"]] = row
        self.songs = [rows[i] for i in wanted if i in rows][:songs]
        self.root, self.song_text, self.bonus, self.gap_cost = Path(root), song_text, bonus, gap_cost
        self.audio: dict[str, np.ndarray] = {}
        print(f"decoded eval: {len(self.songs)} songs from {ids_file}", flush=True)

    def _outputs(self, model, audio, song, device):
        """Stitched letter log-probs [T, V], lead-token share [T, words] (or None), background P(sung) [T]."""
        chunk, overlap = 45 * proto.SAMPLE_RATE, s3.OVERLAP_SECONDS * proto.SAMPLE_RATE
        starts = [s for s in range(0, len(audio), chunk - overlap) if s == 0 or len(audio) - s > overlap]
        head = getattr(model, "text_aligner", None)
        text = None
        if head is not None and self.song_text is not None and song["id"] in self.song_text:
            record = self.song_text[song["id"]]
            n = len(record["voice"])
            text = {"letters": torch.from_numpy(record["letters"].astype(np.int64))[None].to(device),
                    "voice": torch.from_numpy(record["voice"].astype(np.int64))[None].to(device),
                    "line": torch.from_numpy(record["line"].astype(np.int64))[None].to(device),
                    "position": torch.from_numpy(record["position"].astype(np.int64))[None].to(device),
                    "text_padding": torch.zeros((1, n), dtype=torch.bool, device=device)}
        letters, tokens, backing = [], [], []
        for index, start in enumerate(starts):
            piece = audio[start:start + chunk]
            captured = {}
            hook = model.lm_head.register_forward_hook(lambda _m, inputs, _o: captured.__setitem__("hidden", inputs[0]))
            try:
                extra = ({"fusion_input_values": torch.from_numpy(piece)[None].to(device),
                          "fusion_lengths": torch.tensor([len(piece)], device=device)}
                         if getattr(model, "fusion", None) is not None else {})
                normalised = (piece - piece.mean()) / (piece.std() + 1e-7)
                out = model(input_values=torch.from_numpy(normalised)[None].to(device), **extra)
            finally:
                hook.remove()
            letter = torch.log_softmax(out.logits[0].float(), -1)
            token = None
            if text is not None:
                hidden = captured["hidden"].float()
                padding = torch.zeros(hidden.shape[:2], dtype=torch.bool, device=device)
                part = head.forward_parts(hidden, padding, text["letters"], text["voice"], text["text_padding"],
                                          letter[None], text["line"], text["position"])[ta.LEAD]
                token = part["token"][0] if "token" in part else torch.log_softmax(part["combined"][0][:, 1:], -1)
            bg = getattr(out, "background_logits", None)
            bg = (1 - torch.softmax(bg[0].float(), -1)[:, BLANK]) if bg is not None else None
            frames = round(letter.shape[0] * overlap / max(1, len(piece)))
            left = frames // 2 if index else 0
            right = frames - frames // 2 if index + 1 < len(starts) else 0
            stop = letter.shape[0] - right if right else None
            letters.append(letter[left:stop].cpu())
            if token is not None:
                tokens.append(token[left:stop].float().cpu())
            if bg is not None:
                backing.append(bg[left:stop].cpu())
        letters = torch.cat(letters).numpy()
        tokens = torch.cat(tokens) if tokens else None
        backing = torch.cat(backing).numpy() if backing else None
        return letters, tokens, backing

    def _bonus_decode(self, letters, audio, words, share, exact):
        em = torch.from_numpy(letters)
        adjusted = with_vocal_activity(em, vocal_activity(torch.from_numpy(audio), em.shape[0]), BLANK, s3.ACTIVITY)
        groups = [w["ids"] for w in words]
        lines = {}
        for index, w in enumerate(words):
            lines.setdefault(w["line"], [index, index + 1])[1] = index + 1
        ranges = [tuple(v) for v in lines.values()]
        plain = align_with_source_priors(adjusted, groups, blank=BLANK, frame_shift_seconds=proto.FRAME, source_priors=[])
        floors = paced_minimum_frames(groups, plain, ranges, ratio=s3.PACE, letter_cap=s3.LETTER_CAP,
                                      group_cap=s3.GROUP_CAP, pause_frames=s3.PAUSE, vowel_ids=s3.VOWELS)
        letter_cols = [letter for g in groups for letter in g]
        owner = [k for k, g in enumerate(groups) for _ in g]
        augmented = torch.cat((adjusted[:, letter_cols] + self.bonus * share[:, owner], adjusted[:, BLANK:BLANK + 1]), 1)
        pseudo, cursor = [], 0
        for g in groups:
            pseudo.append(list(range(cursor, cursor + len(g))))
            cursor += len(g)
        out = align_with_minimum_durations(augmented, pseudo, blank=len(letter_cols), minimum_frames=floors,
                                           blank_free_tokens=True, line_ranges=ranges, in_line_gap_cost=self.gap_cost)
        return np.array([float(exact[g[0].start]) for g in out])

    def run(self, model, device) -> dict:
        was_training = model.training
        model.eval()
        plain, bonus, scores, truth, failed = [], [], [], [], 0
        try:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for song in self.songs:
                    if song["id"] not in self.audio:
                        self.audio[song["id"]] = proto.decode_audio(self.root / s3.mix_path(song))
                    audio = self.audio[song["id"]]
                    words = base.song_words(song)
                    reference = np.array([w["start"] for w in words])
                    try:
                        letters, tokens, backing = self._outputs(model, audio, song, device)
                        exact = stitched_frame_times(len(audio), overlap_seconds=s3.OVERLAP_SECONDS)
                        if exact.shape[0] != letters.shape[0]:
                            failed += 1
                            continue
                        with torch.autocast(device_type="cuda", enabled=False):
                            spans, _ = s3.decode(letters, audio, words, "cpu", blank_free=True, gap_cost=self.gap_cost)
                            plain.append((s3.start_times(spans, exact) - reference) * 1000)
                            if tokens is not None:
                                record = self.song_text[song["id"]]
                                column = {(int(record["line"][j]), round(float(record["start"][j]), 3)): j
                                          for j in range(len(record["voice"])) if record["voice"][j] == ta.LEAD}
                                share = torch.zeros((tokens.shape[0], len(words)))
                                for k, w in enumerate(words):
                                    j = column.get((int(w["line"]), round(float(w["start"]), 3)))
                                    if j is not None:
                                        share[:, k] = tokens[:, j]
                                bonus.append((self._bonus_decode(letters, audio, words, share, exact) - reference) * 1000)
                        if backing is not None:
                            times = np.asarray(exact, dtype=np.float64)
                            sung = np.zeros(len(times), dtype=bool)
                            for t in song.get("background_tokens") or []:
                                if t.get("ids") and not t.get("ignore"):
                                    sung |= (times >= t["start"]) & (times < t["end"])
                            if sung.any():
                                scores.append(backing[: len(times)])
                                truth.append(sung[: len(backing)])
                    except (RuntimeError, ValueError):
                        failed += 1
        finally:
            if was_training:
                model.train()
        metrics = {"decoded_songs": len(plain), "decoded_failed": failed}
        if plain:
            metrics.update(_summary(np.concatenate(plain), "decoded_"))
        if bonus:
            metrics.update(_summary(np.concatenate(bonus), "decoded_bonus_"))
        if scores:
            metrics["background_ap"] = _average_precision(np.concatenate(scores), np.concatenate(truth))
        return metrics
