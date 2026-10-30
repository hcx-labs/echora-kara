#!/usr/bin/env python3
"""Train the Echora Kara aligner on timed lyrics without audio caches.

Needs a CUDA GPU and FFmpeg. AAC is decoded by FFmpeg into a short,
mono, 16 kHz window in each loader worker. Timed character labels are made in
memory once the MMS feature-frame length is known.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    EarlyStoppingCallback,
    SeamlessM4TFeatureExtractor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    Wav2Vec2BertConfig,
    Wav2Vec2BertForCTC,
    Wav2Vec2BertProcessor,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    set_seed,
)
from transformers.modeling_outputs import CausalLMOutput
from transformers.optimization import _get_cosine_schedule_with_warmup_lr_lambda as cosine_lambda
from transformers.utils import ModelOutput
from transformers.trainer_utils import get_last_checkpoint

from echora_kara.train.targets import (
    ctc_regions as find_ctc_regions,
    frame_labels,
    ignore_unregioned,
    tolerant_frame_targets,
)
from echora_kara.train.overlap import LETTERS, PAD_TOKEN_ID as SILENCE_ID, aligned_overlap_targets, multilabel_targets
from echora_kara.train.streams import assign_streams
from echora_kara.train import augment as augmentation
from echora_kara.data.integrity import check_row, new_assignments, SPLITS

from echora_kara.model import muq_fusion  # noqa: E402
from echora_kara.model.song_text import canonical_tokens, text_features, CURRENT_VERSION
from echora_kara.train.checkpoints import mark_and_prune, validate_resume
from echora_kara.model import text_aligner as ta  # noqa: E402  (text-conditioned head, shared with inference)

SAMPLE_RATE = 16_000
PAD_TOKEN_ID = 29


@dataclass(frozen=True)
class FrameClock:
    """Where model frame k sits in the audio: its receptive-field centre is
    ``offset_seconds + k * stride_seconds``. Labels, regions and edge targets are all placed on it.

    ``wav2vec2`` (MMS, omniASR): the 7-layer conv feature encoder, 320-sample stride, 400-sample
    receptive field (centre 199.5). ``wav2vec2-bert`` (w2v-BERT 2.0 and its fine-tunes):
    Kaldi fbank frames of 400 samples every 160, stacked in pairs by the SeamlessM4T feature
    extractor, so a model frame spans samples [320 k, 320 k + 560) and its centre is 320 k + 279.5;
    the frame count is floor(fbank frames / 2) with fbank frames = floor((N - 400) / 160) + 1.
    """

    backbone: str
    stride_seconds: float
    offset_seconds: float

    def frames(self, samples: int, model=None) -> int:
        """Model output frames for ``samples`` input samples (the valid ones: padding excluded)."""
        if self.backbone == "wav2vec2-bert":
            fbank = (samples - 400) // 160 + 1 if samples >= 400 else 0
            return fbank // 2
        if model is not None:
            return int(model._get_feat_extract_output_lengths(samples))
        length = samples
        for kernel, stride in zip((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)):
            length = (length - kernel) // stride + 1
        return max(0, length)


MMS_CLOCK = FrameClock("wav2vec2", 320 / SAMPLE_RATE, 199.5 / SAMPLE_RATE)
W2V_BERT_CLOCK = FrameClock("wav2vec2-bert", 320 / SAMPLE_RATE, 279.5 / SAMPLE_RATE)
CLOCKS = {clock.backbone: clock for clock in (MMS_CLOCK, W2V_BERT_CLOCK)}


REGION_LOSSES = ("constrained_alignment", "constrained_ctc")
_NEG = -1e4  # finite "impossible": logaddexp(-inf, -inf) has NaN gradients


def monotonic_alignment_loss(segments, letters):
    """-log of the summed probability of every blank-free monotonic path.

    Each frame of a region belongs to exactly one of its letters, in order,
    and every letter holds at least one frame; where boundaries fall is free.
    Unlike CTC there is no blank, so letters must fill the measured fragment
    instead of collapsing to one-frame spikes. Returns the sum over regions.
    """
    lengths = torch.tensor([len(x) for x in letters], device=segments[0].device)
    frames = torch.tensor([len(x) for x in segments], device=segments[0].device)
    log_probs = torch.nn.utils.rnn.pad_sequence(segments, batch_first=True)  # R, T, C
    targets = torch.nn.utils.rnn.pad_sequence(letters, batch_first=True)  # R, L
    emissions = log_probs.gather(2, targets.unsqueeze(1).expand(-1, log_probs.shape[1], -1))  # R, T, L
    alpha = torch.full_like(emissions[:, 0], _NEG)
    alpha[:, 0] = emissions[:, 0, 0]
    for t in range(1, emissions.shape[1]):
        advance = torch.cat([torch.full_like(alpha[:, :1], _NEG), alpha[:, :-1]], dim=1)
        step = torch.logaddexp(alpha, advance) + emissions[:, t]
        alpha = torch.where((t < frames).unsqueeze(1), step, alpha)
    return -alpha.gather(1, (lengths - 1).unsqueeze(1)).sum()


@dataclass
class AlignerOutput(ModelOutput):
    """CTC output plus the overlap head. During evaluation the overlap logits are summarised
    as ``overlap_stats`` (per window: overlap-frame tp/fp/fn and letter tp/fp/fn), so validation
    never accumulates a second full set of logits."""

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    overlap_logits: Optional[torch.FloatTensor] = None
    overlap_stats: Optional[torch.FloatTensor] = None
    stream2_logits: Optional[torch.FloatTensor] = None
    edge_logits: Optional[torch.FloatTensor] = None
    background_logits: Optional[torch.FloatTensor] = None


def monotonic_path_nll(segments, letters):
    """Per region: -log of the summed probability of every in-order, blank-free filling of the
    region's frames by its letters (as monotonic_alignment_loss, one value per region)."""
    lengths = torch.tensor([len(x) for x in letters], device=segments[0].device)
    frames = torch.tensor([len(x) for x in segments], device=segments[0].device)
    log_probs = torch.nn.utils.rnn.pad_sequence(segments, batch_first=True)
    targets = torch.nn.utils.rnn.pad_sequence(letters, batch_first=True)
    emissions = log_probs.gather(2, targets.unsqueeze(1).expand(-1, log_probs.shape[1], -1))
    alpha = torch.full_like(emissions[:, 0], _NEG)
    alpha[:, 0] = emissions[:, 0, 0]
    for t in range(1, emissions.shape[1]):
        advance = torch.cat([torch.full_like(alpha[:, :1], _NEG), alpha[:, :-1]], dim=1)
        step = torch.logaddexp(alpha, advance) + emissions[:, t]
        alpha = torch.where((t < frames).unsqueeze(1), step, alpha)
    return -alpha.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)


class ScaleGradient(torch.autograd.Function):
    """Identity forward; multiplies the gradient flowing back by ``scale``."""

    @staticmethod
    def forward(context, tensor, scale):
        """Return ``tensor`` unchanged, remembering ``scale`` for the backward pass."""
        context.scale = scale
        return tensor.view_as(tensor)

    @staticmethod
    def backward(context, gradient):
        """Scale the incoming gradient; ``scale`` itself gets none."""
        return gradient * context.scale, None


class AlignerMixin:
    """The aligner heads and losses, independent of the backbone. ``ForcedAligner`` (Wav2Vec2ForCTC:
    MMS, omniASR) and ``ForcedAlignerBert`` (Wav2Vec2BertForCTC: w2v-BERT 2.0) share everything here;
    each supplies ``backbone_forward`` and calls ``init_aligner_heads`` after the backbone's init."""

    z_loss_weight = 1e-4
    # How much of the overlap loss's gradient reaches the shared encoder (1: all of it). Lower
    # values let the overlap head learn while protecting the CTC head's timing.
    overlap_encoder_grad_scale = 1.0
    # "aligned" scores each word's letters in order over its frames (no guessed split);
    # "frame" uses per-frame multi-label targets.
    overlap_mode = "aligned"
    freeze_encoder = False
    # Weight of the overlap head's multi-label loss, and the extra weight of frames where two
    # or more words are sung at once (they are about 3% of sung time).
    overlap_loss_weight = 0.0
    overlap_frame_weight = 2.0
    # Second CTC stream for a word sung while another is (train.streams).
    stream2_loss_weight = 1.0
    stream2_blank_weight = 0.2  # stream 2 is silent most of the time
    stream2_encoder_grad_scale = 1.0
    # Background head: a second CTC-vocabulary head on the
    # shared encoder trained frame-wise on the manifest's ``background_tokens`` (backing vocals and
    # echoes, whole phrases, overlap with the lead allowed), blank elsewhere. Weight 0: no head.
    background_loss_weight = 0.0
    background_blank_weight = 0.2
    # Distillation: keep stream 1 close to a frozen teacher while the encoder learns new skills.
    distill_weight = 0.0
    # Edge head (as in SOFA): a 1-logit head on the encoder output trained on measured fragment
    # starts; BCE (positives weighted up) plus the cumulative-sum EMD term that makes a one-frame
    # miss cheap. Weight 0: no edge loss.
    edge_loss_weight = 0.0
    edge_positive_weight = 20.0
    edge_emd_weight = 1.0
    # Text-conditioned aligner head (config.text_aligner): frames scored against the whole song's
    # lyric tokens; loss = lead + text_back_weight x backing (text_aligner.voice_loss). Weight 0: no head.
    text_aligner_weight = 0.0
    text_fs_weight = 0.2
    text_back_weight = 0.5
    # Loss guidance flags (all off by default):
    # text_encoder_grad_scale: share of the text head's gradient that reaches the shared encoder/fusion.
    text_encoder_grad_scale = 1.0
    # loss_balance "ema": every term divided by its running mean (training only), then weighted.
    loss_balance = "none"
    alignment_weight = 1.0
    # background_balance: blank frames of the background head weighted n_sung / n_blank per batch (vs fixed).
    background_balance = False
    # text_fs_mode: "ctc" (forward-sum over token columns) or "region" (blank-free token filling of fragments).
    text_fs_mode = "ctc"
    region_loss = "constrained_alignment"
    # Focal weighting shrinks easy frames, and blank frames are nearly always
    # easy. Against an unweighted region loss that starves the blank signal and
    # letters spill into silence, so region-loss runs use plain cross-entropy.
    focal_frames = True
    # The loss below is already a per-frame mean. Without this, Trainer sees
    # forward(**kwargs), assumes the model normalizes across gradient
    # accumulation, and skips dividing by it: loss and gradients scale by GA.
    accepts_loss_kwargs = False

    def init_aligner_heads(self, config):
        """Create the optional heads the config asks for (overlap, stream 2, background, edge, MuQ fusion adapter and
        LoRA, text aligner); each is ``None`` when off. Fresh heads still need their ``warm_start_*`` call."""
        # Overlap head: one independent (sigmoid) output per vocabulary entry, so a frame can
        # carry the letters of every word sung there. Off unless the config asks for it.
        self.overlap_head = (
            torch.nn.Linear(self.lm_head.in_features, config.vocab_size)
            if getattr(config, "overlap_head", False)
            else None
        )
        # Stream 2: a second CTC head (softmax over the same vocabulary, same blank).
        self.stream2_head = (
            torch.nn.Linear(self.lm_head.in_features, config.vocab_size)
            if getattr(config, "stream2_head", False)
            else None
        )
        # Background head (the manifest's background_tokens). Off unless the config asks for it.
        self.background_head = (
            torch.nn.Linear(self.lm_head.in_features, config.vocab_size)
            if getattr(config, "background_head", False)
            else None
        )
        # Edge head: P(an acoustic token starts on this frame). Off unless the config asks for it.
        self.edge_head = torch.nn.Linear(self.lm_head.in_features, 1) if getattr(config, "edge_head", False) else None
        # MuQ fusion: the trained projection + gate (``fusion.*`` in the checkpoint). The frozen
        # MuQ encoder itself lives outside the module tree (``attach_fusion_backbone``).
        fusion = getattr(config, "fusion", None)
        self.fusion = (
            muq_fusion.FusionAdapter(
                self.lm_head.in_features, int(fusion.get("dim", muq_fusion.MUQ_DIM)), fusion.get("gate", "scalar")
            )
            if fusion
            else None
        )
        self._fusion_branch = None
        # Opt-in LoRA on the frozen MuQ branch (``config.fusion["lora"]``), saved as ``fusion_lora.*``.
        self.fusion_lora = muq_fusion.MuQLoRA(fusion["lora"]) if fusion and fusion.get("lora") else None
        text = getattr(config, "text_aligner", None)
        self.text_aligner = ta.TextAligner(ta.TextAlignerConfig(**text)) if text else None

    def attach_fusion_backbone(self, branch):
        """The frozen MuQ encoder (``muq_fusion.MuQBranch``): never trained, saved or moved by the
        Trainer; it follows the fusion input's device at forward time."""
        if self.fusion is None:
            raise RuntimeError("the checkpoint config has no fusion record")
        self._fusion_branch = branch
        if getattr(self, "fusion_lora", None) is not None:
            self.fusion_lora.attach(branch)

    def warm_start_fusion(self):
        """Fresh adapter: gate at zero, so the first step is exactly the plain model."""
        self.fusion.reset_parameters()

    def fusion_hidden(self, hidden, fusion_input_values, fusion_lengths):
        """The encoder's last hidden state plus the gated MuQ projection, on the model's clock."""
        if self.fusion.force_zero:
            return hidden
        if fusion_input_values is None:
            raise RuntimeError("fusion checkpoint: the batch carries no fusion_input_values (collator mismatch)")
        branch = self._fusion_branch
        if branch is not None and branch.device != fusion_input_values.device:
            branch.to(fusion_input_values.device)
        term = muq_fusion.fusion_term(
            self.fusion,
            branch,
            fusion_input_values,
            fusion_lengths,
            hidden.shape[1],
            int(self.config.fusion["layer"]),
            self.frame_clock.stride_seconds,
            self.frame_clock.offset_seconds,
        )
        return hidden if term is None else hidden + term.to(hidden.dtype)

    def set_teacher(self, teacher):
        """A frozen model whose CTC outputs stream 1 is pulled toward. Kept outside the module tree
        so it is never trained, saved or moved by the Trainer."""
        teacher.eval().requires_grad_(False)
        object.__setattr__(self, "_teacher", teacher)

    def distill_loss(self, logits, input_values, attention_mask):
        """Mean per-frame KL(teacher || stream 1) over the CTC vocabulary."""
        teacher = self.__dict__.get("_teacher")
        if teacher is None:
            return logits.sum() * 0
        with torch.no_grad():
            target = teacher(input_values=input_values, attention_mask=attention_mask).logits.float()
        frames = min(target.shape[1], logits.shape[1])
        target = F.log_softmax(target[:, :frames], dim=-1)
        student = F.log_softmax(logits[:, :frames].float(), dim=-1)
        return F.kl_div(student, target, log_target=True, reduction="none").sum(-1).mean()

    def warm_start_stream2_head(self, blank_bias: float = 5.0):
        """Start stream 2 from the CTC head but strongly biased to blank: it is silent from the first
        step, so training only teaches it the overlapping words instead of pulling the shared
        encoder toward silence (which drags stream 1 down with it)."""
        with torch.no_grad():
            self.stream2_head.weight.copy_(self.lm_head.weight)
            self.stream2_head.bias.copy_(self.lm_head.bias)
            self.stream2_head.bias[PAD_TOKEN_ID] += blank_bias

    def warm_start_background_head(self, blank_bias: float = 5.0):
        """A fresh background head starts as a copy of the CTC head biased to blank (silent at step 0,
        like stream 2), so it learns the backing letters without dragging the encoder to silence."""
        with torch.no_grad():
            self.background_head.weight.copy_(self.lm_head.weight)
            self.background_head.bias.copy_(self.lm_head.bias)
            self.background_head.bias[PAD_TOKEN_ID] += blank_bias

    def warm_start_edge_head(self, prior: float = 0.05):
        """Start the edge head at a flat prior (about one token start per 20 frames)."""
        with torch.no_grad():
            self.edge_head.weight.zero_()
            self.edge_head.bias.fill_(float(np.log(prior / (1 - prior))))

    def edge_loss(self, edge_logits, edge_targets):
        """Per-frame BCE on the edge logits (positives x ``edge_positive_weight``, -100 ungraded) plus
        ``edge_emd_weight`` x the L1 distance between the cumulative sums of the predicted and true
        edge curves, forward and backward (SOFA's BinaryEMDLoss), so that a boundary predicted one
        frame off costs little while a missing or doubled boundary costs its full mass."""
        logits = edge_logits.float()
        known = edge_targets != -100  # masks, overlaps and context are outside both terms
        graded = known & (edge_targets != -1)  # -1: tolerance neighbour of a start, BCE skips it
        target = (edge_targets == 1).float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        weights = torch.where(target > 0, self.edge_positive_weight, 1.0) * graded.float()
        loss = (bce * weights).sum() / weights.sum().clamp(min=1.0)
        if self.edge_emd_weight:
            probability = torch.sigmoid(logits) * known.float()
            forward = (probability.cumsum(1) - target.cumsum(1)).abs()
            backward = (probability.flip(1).cumsum(1) - target.flip(1).cumsum(1)).abs()
            frames = known.float().sum(1).clamp(min=1.0)
            emd = 0.5 * ((forward * known).sum(1) + (backward * known.flip(1)).sum(1)) / frames
            loss = loss + self.edge_emd_weight * emd.mean()
        return loss

    def warm_start_overlap_head(self):
        """Start the overlap head from the trained CTC head instead of from noise."""
        with torch.no_grad():
            self.overlap_head.weight.copy_(self.lm_head.weight)
            self.overlap_head.bias.copy_(self.lm_head.bias)

    def forward(
        self,
        input_values=None,
        attention_mask=None,
        labels=None,
        frame_targets=None,
        ctc_regions=None,
        ctc_targets=None,
        overlap_targets=None,
        overlap_graded=None,
        overlap_frames=None,
        ov_regions=None,
        ov_region_weights=None,
        ov_targets=None,
        ov_negative=None,
        ov_silence=None,
        ov_silence_graded=None,
        s2_labels=None,
        s2_frame_targets=None,
        s2_ctc_regions=None,
        s2_ctc_targets=None,
        edge_targets=None,
        input_features=None,
        fusion_input_values=None,
        fusion_lengths=None,
        bg_labels=None,
        bg_frame_targets=None,
        bg_ctc_regions=None,
        bg_ctc_targets=None,
        ta_letters=None,
        ta_voice=None,
        ta_line=None,
        ta_position=None,
        ta_text_padding=None,
        ta_frames=None,
        ta_lead_targets=None,
        ta_back_targets=None,
        ta_lead_sequences=None,
        ta_lead_lengths=None,
        ta_back_sequences=None,
        ta_back_lengths=None,
        ta_lead_regions=None,
        ta_back_regions=None,
        **kwargs,
    ):
        """Focal frame loss, plus a region loss inside each whole source fragment.

        ``labels`` are the weak per-frame letters, kept for metrics. The frame
        loss uses ``frame_targets`` when given: those exclude CTC regions and
        tolerate one blank frame at region edges. ``ctc_regions`` rows are
        (first_frame, stop_frame, target_length); padded rows have length 0.
        ``input_values`` is the waveform (wav2vec2); ``input_features`` the stacked
        fbank frames (wav2vec2-bert). ``fusion_input_values`` is the raw (unnormalised) 16 kHz
        waveform for the MuQ branch of a fusion checkpoint, ``fusion_lengths`` its valid samples.
        """
        # The overlap head reads the same features as the CTC head, captured on their way in, so
        # the plain path (no extra heads) is unchanged. A fusion checkpoint adds the gated
        # MuQ term to that hidden state first, so the CTC head and the edge head both see it.
        captured, hook = {}, None
        fusion = getattr(self, "fusion", None)

        def before_lm_head(_, args):
            hidden = args[0]
            if fusion is not None:
                hidden = self.fusion_hidden(hidden, fusion_input_values, fusion_lengths)
            captured["hidden"] = hidden
            return (hidden, *args[1:]) if fusion is not None else None

        background_head = getattr(self, "background_head", None)
        text_head = getattr(self, "text_aligner", None)
        if (
            self.overlap_head is not None
            or self.stream2_head is not None
            or self.edge_head is not None
            or fusion is not None
            or background_head is not None
            or text_head is not None
        ):
            hook = self.lm_head.register_forward_pre_hook(before_lm_head)
        try:
            output = self.backbone_forward(
                input_values=input_values, input_features=input_features, attention_mask=attention_mask, **kwargs
            )
        finally:
            if hook is not None:
                hook.remove()
        overlap_input = captured.get("hidden")
        if overlap_input is not None and self.training and self.overlap_encoder_grad_scale != 1.0:
            overlap_input = ScaleGradient.apply(overlap_input, self.overlap_encoder_grad_scale)
        overlap_logits = self.overlap_head(overlap_input) if self.overlap_head is not None else None
        stream2_input = captured.get("hidden")
        if stream2_input is not None and self.training and self.stream2_encoder_grad_scale != 1.0:
            stream2_input = ScaleGradient.apply(stream2_input, self.stream2_encoder_grad_scale)
        stream2_logits = self.stream2_head(stream2_input) if self.stream2_head is not None else None
        edge_logits = self.edge_head(captured["hidden"])[..., 0] if self.edge_head is not None else None
        background_logits = background_head(captured["hidden"]) if background_head is not None else None
        if labels is None:
            if overlap_logits is None and stream2_logits is None and edge_logits is None and background_logits is None:
                return output
            return AlignerOutput(
                logits=output.logits,
                overlap_logits=overlap_logits,
                stream2_logits=stream2_logits,
                edge_logits=edge_logits,
                background_logits=background_logits,
            )
        targets = labels if frame_targets is None else frame_targets
        cross_entropy = F.cross_entropy(output.logits.transpose(1, 2), targets, ignore_index=-100, reduction="none")
        graded = targets != -100
        valid = cross_entropy[graded]
        weights = (1 - torch.exp(-valid)).square() if self.focal_frames else 1.0
        total = (weights * valid).sum()
        count = valid.numel()
        # A separate mask: ``graded`` already indexed tensors autograd keeps.
        z_mask = graded.clone()
        if ctc_regions is not None and ctc_regions.numel():
            log_probs = F.log_softmax(output.logits.float(), dim=-1)
            segments, letters, input_lengths, target_lengths = [], [], [], []
            for batch, region in (ctc_regions[..., 2] > 0).nonzero().tolist():
                first, stop, length = ctc_regions[batch, region].tolist()
                segments.append(log_probs[batch, first:stop])
                letters.append(ctc_targets[batch, region, :length])
                input_lengths.append(stop - first)
                target_lengths.append(length)
                z_mask[batch, first:stop] = True
            if segments and self.region_loss == "constrained_ctc":
                # Letters may sit anywhere, in order; CTC also allows blank,
                # which in practice collapses letters to one-frame spikes.
                total = total + F.ctc_loss(
                    torch.nn.utils.rnn.pad_sequence(segments),
                    torch.cat(letters),
                    input_lengths,
                    target_lengths,
                    blank=PAD_TOKEN_ID,
                    reduction="sum",
                    zero_infinity=True,
                )
                count += sum(input_lengths)
            elif segments:
                # Letters fill their measured fragment, in order, boundaries free.
                total = total + monotonic_alignment_loss(segments, letters)
                count += sum(input_lengths)
        z_values = torch.logsumexp(output.logits, dim=-1).square()[z_mask]
        z_loss = self.z_loss_weight * z_values.sum() / max(z_values.numel(), 1)
        loss = total / max(count, 1) + z_loss
        # Unweighted loss components of this forward, for PartsTrainer's per-component logging.
        parts = self._loss_parts = {"alignment": loss.detach().float()}
        terms = [("alignment", self.alignment_weight, loss)]  # (name, weight, unweighted term) for loss_balance
        loss = self.alignment_weight * loss
        if edge_logits is not None and edge_targets is not None and self.edge_loss_weight:
            edge = self.edge_loss(edge_logits, edge_targets)
            loss = loss + self.edge_loss_weight * edge
            parts["edge"] = edge.detach().float()
            terms.append(("edge", self.edge_loss_weight, edge))
        if self.distill_weight:
            distill = self.distill_loss(output.logits, input_values, attention_mask)
            loss = loss + self.distill_weight * distill
            parts["distill"] = distill.detach().float()
        if text_head is not None and ta_letters is not None and self.text_aligner_weight:
            # The head reads the fused frame embeddings (gradients reach the encoder) and the CTC
            # letter log-probabilities (detached: an input feature, not a second path into the CTC head).
            hidden = captured["hidden"]
            if self.training and self.text_encoder_grad_scale != 1.0:
                hidden = ScaleGradient.apply(hidden, self.text_encoder_grad_scale)
            frame_padding = torch.arange(hidden.shape[1], device=hidden.device)[None, :] >= ta_frames[:, None]
            text_parts = text_head.forward_parts(
                hidden.float(),
                frame_padding,
                ta_letters,
                ta_voice,
                ta_text_padding,
                F.log_softmax(output.logits.float(), -1).detach(),
                ta_line,
                ta_position,
            )
            lead_loss, lead_stats = ta.voice_loss(
                text_parts[ta.LEAD],
                ta_lead_targets,
                ta_lead_sequences,
                ta_lead_lengths,
                ta_frames,
                self.text_fs_weight,
                fs_mode=self.text_fs_mode,
                regions=ta_lead_regions,
                region_loss=monotonic_alignment_loss,
            )
            back_loss, back_stats = ta.voice_loss(
                text_parts[ta.BACKING],
                ta_back_targets,
                ta_back_sequences,
                ta_back_lengths,
                ta_frames,
                self.text_fs_weight,
                fs_mode=self.text_fs_mode,
                regions=ta_back_regions,
                region_loss=monotonic_alignment_loss,
            )
            loss = loss + self.text_aligner_weight * (lead_loss + self.text_back_weight * back_loss)
            parts["text"] = (lead_loss + self.text_back_weight * back_loss).detach().float()
            terms.append(("text", self.text_aligner_weight, lead_loss + self.text_back_weight * back_loss))
            parts.update({f"text_lead_{k}": v.float() for k, v in lead_stats.items()})
            parts.update({f"text_back_{k}": v.float() for k, v in back_stats.items()})
        if background_logits is not None and bg_labels is not None and self.background_loss_weight:
            background_part = self.stream_loss(
                background_logits,
                bg_labels,
                bg_frame_targets,
                bg_ctc_regions,
                bg_ctc_targets,
                blank_weight="balanced" if self.background_balance else self.background_blank_weight,
            )
            loss = loss + self.background_loss_weight * background_part
            parts["background"] = background_part.detach().float()
            terms.append(("background", self.background_loss_weight, background_part))
            if self.overlap_head is None and stream2_logits is None:
                stats = None if self.training else background_statistics(background_logits, bg_labels)
                return AlignerOutput(loss=self.balanced(loss, terms), logits=output.logits, overlap_stats=stats)
        if stream2_logits is not None and s2_labels is not None:
            loss = loss + self.stream2_loss_weight * self.stream_loss(
                stream2_logits, s2_labels, s2_frame_targets, s2_ctc_regions, s2_ctc_targets
            )
            stats = None if self.training else stream2_statistics(stream2_logits, s2_labels)
            return AlignerOutput(loss=loss, logits=output.logits, overlap_stats=stats)
        if overlap_logits is None or (overlap_targets is None and ov_negative is None):
            return CausalLMOutput(
                loss=self.balanced(loss, terms),
                logits=output.logits,
                hidden_states=output.hidden_states,
                attentions=output.attentions,
            )
        if self.freeze_encoder:
            loss = loss.detach()  # head-only probe: the CTC loss is reported, not trained
        if ov_negative is not None:
            parts = self.aligned_overlap_loss(
                overlap_logits, ov_regions, ov_region_weights, ov_targets, ov_negative, ov_silence, ov_silence_graded
            )
            loss = loss + self.overlap_loss_weight * (parts[0] + parts[1] + parts[2])
            stats = None if self.training else torch.stack(parts).detach().float().expand(overlap_logits.shape[0], 3)
            return AlignerOutput(loss=loss, logits=output.logits, overlap_stats=stats)
        loss = loss + self.overlap_loss_weight * self.overlap_loss(
            overlap_logits, overlap_targets, overlap_graded, overlap_frames
        )
        stats = (
            None
            if self.training
            else overlap_statistics(overlap_logits, overlap_targets, overlap_graded, overlap_frames)
        )
        return AlignerOutput(loss=loss, logits=output.logits, overlap_stats=stats)

    def balanced(self, loss, terms):
        """``loss_balance="ema"``: sum of weight x term / running mean of the term (updated in training),
        so each term contributes by its weight instead of its raw scale; otherwise ``loss`` unchanged."""
        if self.loss_balance != "ema":
            return loss
        ema = self.__dict__.setdefault("_loss_ema", {})
        total = loss.new_zeros(())
        for name, weight, term in terms:
            value = float(term.detach())
            if self.training and math.isfinite(value):
                ema[name] = value if name not in ema else 0.99 * ema[name] + 0.01 * value
            total = total + weight * term / max(ema.get(name, value), 1e-3)
        return total

    def stream_loss(self, logits, labels, frame_targets, ctc_regions, ctc_targets, blank_weight=None):
        """Stream 2 / the background head: the main head's objective (plain frame cross-entropy outside
        fragments, in-order blank-free alignment inside each fragment, z-loss) with blank frames
        down-weighted (``blank_weight``, default the stream-2 weight)."""
        blank_weight = self.stream2_blank_weight if blank_weight is None else blank_weight
        targets = labels if frame_targets is None else frame_targets
        cross_entropy = F.cross_entropy(logits.float().transpose(1, 2), targets, ignore_index=-100, reduction="none")
        graded = targets != -100
        if blank_weight == "balanced":
            # --background-balance: blank frames together weigh as much as the sung ones (clamped).
            sung = (graded & (targets != PAD_TOKEN_ID)).sum().float()
            blank = (graded & (targets == PAD_TOKEN_ID)).sum().float().clamp_min(1.0)
            blank_weight = float((sung / blank).clamp(0.02, 1.0))
        weights = torch.where(targets == PAD_TOKEN_ID, blank_weight, 1.0)[graded]
        total = (weights * cross_entropy[graded]).sum()
        count = weights.sum()
        z_mask = graded.clone()
        if ctc_regions is not None and ctc_regions.numel():
            log_probs = F.log_softmax(logits.float(), dim=-1)
            segments, letters, lengths = [], [], []
            for batch, region in (ctc_regions[..., 2] > 0).nonzero().tolist():
                first, stop, length = ctc_regions[batch, region].tolist()
                segments.append(log_probs[batch, first:stop])
                letters.append(ctc_targets[batch, region, :length])
                lengths.append(stop - first)
                z_mask[batch, first:stop] = True
            if segments:
                total = total + monotonic_alignment_loss(segments, letters)
                count = count + sum(lengths)
        z_values = torch.logsumexp(logits.float(), dim=-1).square()[z_mask]
        z_loss = self.z_loss_weight * z_values.sum() / max(z_values.numel(), 1)
        return total / torch.clamp(torch.as_tensor(count, dtype=torch.float32, device=logits.device), min=1.0) + z_loss

    def aligned_overlap_loss(self, logits, regions, weights, targets, negative, silence, silence_graded):
        """Aligned mode: (word alignment NLL per frame, negative-letter BCE, silence BCE)."""
        logits = logits.float()
        log_on = F.logsigmoid(logits)
        segments, letters, region_weights, region_frames = [], [], [], []
        if regions is not None and regions.numel():
            for batch, index in (regions[..., 2] > 0).nonzero().tolist():
                first, stop, length = regions[batch, index].tolist()
                segments.append(log_on[batch, first:stop])
                letters.append(targets[batch, index, :length])
                region_weights.append(weights[batch, index])
                region_frames.append(stop - first)
        zero = logits.sum() * 0
        if segments:
            # Average over paths, not the sum: sigmoid outputs are not normalised across letters, so
            # a sum over the C(T-1, K-1) in-order fillings can exceed 1 and rewards switching every
            # letter of the word on everywhere. -log mean = -log sum + log C(T-1, K-1); best is 0.
            frames = torch.tensor(region_frames, device=logits.device, dtype=torch.float32)
            lengths = torch.tensor([len(x) for x in letters], device=logits.device, dtype=torch.float32)
            log_paths = torch.lgamma(frames) - torch.lgamma(lengths) - torch.lgamma(frames - lengths + 1)
            nll = monotonic_path_nll(segments, letters) + log_paths
            w = torch.stack(region_weights)
            positive = (w * nll).sum() / (w * frames).sum().clamp(min=1.0)
        else:
            positive = zero
        log_off = F.logsigmoid(-logits)
        count = negative.sum()
        negative_loss = -(log_off * negative).sum() / count.clamp(min=1) if count else zero
        silence_logits = logits[..., SILENCE_ID]
        graded = silence_graded.float()
        silence_loss = (
            F.binary_cross_entropy_with_logits(silence_logits, silence, reduction="none") * graded
        ).sum() / graded.sum().clamp(min=1.0)
        return positive, negative_loss, silence_loss

    def overlap_loss(self, logits, targets, graded, frames):
        """Masked multi-label binary cross-entropy, frames with simultaneous words weighted up."""
        losses = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="none")
        weights = graded.float() * torch.where(frames, self.overlap_frame_weight, 1.0).unsqueeze(-1)
        return (losses * weights).sum() / weights.sum().clamp(min=1.0)


class ForcedAligner(AlignerMixin, Wav2Vec2ForCTC):
    """MMS / omniASR backbone (waveform in): every released checkpoint."""

    frame_clock = MMS_CLOCK

    def __init__(self, config):
        super().__init__(config)
        self.init_aligner_heads(config)

    def backbone_forward(self, input_values=None, input_features=None, attention_mask=None, **kwargs):
        """The plain ``Wav2Vec2ForCTC`` forward on the waveform, without its CTC loss (``input_features`` unused)."""
        return Wav2Vec2ForCTC.forward(
            self, input_values=input_values, attention_mask=attention_mask, labels=None, **kwargs
        )


class ForcedAlignerBert(AlignerMixin, Wav2Vec2BertForCTC):
    """w2v-BERT 2.0 backbone (stacked fbank frames in). Same heads,
    losses and warm-ups as ForcedAligner; only the input and the frame clock differ."""

    frame_clock = W2V_BERT_CLOCK

    def __init__(self, config):
        super().__init__(config)
        self.init_aligner_heads(config)

    def backbone_forward(self, input_values=None, input_features=None, attention_mask=None, **kwargs):
        """The plain ``Wav2Vec2BertForCTC`` forward without its CTC loss; ``input_values`` stands in for
        ``input_features`` when only that is given."""
        features = input_features if input_features is not None else input_values
        return Wav2Vec2BertForCTC.forward(
            self, input_features=features, attention_mask=attention_mask, labels=None, **kwargs
        )


def aligner_class(model_dir: str):
    """The aligner class for a checkpoint directory, from its config's ``model_type``."""
    model_type = AutoConfig.from_pretrained(model_dir, local_files_only=True).model_type
    if model_type == "wav2vec2-bert":
        return ForcedAlignerBert
    if model_type == "wav2vec2":
        return ForcedAligner
    raise ValueError(f"unsupported backbone {model_type!r} in {model_dir}")


def load_aligner(model_dir: str, fusion: str = "on", fusion_model: str | None = None, **kwargs):
    """``from_pretrained`` with the class picked from the checkpoint (extra heads included).

    A fusion checkpoint (``config.fusion``) also needs its frozen MuQ branch: ``fusion="on"``
    loads it (``fusion_model`` overrides the checkpoint's model id/path), ``"gate0"`` skips it
    and forces the adapter to zero (the plain-model ablation), ``"off"`` refuses the checkpoint.
    Checkpoints without a fusion record ignore the argument."""
    model = aligner_class(model_dir).from_pretrained(model_dir, **kwargs)
    record = getattr(model.config, "fusion", None)
    if record:
        if fusion not in muq_fusion.FUSION_MODES:
            raise ValueError(f"fusion must be one of {muq_fusion.FUSION_MODES}")
        if fusion == "off":
            raise RuntimeError(f"{model_dir} is a fusion checkpoint; load it with fusion='on' (or 'gate0')")
        if fusion == "gate0":
            model.fusion.force_zero = True
        else:
            model.attach_fusion_backbone(muq_fusion.MuQBranch(fusion_model or record["model"]))
    return model


def stream2_statistics(logits, labels):
    """Per window: [stream-2 word frames heard (argmax not blank), stream-2 word frames,
    letters emitted where stream 2 should be silent, silent frames]."""
    with torch.no_grad():
        predicted = logits.argmax(-1) != PAD_TOKEN_ID
        words = (labels != -100) & (labels != PAD_TOKEN_ID)
        silent = labels == PAD_TOKEN_ID
        return torch.stack(
            [(predicted & words).sum(1), words.sum(1), (predicted & silent).sum(1), silent.sum(1)], dim=1
        ).float()


def background_statistics(logits, labels):
    """Per window: stream2_statistics' four counts plus a marker column (5 wide, so ``metrics``
    reports them as background_* instead of stream2_*)."""
    counts = stream2_statistics(logits, labels)
    return torch.cat([counts, torch.ones_like(counts[:, :1])], dim=1)


def overlap_statistics(logits, targets, graded, frames):
    """Per window: [overlap frames tp, fp, fn, letter tp, fp, fn] at probability 0.5."""
    with torch.no_grad():
        letters = list(LETTERS)
        predicted = torch.sigmoid(logits.float()) > 0.5
        valid = graded[..., letters].any(-1)
        predicted_overlap = (predicted[..., letters].sum(-1) >= 2) & valid
        actual = frames & valid
        lp, lt, lg = predicted[..., letters], targets[..., letters] > 0.5, graded[..., letters]
        return torch.stack(
            [
                (predicted_overlap & actual).sum(1),
                (predicted_overlap & ~actual).sum(1),
                (~predicted_overlap & actual).sum(1),
                (lp & lt & lg).sum((1, 2)),
                (lp & ~lt & lg).sum((1, 2)),
                (~lp & lt & lg).sum((1, 2)),
            ],
            dim=1,
        ).float()


def decode_window(path: Path, start: float, end: float, filters: list[str] | None = None) -> np.ndarray:
    """Decode only the requested AAC window. No decoded file is ever written.

    ``filters`` (augmentation) run in the same FFmpeg call. Then the window length limits the
    input, so a tempo change shortens or lengthens the output instead of reading other audio.
    """
    duration = max(0.1, end - start)
    if filters:
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-af",
            ",".join([f"aresample={SAMPLE_RATE}", *filters, f"aresample={SAMPLE_RATE}"]),
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "f32le",
            "pipe:1",
        ]
    else:
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(path),
            "-t",
            f"{duration:.3f}",
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
    waveform = np.frombuffer(result.stdout, dtype=np.float32).copy()
    # MMS needs a 400-sample receptive field, not half a second. Short
    # nonempty tail windows are valid training examples. Keep dataset ordering
    # unchanged so existing checkpoints can resume against the same windows.
    if waveform.size < 400:
        raise RuntimeError(
            f"decoded {waveform.size} samples from {path} for window {start:.3f}:{end:.3f}; "
            "MMS requires at least 400 samples"
        )
    return waveform


def region(token: dict, offset: float) -> dict:
    """The token's measured source fragment, relative to the window."""
    spans = token.get("source_spans") or []
    if token.get("ignore") or len(spans) != 1:
        return {}
    return {"region_start": spans[0]["start"] - offset, "region_end": spans[0]["end"] - offset}


WINDOW_SNAP = {"min": 20.0, "max": 27.0}  # --window-policy snap: window length range (seconds)


def snapped_window_end(
    tokens: list[dict], start: float, duration: float, window_seconds: float, snap: dict | None
) -> float:
    """End of the window starting at ``start``. Fixed policy: ``start + window_seconds``. Snap policy:
    end inside [start+min, start+max] in the middle of a gap between two
    lead fragments, preferring a line change, then the widest gap, so no fragment is cut by the window
    edge (windows are back to back: a cut fragment was never seen whole). No gap in range: fixed cut."""
    fixed = min(duration, start + window_seconds)
    if not snap or start + snap["max"] >= duration:
        return min(duration, start + snap["max"]) if snap else fixed
    low, high = start + snap["min"], start + snap["max"]
    best = None
    for left, right in zip(tokens, tokens[1:]):
        gap_start, gap_end = left["end"], right["start"]
        if gap_end < low or gap_start > high or gap_end < gap_start:
            continue
        middle = min(max((gap_start + gap_end) / 2, low), high)
        if not (gap_start <= middle <= gap_end):
            continue  # the usable part of the gap lies outside the window range
        score = (left.get("line") != right.get("line"), gap_end - gap_start)
        if best is None or score > best[0]:
            best = (score, middle)
    return best[1] if best else min(duration, high)


def make_windows(
    rows: list[dict],
    window_seconds: float,
    background: bool = False,
    snap: dict | None = None,
    mask_unlabelled: bool = False,
) -> list[dict]:
    """Windows over the lead tokens. ``background``: also carry each window's background-head tokens
    (schema-7 ``background_tokens``) and whether the song has any (``background_graded``: a song
    without background tags is not graded on the background head, it was never annotated).
    ``mask_unlabelled`` (--mask-unlabelled-vocals): the row's ``unlabelled_vocal_spans`` (voiced in the
    separated vocal stem, no label) become ignore tokens, so they are not taught as "not singing"."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    windows = []
    for row in rows:
        tokens = sorted(row["alignment_tokens"], key=lambda token: token["start"])
        if not tokens:
            continue
        start = max(0.0, tokens[0]["start"] - 0.5)
        stop = min(float(row["duration"]), max(t["end"] for t in tokens) + 0.5)
        while start < stop:
            end = snapped_window_end(tokens, start, float(row["duration"]), window_seconds, snap)
            selected = [
                {**t, "start": t["start"] - start, "end": t["end"] - start, **region(t, start)}
                for t in tokens
                if t["start"] < end and t["end"] > start
            ]
            if selected and mask_unlabelled:
                selected += [
                    {
                        "start": max(a, start) - start,
                        "end": min(b, end) - start,
                        "ids": [],
                        "ignore": True,
                        "unlabelled_mask": True,
                    }
                    for a, b in row.get("unlabelled_vocal_spans") or []
                    if a < end and b > start
                ]
            if selected:
                window = {
                    "audio": row["audio"],
                    "start": start,
                    "end": end,
                    "tokens": selected,
                    "id": row["id"],
                    "language": row["language"],
                }
                if background:
                    window["background_graded"] = bool(row.get("background_tokens"))
                    window["background"] = [
                        {**t, "start": t["start"] - start, "end": t["end"] - start, **region(t, start)}
                        for t in row.get("background_tokens") or []
                        if t["start"] < end and t["end"] > start
                    ]
                windows.append(window)
            start = end
    return windows


def windows_from_manifest_line(
    line: str, window_seconds: float, background: bool = False, snap: dict | None = None, mask_unlabelled: bool = False
) -> list[dict]:
    """``make_windows`` for one JSON manifest line (picklable for the indexing pool)."""
    return make_windows([json.loads(line)], window_seconds, background, snap, mask_unlabelled)


def song_text_record(
    line: str, max_letters: int = 16, pad: int = 32, version: int = CURRENT_VERSION
) -> tuple[str, dict] | None:
    """One song's lyric tokens (lead + backing, text order = line, then time) as compact arrays:
    letters [N, max_letters] (pad = ``pad``), voice / line / position [N], start / end [N] (seconds)."""
    row = json.loads(line)
    tokens = canonical_tokens(row, version)
    back = [t for t, voice in tokens if voice == 1]
    if not tokens:
        return None
    features = text_features(tokens, max_letters, pad)
    return row["id"], {
        "letters": np.array(features["letters"], dtype=np.int16),
        "voice": np.array(features["voice"], dtype=np.int8),
        "line": np.array(features["line"], dtype=np.int32),
        "position": np.array(features["position"], dtype=np.int32),
        "start": np.array([t["start"] for t, _ in tokens], dtype=np.float32),
        "end": np.array([t["end"] for t, _ in tokens], dtype=np.float32),
        "graded_background": bool(back),
    }


def build_song_text(
    manifests: list[Path], workers: int, max_letters: int, pad: int, version: int = CURRENT_VERSION
) -> dict:
    """Song id -> ``song_text_record`` arrays for every song with lyric tokens in ``manifests``, in a process pool."""
    store = {}
    for manifest in manifests:
        with manifest.open() as source, mp.Pool(workers) as pool:
            for item in pool.imap(
                partial(song_text_record, max_letters=max_letters, pad=pad, version=version), source, chunksize=16
            ):
                if item is not None:
                    store[item[0]] = item[1]
    return store


def text_frame_targets(song: dict, voice: int, decode_start: float, frames: int, clock: "FrameClock"):
    """(per-frame columns [frames] with 0 = none and 1 + j = token j, forward-sum columns in time order)."""
    times = decode_start + clock.offset_seconds + np.arange(frames) * clock.stride_seconds
    targets = np.zeros(frames, dtype=np.int64)
    window_end = decode_start + frames * clock.stride_seconds
    chosen = np.nonzero((song["voice"] == voice) & (song["end"] > decode_start) & (song["start"] < window_end))[0]
    sequence = []
    for j in chosen:
        first, stop = np.searchsorted(times, song["start"][j]), np.searchsorted(times, song["end"][j])
        if stop > first:
            targets[first:stop] = 1 + j
            sequence.append((song["start"][j], 1 + j))
    return targets, [column for _, column in sorted(sequence)]


def text_regions(song: dict, voice: int, decode_start: float, frames: int, clock: "FrameClock", gap: float = 0.02):
    """--text-fs-mode region: groups of a voice's tokens sung back to back (gaps <= ``gap`` s) as
    (first frame, stop frame, [columns in time order]) inside the window, for the blank-free token filling."""
    times = decode_start + clock.offset_seconds + np.arange(frames) * clock.stride_seconds
    window_end = decode_start + frames * clock.stride_seconds
    chosen = sorted(
        np.nonzero((song["voice"] == voice) & (song["end"] > decode_start) & (song["start"] < window_end))[0],
        key=lambda j: song["start"][j],
    )
    groups, current = [], []
    for j in chosen:
        if current and song["start"][j] - song["end"][current[-1]] > gap:
            groups.append(current)
            current = []
        current.append(j)
    if current:
        groups.append(current)
    regions = []
    for group in groups:
        first = int(np.searchsorted(times, song["start"][group[0]]))
        stop = int(np.searchsorted(times, song["end"][group[-1]]))
        if stop - first >= len(group) and first >= 0 and stop <= frames:
            regions.append((first, stop, [1 + int(j) for j in group]))
    return regions


class TimedWindowDataset(Dataset):
    """Training or validation windows of a manifest, decoded from the song's audio on every access.

    Indexing (``make_windows``) runs over the whole manifest in ``index_workers`` processes; with ``limit`` it reads
    lines sequentially and stops after ``limit`` windows. ``windows_per_track`` keeps only each song's first windows.
    """

    def __init__(
        self,
        root: Path,
        manifest: Path,
        window_seconds: float,
        limit: int | None = None,
        windows_per_track: int | None = None,
        index_workers: int = 8,
        augment: "augmentation.AugmentConfig | None" = None,
        background: bool = False,
        snap: dict | None = None,
        mask_unlabelled: bool = False,
    ):
        self.root, self.windows, self.augment = root, [], augment
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        if windows_per_track is not None and windows_per_track <= 0:
            raise ValueError("windows_per_track must be positive")
        # Smoke/evaluation stops early. Full training expands the complete
        # manifest with CPU workers instead of holding one core for 20+ minutes.
        if limit is not None:
            with manifest.open() as source:
                for line in source:
                    if not line:
                        continue
                    track_windows = windows_from_manifest_line(line, window_seconds, background, snap, mask_unlabelled)
                    if windows_per_track is not None:
                        track_windows = track_windows[:windows_per_track]
                    self.windows.extend(track_windows)
                    if len(self.windows) >= limit:
                        self.windows = self.windows[:limit]
                        break
        else:
            with manifest.open() as source, mp.Pool(index_workers) as pool:
                for track_windows in pool.imap(
                    partial(
                        windows_from_manifest_line,
                        window_seconds=window_seconds,
                        background=background,
                        snap=snap,
                        mask_unlabelled=mask_unlabelled,
                    ),
                    source,
                    chunksize=8,
                ):
                    if windows_per_track is not None:
                        track_windows = track_windows[:windows_per_track]
                    self.windows.extend(track_windows)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        """One window as ``input_values`` (float32 16 kHz waveform), window-relative ``tokens`` and metadata.

        A window shorter than 1 s is decoded from up to 1 s before its end, with the borrowed context masked by an
        ignore token; ``decode_start`` is the absolute start of the decoded audio in seconds. With augmentation,
        the waveform is cut to the tempo-scaled window length and token times are scaled to match.
        """
        item = self.windows[index]
        # Supply real preceding context for short tails without removing or
        # reordering examples. Mask context whose labels are not in this window.
        decode_start = min(item["start"], max(0.0, item["end"] - 1.0))
        shift = item["start"] - decode_start
        tokens = item["tokens"]
        background = item.get("background")
        if shift > 0:
            tokens = [
                {
                    **token,
                    **{
                        key: token[key] + shift
                        for key in ("start", "end", "region_start", "region_end")
                        if key in token
                    },
                }
                for token in tokens
            ]
            tokens.append({"start": 0.0, "end": shift, "ids": [], "ignore": True})
            if background is not None:
                background = [
                    {
                        **token,
                        **{
                            key: token[key] + shift
                            for key in ("start", "end", "region_start", "region_end")
                            if key in token
                        },
                    }
                    for token in background
                ]
                background.append({"start": 0.0, "end": shift, "ids": [], "ignore": True})
        if self.augment is None:
            waveform = decode_window(self.root / item["audio"], decode_start, item["end"])
        else:
            rng = random.Random()  # seeded from the OS: every visit of a window gets new effects
            chosen = augmentation.plan(self.augment, rng)
            waveform = augmentation.finish(
                decode_window(self.root / item["audio"], decode_start, item["end"], chosen.filters), chosen, rng
            )
            # AAC frames can add a few tens of milliseconds at the end; keep exactly the window.
            waveform = waveform[: round((item["end"] - decode_start) / chosen.tempo * SAMPLE_RATE)]
            tokens = augmentation.scale_tokens(tokens, chosen.tempo)
            if background is not None:
                background = augmentation.scale_tokens(background, chosen.tempo)
        example = {
            "input_values": waveform,
            "tokens": tokens,
            "duration": len(waveform) / SAMPLE_RATE,
            "song_id": item["id"],
            "decode_start": decode_start,
            "augmented": self.augment is not None,
        }
        if background is not None:
            example["background_tokens"] = background
            example["background_graded"] = item.get("background_graded", False)
        return example


def edge_frame_targets(
    tokens: list[dict], labels: list[int], tolerance_frames: int = 1, clock: FrameClock = MMS_CLOCK
) -> list[int]:
    """Edge-head targets: 1 on the frame whose receptive-field centre is first at/after each measured
    fragment start (``region_start``, as ``ctc_regions`` places regions), 0 elsewhere, -1 within
    ``tolerance_frames`` of a positive (the BCE skips it, the EMD still counts a prediction there)
    and -100 on frames the frame loss ignores (masks, overlaps, context). Fragments cut by the
    window start (``region_start`` < 0) have no measured start here."""
    from echora_kara.train.targets import frame_index

    frames = len(labels)
    targets = [0 if label != -100 else -100 for label in labels]
    starts = sorted(
        {
            token["region_start"]
            for token in tokens
            if not token.get("ignore") and "region_start" in token and token["region_start"] >= 0
        }
    )
    positives = []
    for start in starts:
        frame = frame_index(start, clock.stride_seconds, clock.offset_seconds)
        if 0 <= frame < frames and labels[frame] != -100:
            positives.append(frame)
    for frame in positives:
        for near in range(frame - tolerance_frames, frame + tolerance_frames + 1):
            if 0 <= near < frames and targets[near] == 0:
                targets[near] = -1
    for frame in positives:
        targets[frame] = 1
    return targets


@dataclass
class AlignmentCollator:
    """Pads a batch of windows and builds every target the model's enabled heads need, on the backbone's frame clock.

    Frame targets are padded with -100 to the longest window (or the padded w2v-BERT feature length); region tensors
    are ``[batch, regions, 3]`` rows of (first_frame, stop_frame, length) with matching padded letter ids.
    """

    processor: Wav2Vec2Processor
    model: ForcedAligner
    loss: str = "frame"
    boundary_tolerance_frames: int = 1
    overlap_token_weight: float = 4.0
    stream_min_overlap: float = 0.1  # streams.MIN_OVERLAP
    edge_tolerance_frames: int = 1  # edge head: ungraded BCE frames on each side of a fragment start
    # Frame clock of the backbone (MMS by default).
    clock: FrameClock = MMS_CLOCK
    # "proportional" (default): a fragment without a region is graded with the
    # proportional per-letter frame labels; "ignore": its frames are ignored.
    region_fallback: str = "proportional"
    # Song id -> build_song_text arrays (the whole song's tokens for the text-aligner head).
    song_text: Optional[dict] = None

    def text_aligner_batch(self, examples, labels, max_length):
        """Tensors of the text-aligner head: the whole song's tokens per window and its frame / forward-sum targets."""
        config = self.model.text_aligner.config
        songs = [self.song_text[x["song_id"]] for x in examples]
        most = max(1, max(len(s["voice"]) for s in songs))
        batch = {
            "ta_letters": torch.full((len(examples), most, config.max_letters), config.letters, dtype=torch.long),
            "ta_voice": torch.zeros((len(examples), most), dtype=torch.long),
            "ta_line": torch.zeros((len(examples), most), dtype=torch.long),
            "ta_position": torch.zeros((len(examples), most), dtype=torch.long),
            "ta_text_padding": torch.ones((len(examples), most), dtype=torch.bool),
            "ta_frames": torch.tensor([len(x) for x in labels], dtype=torch.long),
        }
        sequences = {"lead": [], "back": []}
        targets = {"lead": [], "back": []}
        regions = {"lead": [], "back": []}
        region_mode = getattr(self.model, "text_fs_mode", "ctc") == "region"
        for row, (song, example, frame_list) in enumerate(zip(songs, examples, labels)):
            if example.get("augmented"):
                raise RuntimeError("the text-aligner head does not support --augment (tempo changes its targets)")
            n = len(song["voice"])
            batch["ta_letters"][row, :n] = torch.from_numpy(song["letters"].astype(np.int64))
            batch["ta_voice"][row, :n] = torch.from_numpy(song["voice"].astype(np.int64))
            batch["ta_line"][row, :n] = torch.from_numpy(song["line"].astype(np.int64))
            batch["ta_position"][row, :n] = torch.from_numpy(song["position"].astype(np.int64))
            batch["ta_text_padding"][row, :n] = False
            frames = len(frame_list)
            for key, voice in (("lead", ta.LEAD), ("back", ta.BACKING)):
                if voice == ta.BACKING and not song["graded_background"]:
                    found, sequence = np.full(frames, -100, dtype=np.int64), []
                    found_regions = []
                else:
                    found, sequence = text_frame_targets(song, voice, example["decode_start"], frames, self.clock)
                    if voice == ta.LEAD:  # --mask-unlabelled-vocals: no "not singing" target on unlabelled vocals
                        times = self.clock.offset_seconds + np.arange(frames) * self.clock.stride_seconds
                        for token in example["tokens"]:
                            if token.get("unlabelled_mask"):
                                first, stop = (
                                    np.searchsorted(times, token["start"]),
                                    np.searchsorted(times, token["end"]),
                                )
                                found[first:stop] = np.where(found[first:stop] == 0, -100, found[first:stop])
                    found_regions = (
                        text_regions(song, voice, example["decode_start"], frames, self.clock) if region_mode else []
                    )
                targets[key].append(np.concatenate((found, np.full(max_length - frames, -100, dtype=np.int64))))
                sequences[key].append(sequence)
                regions[key].append(found_regions)
        for key in ("lead", "back"):
            batch[f"ta_{key}_targets"] = torch.from_numpy(np.stack(targets[key]))
            longest = max(1, max(len(x) for x in sequences[key]))
            padded = torch.zeros((len(examples), longest), dtype=torch.long)
            for row, sequence in enumerate(sequences[key]):
                if sequence:
                    padded[row, : len(sequence)] = torch.tensor(sequence)
            batch[f"ta_{key}_sequences"] = padded
            batch[f"ta_{key}_lengths"] = torch.tensor([len(x) for x in sequences[key]], dtype=torch.long)
            if region_mode:
                most = max(1, max(len(x) for x in regions[key]))
                longest = max([1, *(len(cols) for x in regions[key] for *_, cols in x)])
                packed = torch.zeros((len(examples), most, 3 + longest), dtype=torch.long)
                for row, found_regions in enumerate(regions[key]):
                    for column, (first, stop, cols) in enumerate(found_regions):
                        packed[row, column, :3] = torch.tensor([first, stop, len(cols)])
                        packed[row, column, 3 : 3 + len(cols)] = torch.tensor(cols)
                batch[f"ta_{key}_regions"] = packed
        return batch

    def background_targets(self, example, frames):
        """(frame labels, frame-loss targets, regions) of the background head for one window: each
        background letter held over its fragment (blank-free region loss inside measured
        fragments, as the lead), blank elsewhere; all ignored for a song without background tags."""
        if not example.get("background_graded"):
            return [-100] * frames, [-100] * frames, []
        tokens = example.get("background_tokens") or []
        labels = (
            frame_labels(tokens, example["duration"], frames, self.clock.stride_seconds, self.clock.offset_seconds)
            if tokens
            else [PAD_TOKEN_ID] * frames
        )
        found = (
            find_ctc_regions(
                tokens,
                labels,
                example["duration"],
                self.clock.stride_seconds,
                self.clock.offset_seconds,
                repeat_blanks=False,
            )
            if tokens
            else []
        )
        if self.region_fallback == "ignore" and tokens:
            labels = ignore_unregioned(tokens, labels, found, self.clock.stride_seconds, self.clock.offset_seconds)
        return labels, tolerant_frame_targets(labels, found, self.boundary_tolerance_frames), found

    def __call__(self, examples):
        """The processor's padded inputs plus ``labels`` and, as enabled, region-loss, stream-2, background, overlap,
        edge, text-aligner and raw MuQ fusion tensors. Raises if the model's frame clock differs from the collator's."""
        model_clock = getattr(self.model, "frame_clock", self.clock)
        if model_clock.backbone != self.clock.backbone:
            raise RuntimeError(f"collator clock {self.clock.backbone} but the model is {model_clock.backbone}")
        batch = self.processor(
            [x["input_values"] for x in examples],
            sampling_rate=SAMPLE_RATE,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        bert = self.clock.backbone == "wav2vec2-bert"
        labels, targets, regions = [], [], []
        two_streams = getattr(self.model, "stream2_head", None) is not None
        stream2 = []
        background_head = getattr(self.model, "background_head", None) is not None
        background = []
        for row, example in enumerate(examples):
            frames = self.clock.frames(len(example["input_values"]), self.model)
            if bert:
                # The SeamlessM4T extractor stacks fbank pairs; its attention mask marks the valid
                # model frames, which must be exactly the frames the labels are placed on.
                valid = int(batch["attention_mask"][row].sum())
                if valid != frames or frames < 1:
                    raise RuntimeError(f"w2v-BERT frame clock mismatch: {valid} valid frames, {frames} expected")
            tokens = example["tokens"]
            if two_streams:
                tokens, second = assign_streams(tokens, self.stream_min_overlap)
                s2_labels = frame_labels(second, example["duration"], frames)
                s2_found = find_ctc_regions(second, s2_labels, example["duration"], repeat_blanks=False)
                stream2.append(
                    (s2_labels, tolerant_frame_targets(s2_labels, s2_found, self.boundary_tolerance_frames), s2_found)
                )
            if background_head:
                background.append(self.background_targets(example, frames))
            labels.append(
                frame_labels(tokens, example["duration"], frames, self.clock.stride_seconds, self.clock.offset_seconds)
            )
            found = (
                find_ctc_regions(
                    tokens,
                    labels[-1],
                    example["duration"],
                    self.clock.stride_seconds,
                    self.clock.offset_seconds,
                    repeat_blanks=self.loss == "constrained_ctc",
                )
                if self.loss in REGION_LOSSES
                else []
            )
            regions.append(found)
            if self.loss in REGION_LOSSES and self.region_fallback == "ignore":
                labels[-1] = ignore_unregioned(
                    tokens, labels[-1], found, self.clock.stride_seconds, self.clock.offset_seconds
                )
            targets.append(
                tolerant_frame_targets(labels[-1], found, self.boundary_tolerance_frames)
                if self.loss in REGION_LOSSES
                else labels[-1]
            )
        max_length = max(map(len, labels))
        if bert:
            # Logits cover the padded feature frames (pairs are padded to a multiple of 2): the
            # extra frames carry no label.
            max_length = max(max_length, int(batch["input_features"].shape[1]))
        if stream2:
            pad = lambda rows: torch.tensor([x + [-100] * (max_length - len(x)) for x in rows], dtype=torch.long)  # noqa: E731
            batch["s2_labels"] = pad([x[0] for x in stream2])
            batch["s2_frame_targets"] = pad([x[1] for x in stream2])
            most = max(1, max(len(x[2]) for x in stream2))
            longest = max([1, *(len(ids) for x in stream2 for *_, ids in x[2])])
            batch["s2_ctc_regions"] = torch.zeros((len(examples), most, 3), dtype=torch.long)
            batch["s2_ctc_targets"] = torch.zeros((len(examples), most, longest), dtype=torch.long)
            for row, (_, _, found) in enumerate(stream2):
                for column, (first, stop, ids) in enumerate(found):
                    batch["s2_ctc_regions"][row, column] = torch.tensor([first, stop, len(ids)])
                    batch["s2_ctc_targets"][row, column, : len(ids)] = torch.tensor(ids)
        if background:
            pad = lambda rows: torch.tensor([x + [-100] * (max_length - len(x)) for x in rows], dtype=torch.long)  # noqa: E731
            batch["bg_labels"] = pad([x[0] for x in background])
            batch["bg_frame_targets"] = pad([x[1] for x in background])
            most = max(1, max(len(x[2]) for x in background))
            longest = max([1, *(len(ids) for x in background for *_, ids in x[2])])
            batch["bg_ctc_regions"] = torch.zeros((len(examples), most, 3), dtype=torch.long)
            batch["bg_ctc_targets"] = torch.zeros((len(examples), most, longest), dtype=torch.long)
            for row, (_, _, found) in enumerate(background):
                for column, (first, stop, ids) in enumerate(found):
                    batch["bg_ctc_regions"][row, column] = torch.tensor([first, stop, len(ids)])
                    batch["bg_ctc_targets"][row, column, : len(ids)] = torch.tensor(ids)
        if getattr(self.model, "overlap_head", None) is not None and self.model.overlap_mode == "aligned":
            vocab = self.model.config.vocab_size
            found = [
                aligned_overlap_targets(
                    example["tokens"], len(frame_list), vocab, self.boundary_tolerance_frames, self.overlap_token_weight
                )
                for example, frame_list in zip(examples, labels)
            ]
            most = max(1, max(len(f["regions"]) for f in found))
            longest = max([1, *(len(ids) for f in found for _, _, ids, _ in f["regions"])])
            batch["ov_regions"] = torch.zeros((len(examples), most, 3), dtype=torch.long)
            batch["ov_region_weights"] = torch.zeros((len(examples), most))
            batch["ov_targets"] = torch.zeros((len(examples), most, longest), dtype=torch.long)
            batch["ov_negative"] = torch.zeros((len(examples), max_length, vocab), dtype=torch.bool)
            batch["ov_silence"] = torch.zeros((len(examples), max_length))
            batch["ov_silence_graded"] = torch.zeros((len(examples), max_length), dtype=torch.bool)
            for row, (f, frame_list) in enumerate(zip(found, labels)):
                n = len(frame_list)
                for column, (first, stop, ids, weight) in enumerate(f["regions"]):
                    batch["ov_regions"][row, column] = torch.tensor([first, stop, len(ids)])
                    batch["ov_region_weights"][row, column] = weight
                    batch["ov_targets"][row, column, : len(ids)] = torch.tensor(ids)
                batch["ov_negative"][row, :n] = torch.from_numpy(f["negative"])
                batch["ov_silence"][row, :n] = torch.from_numpy(f["silence"])
                batch["ov_silence_graded"][row, :n] = torch.from_numpy(f["silence_graded"])
        elif getattr(self.model, "overlap_head", None) is not None:
            vocab = self.model.config.vocab_size
            batch["overlap_targets"] = torch.zeros((len(examples), max_length, vocab))
            batch["overlap_graded"] = torch.zeros((len(examples), max_length, vocab), dtype=torch.bool)
            batch["overlap_frames"] = torch.zeros((len(examples), max_length), dtype=torch.bool)
            for row, (example, frame_list) in enumerate(zip(examples, labels)):
                found_targets, graded, simultaneous = multilabel_targets(
                    example["tokens"], len(frame_list), vocab, self.boundary_tolerance_frames
                )
                batch["overlap_targets"][row, : len(frame_list)] = torch.from_numpy(found_targets)
                batch["overlap_graded"][row, : len(frame_list)] = torch.from_numpy(graded)
                batch["overlap_frames"][row, : len(frame_list)] = torch.from_numpy(simultaneous)
        batch["labels"] = torch.tensor([x + [-100] * (max_length - len(x)) for x in labels], dtype=torch.long)
        if getattr(self.model, "text_aligner", None) is not None and self.song_text is not None:
            batch.update(self.text_aligner_batch(examples, labels, max_length))
        if getattr(self.model, "fusion", None) is not None:
            # The MuQ branch hears the raw window (the processor's normalisation is for the
            # wav2vec2 front end only); the model resamples it to 24 kHz on its own device.
            raw = [torch.as_tensor(np.asarray(x["input_values"], dtype=np.float32)) for x in examples]
            batch["fusion_input_values"] = torch.nn.utils.rnn.pad_sequence(raw, batch_first=True)
            batch["fusion_lengths"] = torch.tensor([len(x) for x in raw], dtype=torch.long)
        if getattr(self.model, "edge_head", None) is not None:
            edges = [
                edge_frame_targets(example["tokens"], frame_list, self.edge_tolerance_frames, self.clock)
                for example, frame_list in zip(examples, labels)
            ]
            batch["edge_targets"] = torch.tensor([x + [-100] * (max_length - len(x)) for x in edges], dtype=torch.long)
        if self.loss in REGION_LOSSES:
            batch["frame_targets"] = torch.tensor(
                [x + [-100] * (max_length - len(x)) for x in targets], dtype=torch.long
            )
            most_regions = max(1, max(map(len, regions)))
            longest = max([1, *(len(ids) for found in regions for *_, ids in found)])
            batch["ctc_regions"] = torch.zeros((len(examples), most_regions, 3), dtype=torch.long)
            batch["ctc_targets"] = torch.zeros((len(examples), most_regions, longest), dtype=torch.long)
            for row, found in enumerate(regions):
                for column, (first, stop, ids) in enumerate(found):
                    batch["ctc_regions"][row, column] = torch.tensor([first, stop, len(ids)])
                    batch["ctc_targets"][row, column, : len(ids)] = torch.tensor(ids)
        return batch


# The "head" group of the warm-up: the CTC head, the edge head, the fusion adapter and the other heads.
HEAD_PREFIXES = ("lm_head", "edge_head", "fusion", "background_head", "text_aligner")


class CompletedSave(TrainerCallback):
    """After each save on the main process, write the checkpoint's completion marker and prune old optimizer states."""

    def on_save(self, args, state, control, **kwargs):
        """Mark the checkpoint just written as complete (main process only)."""
        if state.is_world_process_zero:
            mark_and_prune(Path(args.output_dir) / f"checkpoint-{state.global_step}")


class DivergenceGuard(TrainerCallback):
    """Stops training when it diverges (a diverged run can go on for hundreds of steps with gradient norms
    in the millions): ``patience`` consecutive logged gradient norms above ``max_grad_norm``, or an eval loss more than
    ``eval_rise`` above the best so far. With load_best_model_at_end the final save is the best checkpoint."""

    def __init__(self, max_grad_norm: float, patience: int, eval_rise: float, eval_metric: str = "eval_loss"):
        self.max_grad_norm, self.patience, self.eval_rise = max_grad_norm, patience, eval_rise
        # The watched eval quantity: eval_loss_alignment (raw letter loss) stays comparable over a run; eval_loss
        # does not under --loss-balance ema (terms divided by falling training means drift upward as training improves).
        self.eval_metric = eval_metric
        self.strikes, self.best = 0, None

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Count consecutive logged gradient norms above the limit; stop after ``patience`` of them."""
        grad = (logs or {}).get("grad_norm")
        if grad is None:
            return
        self.strikes = self.strikes + 1 if grad > self.max_grad_norm else 0
        if self.strikes >= self.patience:
            print(
                f"divergence guard: grad_norm > {self.max_grad_norm} for {self.strikes} logs at step "
                f"{state.global_step}; stopping",
                flush=True,
            )
            control.should_training_stop = True

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Stop when the watched eval metric rises more than ``eval_rise`` above its best; track the best."""
        loss = (metrics or {}).get(self.eval_metric)
        if loss is None:
            return
        if self.best is not None and loss > self.best * (1 + self.eval_rise):
            print(
                f"divergence guard: {self.eval_metric} {loss:.3f} > best {self.best:.3f} x {1 + self.eval_rise:.2f} at step "
                f"{state.global_step}; stopping",
                flush=True,
            )
            control.should_training_stop = True
        self.best = loss if self.best is None else min(self.best, loss)


class HeadWarmup(TrainerCallback):
    """For the first ``steps`` optimizer steps only the head groups (``HEAD_PREFIXES``: the CTC, edge, background
    and text heads and the fusion adapter) have gradients, so new heads learn the label statistics before full
    fine-tuning; then everything trains."""

    def __init__(self, model, steps: int):
        self.model, self.steps, self.frozen = model, steps, None

    def on_train_begin(self, args, state, control, **kwargs):
        """Freeze or unfreeze for the starting step (a resumed run may already be past the warm-up)."""
        self._apply(state.global_step)

    def on_step_begin(self, args, state, control, **kwargs):
        """Unfreeze the encoder once the warm-up steps are over."""
        self._apply(state.global_step)

    def _apply(self, step: int):
        frozen = step < self.steps
        if frozen == self.frozen:
            return
        self.frozen = frozen
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(not frozen or name.startswith(HEAD_PREFIXES))
        print(f"head warm-up: {'lm_head only' if frozen else 'encoder unfrozen'} at step {step}", flush=True)


def grow_lora(old, new) -> set[str]:
    """--fusion-lora-grow: copy the adapters of ``old`` (MuQLoRA) into the first rank slots of the same
    adapters of ``new``; every other slot keeps its fresh init (B = 0). Returns the carried parameter names."""
    a, b = old.record, new.record
    if (
        not set(a["blocks"]) <= set(b["blocks"])
        or b["rank"] < a["rank"]
        or not set(muq_fusion.LORA_TARGETS[a["targets"]]) <= set(muq_fusion.LORA_TARGETS[b["targets"]])
        or a["alpha"] / a["rank"] != b["alpha"] / b["rank"]
    ):
        raise ValueError(f"--fusion-lora-grow: {b} does not contain {a} at the same alpha / rank")
    new.reset_parameters()
    names = set()
    with torch.no_grad():
        for key, adapter in old.adapters.items():
            target = new.adapters[key]
            target.lora_a[: a["rank"]] = adapter.lora_a
            target.lora_b[:, : a["rank"]] = adapter.lora_b
            for part in ("lora_a", "lora_b"):
                names.add(f"fusion_lora.adapters.{key}.{part}")
    return names


def grow_lora_optimizer_state(state: dict, model, group: int, old_names: list[str], carried: set[str]) -> dict:
    """Saved AdamW moments for --fusion-lora-grow: the LoRA group (index ``group``) is rebuilt in the new
    model's order, carried tensors zero-padded into their first slots, new tensors without state; the
    other groups keep their moments under renumbered indices."""
    params = dict(model.named_parameters())
    new_names = [name for name in params if name.startswith("fusion_lora")]
    groups, moments, index = [], {}, 0
    for position, saved in enumerate(state["param_groups"]):
        if position == group:
            old_index = dict(zip(old_names, saved["params"]))
            members = []
            for name in new_names:
                source = state["state"].get(old_index[name]) if name in carried else None
                if source is not None:
                    entry = dict(source)
                    for key in ("exp_avg", "exp_avg_sq"):
                        padded = torch.zeros_like(params[name], dtype=source[key].dtype, device=source[key].device)
                        padded[tuple(slice(0, n) for n in source[key].shape)] = source[key]
                        entry[key] = padded
                    moments[index] = entry
                members.append(index)
                index += 1
        else:
            members = []
            for old in saved["params"]:
                if old in state["state"]:
                    moments[index] = state["state"][old]
                members.append(index)
                index += 1
        groups.append({**saved, "params": members})
    return {"state": moments, "param_groups": groups}


def is_qk_norm(name: str) -> bool:
    """Whether a parameter or module name belongs to a text-aligner query/key LayerNorm."""
    return "_q_norm." in name or "_k_norm." in name


def head_warmup_optimizer(
    model,
    learning_rate: float,
    head_lr: float,
    freeze_steps: int,
    warmup_steps: int,
    max_steps: int,
    fusion_lr: float | None = None,
    background_lr: float | None = None,
    lora_lr: float | None = None,
    text_lr: float | None = None,
    optimizer_state: dict | None = None,
    lr_schedule: str = "cosine",
    decay_steps: int = 0,
    text_weight_decay: float = 0.0,
):
    """The Trainer's default optimizer (fused AdamW, no weight decay) in two parameter groups with
    their own schedules: the head runs at ``head_lr`` while the encoder is frozen (its own rate is
    zero), after which both follow the cosine-with-warm-up schedule over the remaining steps.
    ``lr_schedule="wsd"`` replaces the cosine with warm-up / stable / decay: the peak rate holds
    until the last ``decay_steps`` steps, which cosine-decay to zero, so every stable-phase checkpoint (with
    its optimizer state) can be extended or branched into its own short decay."""
    # With --fusion-lr the fusion adapter (projection + gate) gets its own group: new parameters
    # barely move at the encoder's rate (the gate stays near 0.016 and the projection barely changes).
    # With background_lr the background head gets its own group likewise (a fresh head on a warm encoder).
    # With lora_lr the MuQ LoRA (fusion_lora.*) gets its own group as well.
    lora = [p for name, p in model.named_parameters() if lora_lr and name.startswith("fusion_lora")]
    fusion = [
        p
        for name, p in model.named_parameters()
        if fusion_lr and name.startswith("fusion") and not (lora_lr and name.startswith("fusion_lora"))
    ]
    background = [p for name, p in model.named_parameters() if background_lr and name.startswith("background_head")]
    # With text_lr the text-aligner head gets its own group (a fresh head: it also trains at text_lr
    # while the encoder is frozen).
    text = [
        p
        for name, p in model.named_parameters()
        if text_lr and name.startswith("text_aligner") and not is_qk_norm(name)
    ]
    # QK-norm parameters get their own group at the text rate, appended last, so moments saved
    # without them (--init-optimizer-from an older run) still load into the original groups.
    qk = [p for name, p in model.named_parameters() if text_lr and name.startswith("text_aligner") and is_qk_norm(name)]
    head = [
        p
        for name, p in model.named_parameters()
        if name.startswith(HEAD_PREFIXES)
        and not (fusion_lr and name.startswith("fusion"))
        and not (lora_lr and name.startswith("fusion_lora"))
        and not (background_lr and name.startswith("background_head"))
        and not (text_lr and name.startswith("text_aligner"))
    ]
    rest = [p for name, p in model.named_parameters() if not name.startswith(HEAD_PREFIXES)]
    groups = (
        [{"params": rest}, {"params": head}]
        + ([{"params": fusion}] if fusion else [])
        + ([{"params": background}] if background else [])
        + ([{"params": lora}] if lora else [])
        + ([{"params": text}] if text else [])
    )
    optimizer = torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, fused=True)
    text_index = len(groups) - 1 if text else None
    if qk and (optimizer_state is None or len(optimizer_state["param_groups"]) == len(groups) + 1):
        optimizer.add_param_group({"params": qk, "weight_decay": 0.0})
        qk = []
    if optimizer_state is not None:
        # --init-optimizer-from: the saved Adam moments of an earlier run with the same parameter groups
        # (a warm start without the cold-optimizer jolt that can make a restarted run diverge); the learning-rate
        # schedule starts afresh below. Group sizes and every moment's shape must match.
        saved = optimizer_state["param_groups"]
        if [len(g["params"]) for g in saved] != [len(g["params"]) for g in optimizer.param_groups]:
            raise ValueError(
                f"--init-optimizer-from: parameter groups differ "
                f"({[len(g['params']) for g in saved]} vs {[len(g['params']) for g in optimizer.param_groups]})"
            )
        order = [p for g in optimizer.param_groups for p in g["params"]]
        for index, parameter in zip([i for g in saved for i in g["params"]], order):
            moments = optimizer_state["state"].get(index, {})
            for name in ("exp_avg", "exp_avg_sq"):
                if name in moments and moments[name].shape != parameter.shape:
                    raise ValueError(
                        f"--init-optimizer-from: {name} shape {tuple(moments[name].shape)} "
                        f"!= parameter {tuple(parameter.shape)}"
                    )
        optimizer.load_state_dict(optimizer_state)
        for group in optimizer.param_groups:
            group.pop("initial_lr", None)
            group["lr"] = learning_rate
    if qk:  # moments saved before QK-norm existed: the new parameters start with fresh moments
        optimizer.add_param_group({"params": qk, "lr": learning_rate, "weight_decay": 0.0})
    if text_index is not None:
        optimizer.param_groups[text_index]["weight_decay"] = text_weight_decay

    def cosine(step):
        if lr_schedule == "wsd":
            step -= freeze_steps
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            into_decay = step - (max_steps - freeze_steps - decay_steps)
            if into_decay <= 0:
                return 1.0
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, into_decay / max(1, decay_steps))))
        return cosine_lambda(
            step - freeze_steps,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps - freeze_steps,
            num_cycles=0.5,
        )

    lambdas = [
        lambda step: 0.0 if step < freeze_steps else cosine(step),
        lambda step: head_lr / learning_rate if step < freeze_steps else cosine(step),
    ]
    if fusion:
        lambdas.append(
            lambda step: head_lr / learning_rate if step < freeze_steps else cosine(step) * fusion_lr / learning_rate
        )
    if background:
        lambdas.append(
            lambda step: (
                head_lr / learning_rate if step < freeze_steps else cosine(step) * background_lr / learning_rate
            )
        )
    if lora:
        lambdas.append(
            lambda step: head_lr / learning_rate if step < freeze_steps else cosine(step) * lora_lr / learning_rate
        )
    if text:
        lambdas.append(
            lambda step: (
                text_lr / learning_rate * min(1.0, (step + 1) / max(1, warmup_steps))
                if step < freeze_steps
                else cosine(step) * text_lr / learning_rate
            )
        )
    if len(optimizer.param_groups) > len(lambdas):  # the QK-norm group follows the text schedule
        lambdas.append(lambdas[-1])
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)
    return optimizer, schedule


class PartsTrainer(Trainer):
    """Trainer that also logs the model's unweighted loss components (``model._loss_parts``) next to the
    total: ``loss_<part>`` averaged over the logging interval, ``eval_loss_<part>`` over each evaluation."""

    def __init__(self, *args, decoded_eval=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._parts = {True: {}, False: {}}  # training? -> name -> [sum, count]
        self.decoded_eval = decoded_eval  # decoded_eval.DecodedEval: whole-song timing at each evaluation

    def evaluate(self, *args, **kwargs):
        """The usual evaluation, then the whole-song decoded-timing eval (when configured) logged as ``eval_*``."""
        metrics = super().evaluate(*args, **kwargs)
        if self.decoded_eval is not None:
            extra = {f"eval_{k}": v for k, v in self.decoded_eval.run(self.model, self.args.device).items()}
            self.log(extra)
            metrics.update(extra)
        return metrics

    def compute_loss(self, model, inputs, *args, **kwargs):
        """The usual loss; also accumulates this forward's ``_loss_parts`` into the training or evaluation bucket."""
        result = super().compute_loss(model, inputs, *args, **kwargs)
        parts = getattr(self.accelerator.unwrap_model(model), "_loss_parts", None) or {}
        bucket = self._parts[model.training]
        for name, value in parts.items():
            entry = bucket.setdefault(name, [0.0, 0])
            entry[0] += float(value)
            entry[1] += 1
        return result

    def log(self, logs, *args, **kwargs):
        """Add the averaged loss parts (and reset them) to logs carrying ``loss`` / ``eval_loss``; training logs also
        get ``text_qk_max``, the largest query/key weight norm of the text head's cross-attention layers."""
        for training, prefix, key in ((True, "loss_", "loss"), (False, "eval_loss_", "eval_loss")):
            if key in logs and self._parts[training]:
                logs.update(
                    {prefix + name: round(total / max(1, n), 4) for name, (total, n) in self._parts[training].items()}
                )
                self._parts[training] = {}
        head = getattr(self.model, "text_aligner", None)
        if head is not None and "loss" in logs:
            with torch.no_grad():
                d = head.config.dim
                logs["text_qk_max"] = round(
                    max(
                        w[: 2 * d].view(2, d, -1).norm(dim=(1, 2)).max().item()
                        for w in (layer.cross_attention.in_proj_weight for layer in head.cross)
                    ),
                    2,
                )
        return super().log(logs, *args, **kwargs)


def metrics(prediction):
    """Evaluation metrics from the CTC logits and the weak frame labels: lyric frame accuracy, lyric fill and silence
    spill, plus the extra head's summary when the model returns per-window stats (background or stream-2 recall and
    spill, aligned overlap losses, or overlap/letter precision, recall and F1). The stats width picks the head."""
    logits, labels = prediction.predictions, prediction.label_ids
    if isinstance(labels, (tuple, list)):  # two streams: Trainer also collects s2_labels
        labels = labels[0]
    overlap = {}
    if isinstance(logits, (tuple, list)):
        stats = np.asarray(logits[1]) if len(logits) > 1 and logits[1] is not None else None
        if stats is not None and stats.shape[-1] == 5:
            heard, words, spill, silent, _ = stats.reshape(-1, 5).sum(0)
            overlap = {
                "background_word_recall": float(heard / max(words, 1)),
                "background_silence_spill": float(spill / max(silent, 1)),
            }
        elif stats is not None and stats.shape[-1] == 4:
            heard, words, spill, silent = stats.reshape(-1, 4).sum(0)
            overlap = {
                "stream2_word_recall": float(heard / max(words, 1)),
                "stream2_silence_spill": float(spill / max(silent, 1)),
            }
        elif stats is not None and stats.shape[-1] == 3:
            positive, negative, silence = stats.reshape(-1, 3).mean(0)
            overlap = {
                "overlap_word_nll": float(positive),
                "overlap_negative_bce": float(negative),
                "overlap_silence_bce": float(silence),
            }
        elif stats is not None:
            tp, fp, fn, ltp, lfp, lfn = stats.reshape(-1, 6).sum(0)
            overlap = {
                "overlap_precision": float(tp / max(tp + fp, 1)),
                "overlap_recall": float(tp / max(tp + fn, 1)),
                "overlap_f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
                "letter_f1": float(2 * ltp / max(2 * ltp + lfp + lfn, 1)),
            }
        logits = logits[0]
    predicted = np.argmax(logits, axis=-1)
    # Padding represents non-lyric frames. It belongs in the loss, but would
    # make an accuracy number look good even when lyric characters are wrong.
    lyric = (labels != -100) & (labels != PAD_TOKEN_ID)
    # True silence: blank frames at least two frames from any lyric or mask.
    blank = labels == PAD_TOKEN_ID
    quiet = blank.copy()
    for shift in (1, 2):
        quiet[:, shift:] &= blank[:, :-shift]
        quiet[:, :-shift] &= blank[:, shift:]
    sung = predicted != PAD_TOKEN_ID
    return {
        # Weak-label agreement, not acoustic boundary accuracy.
        "lyric_frame_accuracy": float((predicted[lyric] == labels[lyric]).mean()) if lyric.any() else 0.0,
        # Share of lyric frames predicted as some letter: low means peaky collapse.
        "lyric_fill": float(sung[lyric].mean()) if lyric.any() else 0.0,
        # Share of true-silence frames predicted as a letter: high means spill.
        "silence_spill": float(sung[quiet].mean()) if quiet.any() else 0.0,
        **overlap,
    }


def arguments():
    """Parse and sanity-check the training command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, required=True, help="Prepared dataset directory (split manifests and vocab.json)"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--model", required=True, help="Local checkpoint directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", default="echora-kara")
    parser.add_argument("--window-seconds", type=float, default=25.0)
    parser.add_argument(
        "--window-policy",
        choices=("fixed", "snap"),
        default="fixed",
        help="snap: end each window in a gap between lead fragments (20-27 s), so no fragment is cut",
    )
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument(
        "--eval-limit", type=int, default=None, help="Optional debug cap. Leave unset for full validation."
    )
    parser.add_argument(
        "--eval-windows-per-track",
        type=int,
        default=None,
        help="Optional debug cap. Leave unset to evaluate all song windows.",
    )
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--index-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Evaluation intervals without improvement before stopping. Zero disables it.",
    )
    parser.add_argument(
        "--loss",
        choices=(*REGION_LOSSES, "frame"),
        default="constrained_alignment",
        help="constrained_alignment: letters fill each measured source fragment in order, "
        "boundaries free. constrained_ctc: same with blank allowed (peaky; for "
        "comparison). frame: the weak proportional letter split graded per frame.",
    )
    parser.add_argument(
        "--boundary-tolerance-frames",
        type=int,
        default=1,
        help="Ungraded blank frames at each region edge (region losses only).",
    )
    parser.add_argument(
        "--region-fallback",
        choices=("proportional", "ignore"),
        default="proportional",
        help="Fragments that get no region (window-cut, overlapped, too short): proportional "
        "per-letter frame labels (default) or ignore.",
    )
    parser.add_argument("--resume", action="store_true")
    v7 = parser.add_argument_group("overlap head, second stream, augmentation and regularisation (all off by default)")
    v7.add_argument(
        "--overlap-loss-weight",
        type=float,
        default=0.0,
        help="Add the sigmoid overlap head with this loss weight. Zero: no overlap head.",
    )
    v7.add_argument(
        "--overlap-frame-weight",
        type=float,
        default=2.0,
        help="Extra weight of frames where two or more words are sung at once.",
    )
    v7.add_argument(
        "--overlap-mode",
        choices=("aligned", "frame"),
        default="aligned",
        help="aligned: each word's letters aligned in order over its frames, only letters of "
        "no sung word penalized. frame: per-frame multi-label targets.",
    )
    v7.add_argument(
        "--overlap-token-weight",
        type=float,
        default=4.0,
        help="aligned mode: weight of words that overlap another word.",
    )
    v7.add_argument(
        "--overlap-encoder-grad-scale",
        type=float,
        default=1.0,
        help="Share of the overlap loss's gradient that reaches the encoder (1: all).",
    )
    v7.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Train only the overlap head (a probe of what the encoder already knows).",
    )
    v7.add_argument(
        "--two-stream",
        action="store_true",
        help="Add a second CTC stream for words sung while another is (train.streams).",
    )
    v7.add_argument("--stream2-loss-weight", type=float, default=1.0)
    v7.add_argument("--stream2-blank-weight", type=float, default=0.2)
    v7.add_argument(
        "--stream2-blank-bias",
        type=float,
        default=5.0,
        help="Extra blank bias of the new stream-2 head at the start (it starts silent).",
    )
    v7.add_argument(
        "--distill-from", default=None, help="Frozen teacher checkpoint: stream 1 is pulled toward its outputs."
    )
    v7.add_argument("--distill-weight", type=float, default=1.0)
    v7.add_argument("--save-total-limit", type=int, default=3)
    v7.add_argument(
        "--stream2-encoder-grad-scale",
        type=float,
        default=1.0,
        help="Share of stream 2's gradient that reaches the shared encoder.",
    )
    v7.add_argument(
        "--stream-min-overlap",
        type=float,
        default=0.1,  # streams.MIN_OVERLAP
        help="Seconds a word must overlap a stream-1 word to go to stream 2.",
    )
    v7.add_argument("--augment", action="store_true", help="Musical augmentation on training windows only.")
    v7.add_argument("--dropout", type=float, default=0.0, help="Hidden, attention and projection dropout.")
    v7.add_argument("--layerdrop", type=float, default=0.0)
    v7.add_argument("--mask-time-prob", type=float, default=None, help="SpecAugment; default keeps the model config.")
    v7.add_argument("--mask-time-length", type=int, default=None)
    v7.add_argument("--mask-feature-prob", type=float, default=None)
    v7.add_argument("--mask-feature-length", type=int, default=None)
    large = parser.add_argument_group("new backbone: head warm-up and learning-rate schedule (off by default)")
    large.add_argument(
        "--freeze-encoder-steps",
        type=int,
        default=0,
        help="Head warm-up: for the first N optimizer steps only lm_head trains, at "
        "--head-warmup-lr; then the full model follows the usual cosine schedule "
        "(warm-up 10%% of --max-steps) over the remaining steps. N counts toward --max-steps.",
    )
    large.add_argument(
        "--head-warmup-lr", type=float, default=1e-3, help="Head learning rate while the encoder is frozen."
    )
    guide = parser.add_argument_group("loss guidance (off by default)")
    guide.add_argument(
        "--text-encoder-grad-scale",
        type=float,
        default=1.0,
        help="share of the text head's gradient reaching the shared encoder/fusion (e.g. 0.1)",
    )
    guide.add_argument(
        "--loss-balance",
        choices=("none", "ema"),
        default="none",
        help="ema: each loss term divided by its running mean, then weighted",
    )
    guide.add_argument("--alignment-weight", type=float, default=1.0, help="weight of the letter alignment loss")
    guide.add_argument(
        "--background-balance",
        action="store_true",
        help="background head: blank frames weighted n_sung / n_blank per batch instead of the fixed weight",
    )
    guide.add_argument(
        "--text-fs-mode",
        choices=("ctc", "region"),
        default="ctc",
        help="region: blank-free token filling of fragment groups instead of the CTC forward-sum",
    )
    guide.add_argument(
        "--decoded-eval-ids",
        default=None,
        help="file of validation song ids: whole-song decoded timing at every evaluation (eval_decoded_*)",
    )
    guide.add_argument("--decoded-eval-songs", type=int, default=40)
    guide.add_argument(
        "--best-metric",
        default=None,
        help="metric for the best checkpoint, e.g. decoded_mean_ms (lower is better); default eval_loss",
    )
    guard = parser.add_argument_group("warm start and divergence guard")
    guard.add_argument(
        "--init-optimizer-from",
        default=None,
        help="checkpoint directory whose optimizer.pt (Adam moments) seeds this NEW run; schedule restarts",
    )
    guard.add_argument(
        "--guard-grad-norm",
        type=float,
        default=0.0,
        help="stop after --guard-patience logged grad norms above this (0: off)",
    )
    guard.add_argument("--guard-patience", type=int, default=5)
    guard.add_argument(
        "--guard-eval-rise",
        type=float,
        default=0.2,
        help="stop when the watched eval metric exceeds its best by this fraction",
    )
    guard.add_argument(
        "--guard-eval-metric",
        default="eval_loss",
        help="eval metric the guard watches (eval_loss_alignment: raw letter loss, comparable under --loss-balance)",
    )
    text = parser.add_argument_group("text-conditioned aligner head (off by default)")
    text.add_argument(
        "--song-text-version",
        type=int,
        choices=(1, 2),
        default=None,
        help="Required for new text training: 2. Legacy resume uses 1; no implicit migration.",
    )
    text.add_argument(
        "--text-aligner-weight",
        type=float,
        default=0.0,
        help="Weight of the head's loss (0: no head). The head reads the fused frame embeddings and the "
        "whole song's lyric tokens; factorized output, line embeddings.",
    )
    text.add_argument(
        "--text-aligner-lr", type=float, default=None, help="The head's own learning rate (a fresh head)."
    )
    text.add_argument(
        "--text-aligner-qk-norm",
        action="store_true",
        help="LayerNorm on the cross layers' attention queries/keys (bounds the attention logits, whose "
        "unbounded growth diverges runs at high learning rates; new norms start at identity).",
    )
    parser.add_argument(
        "--mask-unlabelled-vocals",
        action="store_true",
        help="Training windows: the manifest rows' unlabelled_vocal_spans (voiced in the vocal stem, no "
        "label) are ignored instead of taught as not singing (letter, edge and lead text targets).",
    )
    text.add_argument(
        "--text-aligner-weight-decay",
        type=float,
        default=0.0,
        help="Decoupled AdamW weight decay of the text head's parameter group (grouped optimizer).",
    )
    text.add_argument("--text-aligner-dim", type=int, default=512)
    text.add_argument("--text-aligner-heads", type=int, default=8)
    text.add_argument("--text-aligner-text-layers", type=int, default=6)
    text.add_argument("--text-aligner-cross-layers", type=int, default=4)
    text.add_argument(
        "--text-aligner-fs-weight", type=float, default=0.2, help="Forward-sum (CTC over token columns) weight."
    )
    text.add_argument("--text-aligner-back-weight", type=float, default=0.5, help="Weight of the backing voice's loss.")
    bg = parser.add_argument_group("background head (off by default)")
    bg.add_argument(
        "--background-head-weight",
        type=float,
        default=0.0,
        help="Weight of the background head's loss on the manifest's background_tokens (schema 7). "
        "0 (default): no head.",
    )
    bg.add_argument(
        "--background-blank-weight",
        type=float,
        default=0.2,
        help="Frame-loss weight of the background head's blank frames (it is silent most of the time).",
    )
    bg.add_argument(
        "--background-blank-bias",
        type=float,
        default=5.0,
        help="Extra blank bias of a fresh background head (starts silent, copied from the CTC head).",
    )
    bg.add_argument(
        "--background-head-lr",
        type=float,
        default=None,
        help="Own learning rate (same warm-up + cosine shape) for the background head, e.g. 1e-3 for a "
        "fresh head on a warm-started encoder.",
    )
    large.add_argument(
        "--fusion-lr",
        type=float,
        default=None,
        help="Peak learning rate of the fusion adapter (own group, same cosine shape); default: encoder rate.",
    )
    large.add_argument("--warmup-steps", type=int, default=None, help="Override the 10 %% linear warm-up.")
    large.add_argument(
        "--lr-schedule",
        choices=["cosine", "wsd"],
        default="cosine",
        help="cosine (default) or wsd: warm-up, constant peak, cosine decay over the last --decay-steps "
        "(grouped optimizer only; any stable-phase checkpoint can be extended).",
    )
    large.add_argument(
        "--decay-steps", type=int, default=0, help="Length of the wsd decay phase (e.g. 10 %% of --max-steps)."
    )
    large.add_argument(
        "--backbone",
        choices=("auto", "wav2vec2", "wav2vec2-bert"),
        default="auto",
        help="Checkpoint family. auto reads config.json's model_type: wav2vec2 (MMS, omniASR) "
        "or wav2vec2-bert (w2v-BERT 2.0: SeamlessM4T "
        "fbank input, 279.5-sample receptive-field centre, same 20 ms stride). Naming one "
        "makes a mismatch an error instead of a silent choice.",
    )
    s1 = parser.add_argument_group("edge head on measured fragment starts (off by default)")
    s1.add_argument(
        "--edge-loss-weight",
        type=float,
        default=0.0,
        help="Add the 1-logit edge head (P(token start | frame)) with this loss weight.",
    )
    s1.add_argument(
        "--edge-positive-weight",
        type=float,
        default=20.0,
        help="BCE weight of the positive (fragment-start) frames; about 1 in 20 frames is one.",
    )
    s1.add_argument(
        "--edge-emd-weight",
        type=float,
        default=1.0,
        help="Weight of the cumulative-sum L1 (EMD) term inside the edge loss (0: BCE only).",
    )
    s1.add_argument(
        "--edge-tolerance-frames", type=int, default=1, help="Ungraded BCE frames on each side of a fragment start."
    )
    fusion = parser.add_argument_group("MuQ fusion (off by default)")
    fusion.add_argument(
        "--fusion",
        choices=("none", *muq_fusion.FUSION_KINDS),
        default="none",
        help="muq: a frozen MuQ-large-msd-iter encoder hears the raw window (resampled to 24 kHz); "
        "one of its layers is projected and added, through a zero-initialised gate, to the "
        "encoder's last hidden state before lm_head and the edge head. The projection and "
        "gate train with the heads (head warm-up included) and are saved as fusion.*.",
    )
    fusion.add_argument(
        "--fusion-layer",
        type=int,
        default=muq_fusion.DEFAULT_LAYER,
        choices=muq_fusion.MUQ_LAYERS,
        help="MuQ layer: 0 = conformer input, 1-12 = block outputs, 13 = after the final norm "
        "(default chosen by a linear vocal-activity probe).",
    )
    fusion.add_argument(
        "--fusion-model",
        default=muq_fusion.MUQ_MODEL_ID,
        help="MuQ hub id (resolved offline from HF_HOME) or local snapshot directory.",
    )
    fusion.add_argument(
        "--fusion-gate",
        choices=("scalar", "channel"),
        default="scalar",
        help="One gate for the whole projection or one per hidden channel; both start at 0.",
    )
    fusion.add_argument(
        "--fusion-lora-rank",
        type=int,
        default=0,
        help="LoRA rank on the frozen MuQ branch (0, the default: no LoRA, bit-identical). "
        "B starts at 0, so step 0 equals the model without it; saved as fusion_lora.*.",
    )
    fusion.add_argument(
        "--fusion-lora-alpha", type=float, default=16.0, help="LoRA scale numerator (scale = alpha / rank)."
    )
    fusion.add_argument(
        "--fusion-lora-layers",
        type=int,
        default=4,
        help="Number of MuQ conformer blocks, ending at the fused layer, that get LoRA.",
    )
    fusion.add_argument(
        "--fusion-lora-targets",
        choices=("qv", "qv_ffn"),
        default="qv",
        help="Projections adapted: attention q/v, or q/v plus both FFN layers.",
    )
    fusion.add_argument(
        "--fusion-lora-lr",
        type=float,
        default=None,
        help="Own peak learning rate for the LoRA parameters (same warm-up + cosine; default: --fusion-lr).",
    )
    fusion.add_argument(
        "--fusion-lora-grow",
        action="store_true",
        help="--model has a smaller LoRA: carry it into the larger one asked for (a superset "
        "of its blocks and targets, rank >= its rank, same alpha / rank). Old adapters keep their "
        "weights in the first rank slots, new slots and adapters start with B = 0, so step 0 equals "
        "--model; --init-optimizer-from moments are carried into the same slots.",
    )
    args = parser.parse_args()
    if args.edge_loss_weight < 0 or args.edge_tolerance_frames < 0:
        parser.error("--edge-loss-weight and --edge-tolerance-frames must be non-negative")
    if args.boundary_tolerance_frames < 0:
        parser.error("--boundary-tolerance-frames must be non-negative")
    if args.freeze_encoder_steps < 0 or (args.freeze_encoder_steps and args.freeze_encoder_steps >= args.max_steps):
        parser.error("--freeze-encoder-steps must be non-negative and below --max-steps")
    if args.lr_schedule == "wsd" and not (
        0 < args.decay_steps < args.max_steps - args.freeze_encoder_steps - (args.warmup_steps or 0)
    ):
        parser.error(
            "--lr-schedule wsd needs 0 < --decay-steps < --max-steps - --freeze-encoder-steps - --warmup-steps"
        )
    return args


def file_sha256(path: Path):
    """Hex SHA-256 of a file's contents."""
    import hashlib

    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def validate_training_manifests(data: Path, root: Path):
    """Check every row of each split manifest in ``data`` with ``check_row`` (audio under ``root``, no song in two
    splits); raises ``ValueError`` on an empty split."""
    assignments = new_assignments()
    for split in SPLITS:
        count = 0
        with (data / f"{split}.jsonl").open() as source:
            for line in source:
                if line.strip():
                    check_row(json.loads(line), split, assignments, root, prepared=True)
                    count += 1
        if not count:
            raise ValueError(f"Empty {split} manifest")


def main():
    """Train the aligner from ``--model`` on the prepared dataset in ``--data``.

    Builds the config with the requested heads and fusion, loads and warm-starts the model, indexes the windows,
    writes ``data_identity.json`` (dataset, window and training-recipe identity that ``--resume`` must match), trains
    with the HF Trainer and saves the final (best) model and processor to ``--output``."""
    args = arguments()
    set_seed(args.seed)
    if args.save_total_limit is not None and args.save_total_limit < 2:
        raise ValueError("--save-total-limit must be at least 2 to retain the previous complete checkpoint")
    if args.resume:
        checkpoint = get_last_checkpoint(str(args.output)) if args.output.exists() else None
        identity_path = args.output / "data_identity.json"
        # Fail before loading any models, especially for an unrecorded legacy recipe.
        recorded = json.loads(identity_path.read_text()) if identity_path.exists() else {}
        validate_resume(identity_path, recorded, True, checkpoint)
    validate_training_manifests(args.data, args.root)
    if not Path(args.model).is_dir():
        raise ValueError("--model must point to an existing local checkpoint directory")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this trainer")
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        args.model, local_files_only=True, word_delimiter_token="|", unk_token="[UNK]", pad_token="[PAD]"
    )
    backbone = aligner_class(args.model).frame_clock.backbone
    if args.backbone != "auto" and args.backbone != backbone:
        raise ValueError(f"--backbone {args.backbone} but {args.model} is a {backbone} checkpoint")
    clock = CLOCKS[backbone]
    spec_augment = {
        key: value
        for key, value in (
            ("mask_time_prob", args.mask_time_prob),
            ("mask_time_length", args.mask_time_length),
            ("mask_feature_prob", args.mask_feature_prob),
            ("mask_feature_length", args.mask_feature_length),
        )
        if value is not None
    }
    overlap = args.overlap_loss_weight > 0
    if backbone == "wav2vec2-bert":
        if overlap or args.two_stream or args.distill_from or args.augment:
            raise ValueError(
                "the w2v-BERT backbone supports the single-stream recipe only "
                "(no overlap head, stream 2, distillation or augmentation)"
            )
        extractor = SeamlessM4TFeatureExtractor.from_pretrained(args.model, local_files_only=True)
        if (extractor.sampling_rate, extractor.num_mel_bins, extractor.stride) != (SAMPLE_RATE, 80, 2):
            raise ValueError("w2v-BERT feature extractor is not 16 kHz / 80 mel bins / stride 2: label clock differs")
        processor = Wav2Vec2BertProcessor(extractor, tokenizer)
        # By default every dropout is 0; w2v-BERT adds the conformer conv module's and the
        # final (pre-head) dropout, which follow --dropout too.
        config = Wav2Vec2BertConfig.from_pretrained(
            args.model,
            local_files_only=True,
            attention_dropout=args.dropout,
            hidden_dropout=args.dropout,
            feat_proj_dropout=args.dropout,
            final_dropout=args.dropout,
            conformer_conv_dropout=args.dropout,
            layerdrop=args.layerdrop,
            **spec_augment,
        )
        if config.add_adapter or config.feature_projection_input_dim != 160:
            raise ValueError("w2v-BERT checkpoint has an adapter or a non-stacked input: label clock differs")
        if not config.apply_spec_augment and config.mask_time_prob > 0:
            print("note: apply_spec_augment is off in the checkpoint config; no SpecAugment", flush=True)
    else:
        processor = Wav2Vec2Processor(
            Wav2Vec2FeatureExtractor(
                feature_size=1,
                sampling_rate=SAMPLE_RATE,
                padding_value=0.0,
                do_normalize=True,
                return_attention_mask=True,
            ),
            tokenizer,
        )
        config = Wav2Vec2Config.from_pretrained(
            args.model,
            local_files_only=True,
            attention_dropout=args.dropout,
            hidden_dropout=args.dropout,
            feat_proj_dropout=args.dropout,
            layerdrop=args.layerdrop,
            **spec_augment,
        )
    if overlap:
        config.overlap_head = True
    if args.two_stream:
        config.stream2_head = True
    if args.edge_loss_weight > 0:
        config.edge_head = True
    if args.background_head_weight > 0:
        config.background_head = True
    text_head = args.text_aligner_weight > 0
    if text_head:
        version = args.song_text_version
        # A head carried over from --model keeps the text version it was trained with;
        # only a NEW head must use the current version.
        init_config = (
            json.loads((Path(args.model) / "config.json").read_text())
            if (Path(args.model) / "config.json").exists()
            else {}
        )
        carried = init_config.get("text_aligner") is not None and init_config.get("song_text_version", 1) == version
        if (
            not args.resume
            and version != CURRENT_VERSION
            and not carried
            and os.environ.get("FA_DIAG_ALLOW_SONG_TEXT_V1") != "1"
        ):
            raise ValueError("New text-head training requires --song-text-version 2")
        if args.resume:
            checkpoint = get_last_checkpoint(str(args.output))
            if not checkpoint:
                raise ValueError("--resume requested but no checkpoint exists")
            saved_version = json.loads((Path(checkpoint) / "config.json").read_text()).get("song_text_version", 1)
            if version is not None and version != saved_version:
                raise ValueError("Resume cannot change song_text_version; use a new run")
            version = saved_version
        config.song_text_version = version
        config.text_aligner = ta.TextAlignerConfig(
            frame_dim=config.hidden_size,
            letter_dim=config.vocab_size,
            dim=args.text_aligner_dim,
            heads=args.text_aligner_heads,
            text_layers=args.text_aligner_text_layers,
            cross_layers=args.text_aligner_cross_layers,
            factorized=True,
            line_embeddings=True,
            qk_norm=args.text_aligner_qk_norm,
        ).to_dict()
    elif getattr(config, "text_aligner", None):
        config.text_aligner = None  # a checkpoint with the head, trained without it: drop it
    if args.fusion == "muq":
        config.fusion = muq_fusion.fusion_config(
            args.fusion_layer,
            args.fusion_gate,
            args.fusion_model,
            lora=muq_fusion.lora_config(
                args.fusion_lora_rank,
                args.fusion_lora_alpha,
                args.fusion_lora_layers,
                args.fusion_layer,
                args.fusion_lora_targets,
            )
            if args.fusion_lora_rank > 0
            else None,
        )
    elif getattr(config, "fusion", None):
        raise ValueError(f"{args.model} is a fusion checkpoint; pass --fusion muq (and its layer/model) to train it")
    grown_lora = None
    if args.fusion_lora_grow:
        saved_lora = (init_config.get("fusion") or {}).get("lora") if args.fusion == "muq" else None
        if not saved_lora or not config.fusion.get("lora"):
            raise ValueError("--fusion-lora-grow needs --fusion-lora-rank > 0 and a --model trained with LoRA")
        grown_lora = config.fusion["lora"]
        config.fusion = {**config.fusion, "lora": saved_lora}  # load the saved adapters as they are, grow below
    model, loading = aligner_class(args.model).from_pretrained(
        args.model, config=config, local_files_only=True, output_loading_info=True
    )
    old_lora_names = None
    if grown_lora is not None:
        old_lora_names = [name for name, _ in model.named_parameters() if name.startswith("fusion_lora")]
        grown = muq_fusion.MuQLoRA(grown_lora).to(next(model.fusion_lora.parameters()).dtype)
        carried_lora = grow_lora(model.fusion_lora, grown)
        model.fusion_lora = grown
        model.config.fusion = {**model.config.fusion, "lora": grown_lora}
        print(f"note: LoRA grown {saved_lora} -> {grown_lora} ({len(carried_lora)} tensors carried)", flush=True)
    if args.edge_loss_weight > 0 and any(key.startswith("edge_head") for key in loading["missing_keys"]):
        model.warm_start_edge_head()
    if args.background_head_weight > 0:
        if any(key.startswith("background_head") for key in loading["missing_keys"]):
            model.warm_start_background_head(args.background_blank_bias)
        model.background_loss_weight = args.background_head_weight
        model.background_blank_weight = args.background_blank_weight
    if args.fusion == "muq":
        if any(key.startswith("fusion.") for key in loading["missing_keys"]):
            model.warm_start_fusion()
        if model.fusion_lora is not None and any(key.startswith("fusion_lora.") for key in loading["missing_keys"]):
            model.fusion_lora.reset_parameters()  # fresh LoRA: B = 0, the model is unchanged at step 0
        model.attach_fusion_backbone(muq_fusion.MuQBranch(args.fusion_model, device="cuda"))
    if text_head:
        missing_text = [k for k in loading["missing_keys"] if k.startswith("text_aligner.")]
        if missing_text:
            # from_pretrained leaves a head absent from the checkpoint to the backbone's _init_weights, which skips
            # raw parameters (letter_scale stayed uninitialised memory: ~0 on some loads, huge on others) and gives
            # embeddings std 0.02; use the head's own initialisation instead.
            if all(is_qk_norm(k) for k in missing_text):  # a trained head gaining QK-norm: identity norms
                for name, module in model.text_aligner.named_modules():
                    if is_qk_norm(name + "."):
                        module.reset_parameters()
            elif len(missing_text) != len(list(model.text_aligner.state_dict())):
                raise ValueError(f"{args.model} has a partial text head ({len(missing_text)} keys missing)")
            else:
                model.text_aligner.load_state_dict(
                    ta.TextAligner(ta.TextAlignerConfig(**config.text_aligner)).state_dict()
                )
        model.text_aligner_weight = args.text_aligner_weight
        model.text_fs_weight = args.text_aligner_fs_weight
        model.text_back_weight = args.text_aligner_back_weight
        print(
            f"text aligner head: {sum(p.numel() for p in model.text_aligner.parameters()) / 1e6:.1f} M parameters "
            f"({'fresh' if any(k.startswith('text_aligner.') for k in loading['missing_keys']) else 'from checkpoint'})",
            flush=True,
        )
    model.text_encoder_grad_scale = args.text_encoder_grad_scale
    model.loss_balance = args.loss_balance
    model.alignment_weight = args.alignment_weight
    model.background_balance = args.background_balance
    model.text_fs_mode = args.text_fs_mode
    model.edge_loss_weight = args.edge_loss_weight
    model.edge_positive_weight = args.edge_positive_weight
    model.edge_emd_weight = args.edge_emd_weight
    if overlap and any(key.startswith("overlap_head") for key in loading["missing_keys"]):
        model.warm_start_overlap_head()
    model.overlap_loss_weight = args.overlap_loss_weight
    model.overlap_frame_weight = args.overlap_frame_weight
    model.overlap_mode = args.overlap_mode
    if args.two_stream and any(key.startswith("stream2_head") for key in loading["missing_keys"]):
        model.warm_start_stream2_head(args.stream2_blank_bias)
    if args.distill_from:
        model.distill_weight = args.distill_weight
        model.set_teacher(Wav2Vec2ForCTC.from_pretrained(args.distill_from, local_files_only=True).to("cuda"))
    model.stream2_loss_weight = args.stream2_loss_weight
    model.stream2_blank_weight = args.stream2_blank_weight
    model.stream2_encoder_grad_scale = args.stream2_encoder_grad_scale
    model.overlap_encoder_grad_scale = args.overlap_encoder_grad_scale
    if args.freeze_encoder:
        if not overlap:
            raise ValueError("--freeze-encoder trains only the overlap head; set --overlap-loss-weight")
        model.freeze_encoder = True
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("overlap_head")
    if backbone == "wav2vec2" and (
        list(model.config.conv_kernel) != [10, 3, 3, 3, 3, 2, 2]
        or list(model.config.conv_stride) != [5, 2, 2, 2, 2, 2, 2]
    ):
        raise ValueError("Model convolution clock differs from the label clock")
    if backbone == "wav2vec2-bert" and clock.frames(SAMPLE_RATE) != 49:
        raise ValueError("w2v-BERT frame clock self-check failed")
    if tokenizer.pad_token_id != PAD_TOKEN_ID or model.config.pad_token_id != PAD_TOKEN_ID:
        raise ValueError("Model/tokenizer blank ID does not match the manifest")
    manifest_vocab = json.loads((args.data / "vocab.json").read_text())
    tokenizer_vocab = tokenizer.get_vocab()
    if any(tokenizer_vocab.get(token) != index for token, index in manifest_vocab.items()):
        raise ValueError("Tokenizer label IDs do not match prepared dataset")
    if max(manifest_vocab.values()) >= model.config.vocab_size:
        raise ValueError("Model output head cannot represent manifest labels")
    # Unused tokenizer BOS/EOS additions are harmless; never resize a trained
    # output head just to match their count.
    model.config.use_cache = False
    model.region_loss = args.loss
    model.focal_frames = args.loss == "frame"
    background = args.background_head_weight > 0
    train = TimedWindowDataset(
        args.root,
        args.data / "train.jsonl",
        args.window_seconds,
        args.train_limit,
        index_workers=args.index_workers,
        augment=augmentation.AugmentConfig() if args.augment else None,
        background=background,
        snap=WINDOW_SNAP if args.window_policy == "snap" else None,
        mask_unlabelled=args.mask_unlabelled_vocals,
    )
    evaluation = TimedWindowDataset(
        args.root,
        args.data / "validation.jsonl",
        args.window_seconds,
        args.eval_limit,
        args.eval_windows_per_track,
        args.index_workers,
        background=background,
        snap=WINDOW_SNAP if args.window_policy == "snap" else None,
    )
    song_text = None
    if text_head:
        song_text = build_song_text(
            [args.data / "train.jsonl", args.data / "validation.jsonl"],
            args.index_workers,
            model.text_aligner.config.max_letters,
            model.text_aligner.config.letters,
            config.song_text_version,
        )
        print(f"text aligner: {len(song_text)} songs' lyric tokens indexed", flush=True)
    print(
        json.dumps(
            {"train_windows": len(train), "eval_windows": len(evaluation), "output": str(args.output)}, indent=2
        ),
        flush=True,
    )
    train_args = TrainingArguments(
        output_dir=str(args.output),
        run_name=args.run_name,
        max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps
        if args.warmup_steps is not None
        else (1000 if args.max_steps < 0 else max(1, args.max_steps // 10)),
        bf16=True,
        tf32=True,
        gradient_checkpointing=not args.freeze_encoder,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        # Frame accuracy rewards agreeing with the weak in-fragment split,
        # which the region losses deliberately stop enforcing.
        logging_steps=10,
        report_to=["wandb"],
        # Trainer guesses label names from forward's parameters: ``s2_labels`` matches too, so a
        # single-stream run (no s2_labels in the batch) would collect no eval loss or metrics.
        label_names=["labels"],
        metric_for_best_model=(
            f"eval_{args.best_metric}"
            if args.best_metric
            else "eval_loss"
            if args.loss in REGION_LOSSES
            else "lyric_frame_accuracy"
        ),
        greater_is_better=(
            args.best_metric.endswith(("f1_50", "_ap")) if args.best_metric else args.loss not in REGION_LOSSES
        ),
        load_best_model_at_end=True,
        dataloader_num_workers=args.workers,
        dataloader_persistent_workers=args.workers > 0,
        eval_accumulation_steps=8,
        remove_unused_columns=False,
    )
    callbacks = [CompletedSave()]
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
    optimizers = (None, None)
    if args.freeze_encoder_steps:
        callbacks.append(HeadWarmup(model, args.freeze_encoder_steps))
    if args.guard_grad_norm > 0:
        callbacks.append(
            DivergenceGuard(args.guard_grad_norm, args.guard_patience, args.guard_eval_rise, args.guard_eval_metric)
        )
    lora_lr = (args.fusion_lora_lr or args.fusion_lr) if args.fusion_lora_rank > 0 else None
    text_lr = args.text_aligner_lr if text_head else None
    if args.freeze_encoder_steps or args.fusion_lr or args.background_head_lr or lora_lr or text_lr:
        optimizer_state = None
        if args.init_optimizer_from:
            optimizer_state = torch.load(
                Path(args.init_optimizer_from) / "optimizer.pt", map_location="cpu", weights_only=False
            )
            print(
                f"optimizer moments from {args.init_optimizer_from} ({len(optimizer_state['state'])} tensors)",
                flush=True,
            )
            if old_lora_names is not None:  # groups: rest, head, [fusion], [background], lora, ...
                lora_group = 2 + bool(args.fusion_lr) + bool(args.background_head_lr and background)
                optimizer_state = grow_lora_optimizer_state(
                    optimizer_state, model, lora_group, old_lora_names, carried_lora
                )
        optimizers = head_warmup_optimizer(
            model,
            args.learning_rate,
            args.head_warmup_lr,
            args.freeze_encoder_steps,
            train_args.warmup_steps,
            args.max_steps,
            args.fusion_lr,
            args.background_head_lr if background else None,
            lora_lr,
            text_lr,
            optimizer_state,
            args.lr_schedule,
            args.decay_steps,
            args.text_aligner_weight_decay,
        )
    elif args.init_optimizer_from:
        raise ValueError("--init-optimizer-from needs the grouped optimizer (set --fusion-lr / --text-aligner-lr ...)")
    elif args.lr_schedule == "wsd":
        raise ValueError("--lr-schedule wsd needs the grouped optimizer (set --fusion-lr / --text-aligner-lr ...)")
    decoded = None
    if args.decoded_eval_ids:
        from echora_kara.train.decoded_eval import DecodedEval

        decoded = DecodedEval(
            args.root,
            args.data / "validation.jsonl",
            Path(args.decoded_eval_ids),
            args.decoded_eval_songs,
            song_text if text_head else None,
        )
    trainer = PartsTrainer(
        decoded_eval=decoded,
        model=model,
        args=train_args,
        train_dataset=train,
        eval_dataset=evaluation,
        data_collator=AlignmentCollator(
            processor,
            model,
            args.loss,
            args.boundary_tolerance_frames,
            args.overlap_token_weight,
            args.stream_min_overlap,
            args.edge_tolerance_frames,
            clock,
            args.region_fallback,
            song_text,
        ),
        compute_metrics=metrics,
        processing_class=processor,
        callbacks=callbacks,
        optimizers=optimizers,
    )
    # Freeze the exact input manifests and window policy for checkpoint resume.
    identity = {
        "manifests": {
            name: file_sha256(args.data / name) for name in ("train.jsonl", "validation.jsonl", "vocab.json")
        },
        "window_seconds": args.window_seconds,
        "train_limit": args.train_limit,
        **({"window_policy": "snap", "window_snap": WINDOW_SNAP} if args.window_policy == "snap" else {}),
        "eval_limit": args.eval_limit,
        "eval_windows_per_track": args.eval_windows_per_track,
        "seed": args.seed,
        "window_order_version": 2,
        "frame_clock_version": 2,
        "batch_size": args.batch_size,
        "loss": args.loss,
        "boundary_tolerance_frames": args.boundary_tolerance_frames,
        "focal_frames": args.loss == "frame",
        **(
            {
                "v7": {
                    "overlap_loss_weight": args.overlap_loss_weight,
                    "overlap_mode": args.overlap_mode,
                    "two_stream": args.two_stream,
                    "stream2_loss_weight": args.stream2_loss_weight,
                    "stream2_blank_weight": args.stream2_blank_weight,
                    "stream_min_overlap": args.stream_min_overlap,
                    "stream2_encoder_grad_scale": args.stream2_encoder_grad_scale,
                    "stream2_blank_bias": args.stream2_blank_bias,
                    "distill_from": args.distill_from,
                    "distill_weight": args.distill_weight if args.distill_from else 0.0,
                    "overlap_token_weight": args.overlap_token_weight,
                    "freeze_encoder": args.freeze_encoder,
                    "overlap_encoder_grad_scale": args.overlap_encoder_grad_scale,
                    "learning_rate": args.learning_rate,
                    "overlap_frame_weight": args.overlap_frame_weight,
                    "augment": args.augment,
                    "dropout": args.dropout,
                    "layerdrop": args.layerdrop,
                    "spec_augment": spec_augment,
                }
            }
            if overlap or args.two_stream or args.augment or args.dropout or args.layerdrop or spec_augment
            else {}
        ),
        **(
            {
                "v1_large": {
                    "freeze_encoder_steps": args.freeze_encoder_steps,
                    "head_warmup_lr": args.head_warmup_lr,
                    "learning_rate": args.learning_rate,
                }
            }
            if args.freeze_encoder_steps
            else {}
        ),
        **({"fusion_lr": args.fusion_lr} if args.fusion_lr else {}),
        **({"fusion_lora_lr": lora_lr} if lora_lr else {}),
        **({"init_optimizer_from": args.init_optimizer_from} if args.init_optimizer_from else {}),
        **(
            {
                "loss_guidance": {
                    "text_encoder_grad_scale": args.text_encoder_grad_scale,
                    "loss_balance": args.loss_balance,
                    "alignment_weight": args.alignment_weight,
                    "background_balance": args.background_balance,
                    "text_fs_mode": args.text_fs_mode,
                }
            }
            if (
                args.text_encoder_grad_scale != 1.0
                or args.loss_balance != "none"
                or args.alignment_weight != 1.0
                or args.background_balance
                or args.text_fs_mode != "ctc"
            )
            else {}
        ),
        **(
            {
                "background_head": {
                    "weight": args.background_head_weight,
                    "blank_weight": args.background_blank_weight,
                    "blank_bias": args.background_blank_bias,
                    "lr": args.background_head_lr,
                }
            }
            if background
            else {}
        ),
        **({"region_fallback": args.region_fallback} if args.region_fallback != "proportional" else {}),
        **(
            {
                "text_aligner": {
                    "config": config.text_aligner,
                    "weight": args.text_aligner_weight,
                    "lr": args.text_aligner_lr,
                    "fs_weight": args.text_aligner_fs_weight,
                    "back_weight": args.text_aligner_back_weight,
                }
            }
            if text_head
            else {}
        ),
        **(
            {
                "s1": {
                    "edge_loss_weight": args.edge_loss_weight,
                    "edge_positive_weight": args.edge_positive_weight,
                    "edge_emd_weight": args.edge_emd_weight,
                    "edge_tolerance_frames": args.edge_tolerance_frames,
                }
            }
            if args.edge_loss_weight > 0
            else {}
        ),
        **({"fusion": config.fusion} if args.fusion == "muq" else {}),
        # Only a non-MMS backbone records its clock, so earlier identity files stay resumable.
        **(
            {
                "backbone": backbone,
                "frame_clock": {"stride_seconds": clock.stride_seconds, "offset_seconds": clock.offset_seconds},
            }
            if backbone != "wav2vec2"
            else {}
        ),
        "gradient_accumulation": args.gradient_accumulation,
        "world_size": train_args.world_size,
        "model_files": {
            str(p.relative_to(Path(args.model))): file_sha256(p)
            for p in sorted(Path(args.model).rglob("*"))
            if p.is_file() and p.suffix in {".json", ".bin", ".safetensors"}
        },
    }
    identity["training_recipe"] = {
        "version": 1,
        "max_steps": args.max_steps,
        "epochs": args.epochs,
        "warmup_steps": train_args.warmup_steps,
        "scheduler": args.lr_schedule,
        "cycles": 0.5,
        **({"decay_steps": args.decay_steps} if args.lr_schedule == "wsd" else {}),
        **({"text_qk_norm": True} if args.text_aligner_qk_norm else {}),
        **({"text_weight_decay": args.text_aligner_weight_decay} if args.text_aligner_weight_decay else {}),
        **({"mask_unlabelled_vocals": True} if args.mask_unlabelled_vocals else {}),
        "optimizer": "custom_fused_adamw" if optimizers[0] is not None else str(train_args.optim),
        "learning_rate": args.learning_rate,
        "betas": [train_args.adam_beta1, train_args.adam_beta2],
        "eps": train_args.adam_epsilon,
        "weight_decay": train_args.weight_decay,
        "freeze_encoder_steps": args.freeze_encoder_steps,
        "head_lr": args.head_warmup_lr,
        "fusion_lr": args.fusion_lr,
        "background_lr": args.background_head_lr if background else None,
        "lora_lr": lora_lr,
        "text_lr": text_lr,
        "max_grad_norm": train_args.max_grad_norm,
        "song_text_version": config.song_text_version if text_head else None,
    }
    identity_path = args.output / "data_identity.json"
    resume = get_last_checkpoint(str(args.output)) if args.resume else None
    validate_resume(identity_path, identity, args.resume, resume)
    if not args.resume and get_last_checkpoint(str(args.output)):
        raise ValueError("Output already contains checkpoints; use --resume or a new directory")
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model()
    processor.save_pretrained(args.output)


if __name__ == "__main__":
    main()
