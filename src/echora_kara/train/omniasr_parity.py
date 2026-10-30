#!/usr/bin/env python3
"""Parity test for convert_omniasr.py: an official fairseq2 omniASR CTC v2 model
and our Hugging Face conversion must produce the same emissions on the same audio.

Three checks, each printed and asserted:
1. Tensor parity: every converted tensor equals the official tensor it came from (exact), and,
   as a cross-check, equals the independent community conversion (bezzam/omniasr-ctc-1b-v2,
   transformers' native OmniASR port) where that is present (weight-norm g/v recomposed).
2. Emission parity: the official fairseq2 model (built from fairseq2's own Wav2Vec2AsrConfig with
   the "1b_v2" values from omnilingual-asr) and the converted Wav2Vec2ForCTC get identical
   pre-normalised 16 kHz input; log-softmax outputs must agree within --tolerance (fp32, same GPU).
3. Decode parity: greedy CTC collapse of both outputs gives the same token ids and text.

The fairseq2 model needs ``fairseq2`` importable (pip install fairseq2==0.8.1 fairseq2n==0.8.1 in
a throwaway container; fairseq2n must match the torch build). Without it, checks 1 and 3-on-HF
run and the script exits 2 so the gap is visible.

usage: python -m echora_kara.train.omniasr_parity --official omniASR-CTC-1B-v2.pt \
    --converted omniasr-ctc-1b-v2-full --community omniasr-ctc-1b-v2-hf \
    --audio <vocal stem>.npy [--seconds 20]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import Wav2Vec2ForCTC

from echora_kara.train.convert_omniasr import rename  # noqa: E402

SR = 16000


def community_key(key: str) -> str:
    """bezzam/omniasr-ctc-1b-v2 (OmniASRForCTC) parameter name -> Wav2Vec2ForCTC name."""
    k = key.replace("encoder.subsampling.conv_layers.", "wav2vec2.feature_extractor.conv_layers.")
    k = k.replace("encoder.subsampling.layer_norm.", "wav2vec2.feature_projection.layer_norm.")
    k = k.replace("encoder.subsampling.projection.", "wav2vec2.feature_projection.projection.")
    k = k.replace("encoder.encode_positions.conv.", "wav2vec2.encoder.pos_conv_embed.conv.")
    if k.startswith("encoder.layer_norm.") or k.startswith("encoder.layers."):
        k = "wav2vec2." + k
    return k.replace("ctc_head.", "lm_head.")


def load_converted(directory: Path) -> dict[str, torch.Tensor]:
    """All tensors of the ``*.safetensors`` files in a converted model directory."""
    out = {}
    for file in sorted(directory.glob("*.safetensors")):
        with safe_open(str(file), "pt") as handle:
            for key in handle.keys():
                out[key] = handle.get_tensor(key)
    return out


def tensor_parity(official: dict, converted: dict, community: Path | None) -> None:
    """Check 1: assert every official tensor equals its converted counterpart exactly, then cross-check against the
    community conversion when its ``model.safetensors`` is present (positional-conv weight norm recomposed).
    """
    worst = 0.0
    for key, tensor in official.items():
        ours = converted[rename(key)]
        if key.startswith("final_proj.") and ours.shape != tensor.shape:
            continue  # echora-head conversions re-map rows; compare only full-vocab conversions
        worst = max(worst, float((ours.float() - tensor.float()).abs().max()))
    print(f"official -> converted: {len(official)} tensors, max |diff| {worst:.3e}")
    assert worst == 0.0, "converted tensors differ from the official checkpoint"
    if community is None or not (community / "model.safetensors").exists():
        print("community cross-check skipped (no checkpoint given)")
        return
    merged = None
    worst, compared = 0.0, 0
    with safe_open(str(community / "model.safetensors"), "pt") as handle:
        for key in handle.keys():
            tensor, target = handle.get_tensor(key), community_key(key)
            if target == "wav2vec2.encoder.pos_conv_embed.conv.weight":
                merged = tensor
                continue
            if target not in converted or converted[target].shape != tensor.shape:
                continue
            worst = max(worst, float((converted[target].float() - tensor.float()).abs().max()))
            compared += 1
    if merged is not None:
        g = converted["wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0"].float()
        v = converted["wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1"].float()
        recomposed = g * v / v.norm(dim=(0, 1), keepdim=True)
        worst_pos = float((recomposed - merged.float()).abs().max())
        print(
            f"community cross-check: {compared} tensors max |diff| {worst:.3e}; pos-conv recomposed max |diff| {worst_pos:.3e}"
        )
        assert worst_pos < 1e-5, "positional conv differs from the community conversion"
    else:
        print(f"community cross-check: {compared} tensors max |diff| {worst:.3e}")
    assert worst == 0.0, "weights differ from the community conversion"


def build_fairseq2(official: dict, device):
    """The official fairseq2 ASR model with ``official`` loaded strictly, in eval mode on ``device``."""
    from fairseq2.models.transformer import TransformerNormOrder
    from fairseq2.models.wav2vec2 import Wav2Vec2EncoderConfig
    from fairseq2.models.wav2vec2.asr import Wav2Vec2AsrConfig, create_wav2vec2_asr_model

    # omnilingual-asr large_lv60k encoder layout; sizes read from the checkpoint
    # ("1b_v2": 48 x 1280, FFN 5120; "300m_v2": 24 x 1024, FFN 4096), vocab 10288.
    import re

    layers = {int(m.group(1)) for key in official if (m := re.search(r"encoder\.layers\.(\d+)\.", key))}
    model_dim = next(t for k, t in official.items() if k.endswith("final_proj.weight")).shape[1]
    ffn_dim = next(t for k, t in official.items() if re.search(r"layers\.0\.ffn\.inner_proj\.weight$", k)).shape[0]
    print(f"fairseq2 encoder: {len(layers)} x {model_dim}, FFN {ffn_dim}")
    encoder = Wav2Vec2EncoderConfig(
        model_dim=model_dim,
        max_seq_len=4096,
        feature_dim=512,
        use_fbank=False,
        first_pass_dropout_p=0.0,
        layer_norm_features=False,
        feature_extractor_layer_descs=[
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        feature_extractor_bias=True,
        feature_extractor_layer_norm_convs=True,
        feature_grad_scale=1.0,
        pos_encoder_type="conv",
        pos_encoder_depth=1,
        pos_conv_kernel_size=128,
        num_pos_conv_groups=16,
        use_conformer=False,
        num_encoder_layers=len(layers),
        num_encoder_attn_heads=16,
        ffn_inner_dim=ffn_dim,
        dropout_p=0.0,
        attn_dropout_p=0.0,
        ffn_inner_dropout_p=0.1,
        layer_drop_p=0.1,
        norm_order=TransformerNormOrder.PRE,
    )
    config = Wav2Vec2AsrConfig(encoder_config=encoder, target_vocab_size=10288, final_dropout_p=0.0, use_masking=False)
    model = create_wav2vec2_asr_model(config)
    missing, unexpected = model.load_state_dict(official, strict=False)
    assert not unexpected and not missing, f"fairseq2 model mismatch: missing {missing} unexpected {unexpected}"
    return model.to(device).eval()


@torch.no_grad()
def fairseq2_logits(model, seqs: torch.Tensor) -> torch.Tensor:
    """Call the fairseq2 ASR model across its API generations."""
    try:
        from fairseq2.nn import BatchLayout

        output = model(seqs, BatchLayout.of(seqs))
    except ImportError:
        from fairseq2.nn.padding import PaddingMask  # fairseq2 <= 0.4
        from fairseq2.models.sequence import SequenceBatch

        output = model(
            SequenceBatch(seqs, PaddingMask(torch.tensor([seqs.shape[1]], device=seqs.device), seqs.shape[1]))
        )
    logits = output.logits if hasattr(output, "logits") else output[0]
    return logits.float()


def greedy(log_probs: torch.Tensor, blank: int) -> list[int]:
    """Greedy CTC decode of ``[T, V]`` log-probs: frame argmax with repeats collapsed and blanks removed."""
    ids, out, prev = log_probs.argmax(-1).tolist(), [], None
    for i in ids:
        if i != blank and i != prev:
            out.append(i)
        prev = i
    return out


def main() -> int:
    """Run the three parity checks; return 0 when all pass, 2 when fairseq2 is unavailable (failures assert)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True, help="full-vocab conversion directory")
    parser.add_argument("--community", type=Path, default=None)
    parser.add_argument("--audio", type=Path, required=True, help="float32 .npy at 16 kHz (vocal stem)")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--tolerance", type=float, default=1e-3, help="max |delta log-softmax| allowed")
    parser.add_argument("--tokenizer-json", type=Path, default=None, help="official piece list, for the decoded text")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.official, map_location="cpu", mmap=True, weights_only=False)
    official = checkpoint["model"] if "model" in checkpoint else checkpoint
    converted = load_converted(args.converted)
    tensor_parity(official, converted, args.community)

    clip = np.load(args.audio).astype(np.float32)[: int(args.seconds * SR)]
    x = torch.from_numpy((clip - clip.mean()) / (clip.std() + 1e-7))[None].to(device)
    hf = Wav2Vec2ForCTC.from_pretrained(args.converted, local_files_only=True).to(device).eval()
    with torch.no_grad():
        hf_logits = hf(x).logits.float()
    hf_lp = torch.log_softmax(hf_logits, -1)[0]
    blank = hf.config.pad_token_id
    pieces = None
    if args.tokenizer_json and args.tokenizer_json.exists():
        vocab = json.loads(args.tokenizer_json.read_text())["model"]["vocab"]
        pieces = {
            i: p for p, i in (vocab.items() if isinstance(vocab, dict) else ((p, i) for i, (p, _) in enumerate(vocab)))
        }
    hf_ids = greedy(hf_lp, blank)
    if pieces:
        print("HF greedy:", "".join(pieces.get(i, "?") for i in hf_ids)[:200])
    try:
        import fairseq2  # noqa: F401
    except ImportError as error:
        print(f"fairseq2 not importable ({error}); emission parity NOT verified")
        return 2
    reference = build_fairseq2(official, device)
    ref_logits = fairseq2_logits(reference, x)
    frames = min(ref_logits.shape[1], hf_logits.shape[1])
    assert ref_logits.shape[1] == hf_logits.shape[1], (
        f"frame counts differ: fairseq2 {ref_logits.shape[1]} vs HF {hf_logits.shape[1]}"
    )
    ref_lp = torch.log_softmax(ref_logits, -1)[0, :frames]
    delta = (ref_lp - hf_lp[:frames]).abs()
    print(
        f"emission parity: frames {frames}, max |delta log-softmax| {float(delta.max()):.3e}, mean {float(delta.mean()):.3e}"
    )
    ref_ids = greedy(ref_lp, blank)
    same_argmax = float((ref_lp.argmax(-1) == hf_lp[:frames].argmax(-1)).float().mean())
    print(f"decode parity: frame argmax agreement {same_argmax:.5f}; greedy ids equal: {ref_ids == hf_ids}")
    if pieces:
        print("fairseq2 greedy:", "".join(pieces.get(i, "?") for i in ref_ids)[:200])
    assert float(delta.max()) <= args.tolerance, "emissions differ beyond tolerance"
    assert ref_ids == hf_ids, "greedy decodes differ"
    print("PARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
