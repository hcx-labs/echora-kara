#!/usr/bin/env python3
"""Convert Meta's official omniASR CTC v2 checkpoints (fairseq2; 300M or 1B) into a Hugging Face
``Wav2Vec2ForCTC`` directory that the trainer and the aligner load unchanged.

Source: https://dl.fbaipublicfiles.com/mms/omniASR-CTC-300M-v2.pt or omniASR-CTC-1B-v2.pt (asset cards
``omniASR_CTC_300M_v2`` / ``omniASR_CTC_1B_v2`` in facebookresearch/omnilingual-asr, src/omnilingual_asr/cards/models/
rc_models_v2.yaml; Apache-2.0). Architecture: fairseq2 wav2vec2 large_lv60k layout
(conv feature extractor with per-layer LayerNorm and bias, PRE norm order, conv positional
encoder 128/16 with weight norm), 16 heads, vocab 10288; 300M is 24 layers x 1024 (FFN 4096),
1B is 48 layers x 1280 (FFN 5120).

Two output modes:
* ``--head full``   keeps the 10288-way CTC head (blank = <s> = id 0): parity testing only.
* ``--head echora`` (default) writes the aligner's 30-entry character vocabulary (``vocab.json`` of
  an existing Echora Kara checkpoint: | a-z ' [UNK] [PAD]; blank/pad = 29) and initialises its rows from the
  matching rows of the official head (a-z from the letter tokens, [PAD] from the CTC blank <s>,
  | from space, [UNK] from <unk>, ' from the apostrophe token). Tokenizer and feature-extractor
  files are copied from that checkpoint so ``Wav2Vec2Processor.from_pretrained`` works.

Nothing is downloaded here (needs torch + transformers), e.g.
  python -m echora_kara.train.convert_omniasr --source omniASR-CTC-300M-v2.pt \
      --official-tokenizer-json tokenizer.json --tokenizer-from <echora-kara checkpoint> --output omniasr-300m-init
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import torch
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

V8_SNAPSHOT = Path(".")  # an existing Echora Kara checkpoint directory (its tokenizer files are copied)
TOKENIZER_FILES = (
    "vocab.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "processor_config.json",
    "preprocessor_config.json",
    "added_tokens.json",
)
# fairseq2 (official checkpoint) -> transformers Wav2Vec2ForCTC parameter names.
KEY_MAP = [
    (
        r"^encoder_frontend\.feature_extractor\.layers\.(\d+)\.conv\.",
        r"wav2vec2.feature_extractor.conv_layers.\1.conv.",
    ),
    (
        r"^encoder_frontend\.feature_extractor\.layers\.(\d+)\.layer_norm\.",
        r"wav2vec2.feature_extractor.conv_layers.\1.layer_norm.",
    ),
    (r"^encoder_frontend\.post_extract_layer_norm\.", r"wav2vec2.feature_projection.layer_norm."),
    (r"^encoder_frontend\.model_dim_proj\.", r"wav2vec2.feature_projection.projection."),
    (r"^encoder_frontend\.pos_encoder\.conv\.bias$", r"wav2vec2.encoder.pos_conv_embed.conv.bias"),
    (
        r"^encoder_frontend\.pos_encoder\.conv\.weight_g$",
        r"wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0",
    ),
    (
        r"^encoder_frontend\.pos_encoder\.conv\.weight_v$",
        r"wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1",
    ),
    (r"^encoder\.layer_norm\.", r"wav2vec2.encoder.layer_norm."),
    (r"^encoder\.layers\.(\d+)\.self_attn\.q_proj\.", r"wav2vec2.encoder.layers.\1.attention.q_proj."),
    (r"^encoder\.layers\.(\d+)\.self_attn\.k_proj\.", r"wav2vec2.encoder.layers.\1.attention.k_proj."),
    (r"^encoder\.layers\.(\d+)\.self_attn\.v_proj\.", r"wav2vec2.encoder.layers.\1.attention.v_proj."),
    (r"^encoder\.layers\.(\d+)\.self_attn\.output_proj\.", r"wav2vec2.encoder.layers.\1.attention.out_proj."),
    (r"^encoder\.layers\.(\d+)\.self_attn_layer_norm\.", r"wav2vec2.encoder.layers.\1.layer_norm."),
    (r"^encoder\.layers\.(\d+)\.ffn\.inner_proj\.", r"wav2vec2.encoder.layers.\1.feed_forward.intermediate_dense."),
    (r"^encoder\.layers\.(\d+)\.ffn\.output_proj\.", r"wav2vec2.encoder.layers.\1.feed_forward.output_dense."),
    (r"^encoder\.layers\.(\d+)\.ffn_layer_norm\.", r"wav2vec2.encoder.layers.\1.final_layer_norm."),
    (r"^final_proj\.", r"lm_head."),
]
OFFICIAL_BLANK = 0  # fairseq2 CTC blank is <s> (index 0), not <pad> (1)
ECHORA_PAD = 29


def rename(key: str) -> str:
    """Map a fairseq2 parameter name to its ``Wav2Vec2ForCTC`` name via ``KEY_MAP``; ``KeyError`` if none match."""
    for pattern, replacement in KEY_MAP:
        new, count = re.subn(pattern, replacement, key)
        if count:
            return new
    raise KeyError(f"no mapping for official parameter {key}")


def encoder_dims(converted: dict[str, torch.Tensor]) -> dict[str, int]:
    """Layer count and widths read from the checkpoint (1B: 48 x 1280 / 5120; 300M: 24 x 1024 / 4096)."""
    layers = {int(m.group(1)) for key in converted if (m := re.match(r"wav2vec2\.encoder\.layers\.(\d+)\.", key))}
    hidden = converted["wav2vec2.encoder.layer_norm.weight"].shape[0]
    ffn = converted["wav2vec2.encoder.layers.0.feed_forward.intermediate_dense.weight"].shape[0]
    return {
        "hidden_size": hidden,
        "num_hidden_layers": len(layers),
        "intermediate_size": ffn,
        "num_attention_heads": 16,
    }  # omniASR 300M and 1B both use 16 heads


def hf_config(
    vocab_size: int, pad_token_id: int, mask_time_prob: float, mask_time_length: int, dims: dict[str, int]
) -> Wav2Vec2Config:
    """The ``Wav2Vec2Config`` for the omniASR large_lv60k layout (stable layer norm, conv positional encoder 128/16)."""
    return Wav2Vec2Config(
        vocab_size=vocab_size,
        **dims,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        conv_dim=[512] * 7,
        conv_kernel=[10, 3, 3, 3, 3, 2, 2],
        conv_stride=[5, 2, 2, 2, 2, 2, 2],
        conv_bias=True,
        feat_extract_norm="layer",
        do_stable_layer_norm=True,
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
        # Regularisation is a training decision (train.py overrides dropout/layerdrop and
        # SpecAugment); the stored defaults match the released aligner config so a run that passes no
        # overrides trains with its values.
        hidden_dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        feat_proj_dropout=0.0,
        final_dropout=0.0,
        layerdrop=0.0,
        mask_time_prob=mask_time_prob,
        mask_time_length=mask_time_length,
        mask_feature_prob=0.0,
        ctc_loss_reduction="sum",
        ctc_zero_infinity=True,
        pad_token_id=pad_token_id,
        bos_token_id=pad_token_id,
        eos_token_id=pad_token_id,
        apply_spec_augment=True,
    )


def official_vocab(tokenizer_json: Path) -> dict[str, int]:
    """Piece -> id of the official tokenizer, from a ``tokenizer.json`` with a dict or ``[piece, score]`` list vocab."""
    data = json.loads(tokenizer_json.read_text())
    vocab = data["model"]["vocab"]
    return vocab if isinstance(vocab, dict) else {piece: index for index, (piece, _) in enumerate(vocab)}


def echora_head(weight: torch.Tensor, bias: torch.Tensor, vocab: dict[str, int], source_vocab: dict[str, int]):
    """Rows of the aligner's vocabulary taken from the official head."""
    rows = {"[PAD]": "<s>", "[UNK]": "<unk>", "|": " "}
    new_weight = weight.new_zeros((len(vocab), weight.shape[1]))
    new_bias = bias.new_zeros(len(vocab))
    for token, index in vocab.items():
        source = rows.get(token, token)
        if source not in source_vocab:
            raise KeyError(f"official vocabulary lacks {source!r} for aligner token {token!r}")
        new_weight[index] = weight[source_vocab[source]]
        new_bias[index] = bias[source_vocab[source]]
    # The official blank is <s>; the copied bias keeps blank dominant, as for the official head.
    return new_weight, new_bias


def main() -> None:
    """Convert an official omniASR CTC v2 checkpoint into a new, empty output directory.

    Writes the safetensors model and config, the Echora tokenizer files (``--head echora``) and ``CONVERSION.json``
    recording the source SHA-256, encoder dimensions and key map.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="official omniASR CTC v2 .pt (300M or 1B)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head", choices=("echora", "full"), default="echora")
    parser.add_argument(
        "--official-tokenizer-json",
        type=Path,
        default=Path("tokenizer.json"),
        help="tokenizer.json listing the official 10288 pieces (any faithful copy; ids are checked)",
    )
    parser.add_argument(
        "--tokenizer-from",
        dest="v8_snapshot",
        type=Path,
        default=V8_SNAPSHOT,
        help="an existing Echora Kara checkpoint directory: source of vocab.json and the tokenizer files",
    )
    parser.add_argument(
        "--mask-time-prob", type=float, default=0.075, help="stored default (the released aligner config uses 0.075)"
    )
    parser.add_argument("--mask-time-length", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {args.output}")
    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    checkpoint = torch.load(args.source, map_location="cpu", mmap=True, weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    converted = {rename(key): tensor.float().clone() for key, tensor in state.items()}
    weight, bias = converted.pop("lm_head.weight"), converted.pop("lm_head.bias")
    dims = encoder_dims(converted)
    if tuple(weight.shape) != (10288, dims["hidden_size"]):
        raise SystemExit(f"unexpected official head shape {tuple(weight.shape)}")
    print(json.dumps({"encoder": dims}))
    if args.head == "full":
        config = hf_config(10288, OFFICIAL_BLANK, args.mask_time_prob, args.mask_time_length, dims)
        converted["lm_head.weight"], converted["lm_head.bias"] = weight, bias
    else:
        source_vocab = official_vocab(args.official_tokenizer_json)
        for token, expected in (("<s>", 0), ("<pad>", 1), ("</s>", 2), ("<unk>", 3), (" ", 4)):
            if source_vocab.get(token) != expected:
                raise SystemExit(
                    f"official tokenizer order differs from the omniASR_tokenizer_written_v2 layout at {token!r}"
                )
        vocab = json.loads((args.v8_snapshot / "vocab.json").read_text())
        if vocab.get("[PAD]") != ECHORA_PAD or len(vocab) != 30:
            raise SystemExit("unexpected aligner vocabulary")
        config = hf_config(len(vocab), ECHORA_PAD, args.mask_time_prob, args.mask_time_length, dims)
        converted["lm_head.weight"], converted["lm_head.bias"] = echora_head(weight, bias, vocab, source_vocab)
    model = Wav2Vec2ForCTC(config)
    missing, unexpected = model.load_state_dict(converted, strict=False)
    if unexpected or any(not key.endswith("masked_spec_embed") for key in missing):
        raise SystemExit(f"conversion incomplete: missing {missing} unexpected {unexpected}")
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    if args.head == "echora":
        for name in TOKENIZER_FILES:
            if (args.v8_snapshot / name).exists():
                shutil.copy(args.v8_snapshot / name, args.output / name)
    (args.output / "CONVERSION.json").write_text(
        json.dumps(
            {
                "source": str(args.source),
                "source_sha256": digest,
                "head": args.head,
                "official_asset": f"https://dl.fbaipublicfiles.com/mms/{args.source.name}",
                "encoder": dims,
                "official_blank_id": OFFICIAL_BLANK,
                "pad_token_id": config.pad_token_id,
                "vocab_size": config.vocab_size,
                "tensors": len(converted),
                "key_map": [pattern for pattern, _ in KEY_MAP],
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "tensors": len(converted),
                "vocab_size": config.vocab_size,
                "source_sha256": digest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
