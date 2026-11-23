#!/bin/sh
# The recipe of Echora Kara 700M MuQ (hcxlabs/echora-kara-700m-muq, step 64,000 of this run): one clean run from
# the omniASR CTC 300M v2 conversion, the same as scripts/train_clean100k_dm.sh without the decoder margin, and with
# the MuQ gate fixed at 0.1 for the whole run.
#
# Why the gate is fixed: with a trainable gate that starts at zero, the gate never opened (about 1e-5 after 74,000
# steps), so MuQ contributed nothing. Fixed open, the projection and LoRA have to become useful. --fusion-gentle-warmup
# keeps the LoRA untrained and the projection at --fusion-lr while the encoder is frozen: at --head-warmup-lr the LoRA
# flattened MuQ's features to a near constant within a few hundred steps.
#
# GPUS=N (1, 2, 4 or 8) runs N torchrun processes and keeps 8 windows per optimizer step, as in
# scripts/train_clean100k_dm.sh, whose header also lists the dataset layout and requirements. MuQ also needs
# pip install -e '.[fusion]'.
#
# Usage: DATASET=/datasets GPUS=8 sh scripts/train_700m_muq.sh --no-gradient-checkpointing
#        (resume: the same command with --resume)
set -eu
: "${DATASET:?Set DATASET to the dataset root (audio/ and karaoke/ inside)}"
RUNS=/runs
GPUS=${GPUS:-1} WORKERS=${WORKERS:-6}
name=fa-300m-muq-clean100k-gate01
init=$RUNS/omniasr-ctc-300m-v2-fa-kara-init
case $GPUS in
  1) launch="python -u -m echora_kara.train.train"; batch="--batch-size 4 --gradient-accumulation 2" ;;
  2|4|8) launch="python -u -m torch.distributed.run --nproc_per_node $GPUS -m echora_kara.train.train"
         batch="--batch-size $((8 / GPUS)) --gradient-accumulation 1" ;;
  *) echo "GPUS must be 1, 2, 4 or 8 (8 windows per step)" >&2; exit 1 ;;
esac
# The text head is trained on song-text v1 (v2 diverged in earlier runs); allow it explicitly.
export FA_DIAG_ALLOW_SONG_TEXT_V1=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# shellcheck disable=SC2086  # $launch and $batch are word lists
exec $launch \
  --data "$DATASET/karaoke/forced_alignment_apple_100k_v3b_sung/compact_s4_005" --root "$DATASET" \
  --model "$init" --backbone wav2vec2 --output "$RUNS/$name" --run-name "$name" \
  --loss constrained_alignment --boundary-tolerance-frames 1 --window-seconds 25 --window-policy snap \
  $batch --learning-rate 1e-5 \
  --fusion-lr 1e-4 --warmup-steps 1000 --freeze-encoder-steps 1000 --head-warmup-lr 1e-3 \
  --max-steps 100000 --lr-schedule cosine --eval-steps 2000 --save-steps 2000 --save-total-limit 20 \
  --workers "$WORKERS" --index-workers 16 \
  --edge-loss-weight 1.0 --edge-positive-weight 20 --edge-emd-weight 1.0 --edge-tolerance-frames 1 \
  --fusion muq --fusion-layer 11 --fusion-model OpenMuQ/MuQ-large-msd-iter --fusion-gate scalar \
  --fusion-gate-init 0.1 --fusion-gate-hold-steps 100000 --fusion-gentle-warmup \
  --fusion-lora-rank 32 --fusion-lora-alpha 64 --fusion-lora-layers 8 --fusion-lora-targets qv_ffn --fusion-lora-lr 1e-4 \
  --region-fallback ignore \
  --background-head-weight 0.5 --background-blank-weight 0.2 --background-blank-bias 5.0 --background-head-lr 5e-4 \
  --song-text-version 1 --text-aligner-weight 0.5 --text-aligner-lr 1e-4 --text-aligner-dim 512 --text-aligner-heads 8 \
  --text-aligner-text-layers 6 --text-aligner-cross-layers 4 --text-aligner-fs-weight 0.02 --text-aligner-back-weight 0.5 \
  --text-aligner-qk-norm --text-aligner-weight-decay 0.05 \
  --decoded-eval-ids "$DATASET/karaoke/forced_alignment_apple_100k_v2/harness/heldout_clean_80.txt" --decoded-eval-songs 40 \
  --best-metric decoded_bonus_mean_ms \
  "$@"
