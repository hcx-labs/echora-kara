#!/bin/sh
# The reference training run, fa-300m-muq-v3k-sung-lora32: the v3k-sung cont run's checkpoint-22000 with its MuQ
# LoRA grown from rank 8 (4 blocks, q/v) to rank 32 (8 blocks, q/v + FFN), on the v3b-sung labels (compact_s4_005),
# 70k steps, WSD schedule decaying over the last 20k.
#
# Run it with --resume to continue from the newest checkpoint in $RUNS/fa-300m-muq-v3k-sung-lora32. The trainer then
# restores model, optimizer, scheduler and RNG state and skips the batches already trained (same seed, same order),
# so training continues with the next unseen batch.
#
# Resume compares the run's recorded identity (data_identity.json) with these arguments and refuses on any
# difference. The identity holds the literal --init-optimizer-from path, the --model files' hashes, the manifests'
# hashes and world_size 1, so keep RUNS=/runs (mount or symlink it), the same manifests, and one GPU.
#
# Layout:
#   $DATASET/audio/<ISRC>.m4a
#   $DATASET/karaoke/forced_alignment_apple_100k_v3b_sung/compact_s4_005/{train,validation,test}.jsonl, vocab.json
#   $DATASET/karaoke/forced_alignment_apple_100k_v2/harness/heldout_clean_80.txt   (decoded-eval song ids)
#   /runs/fa-300m-muq-v3k-sung-cont/checkpoint-22000                              (--model / --init-optimizer-from)
#   /runs/fa-300m-muq-v3k-sung-lora32/{data_identity.json,checkpoint-34000}       (the run being resumed)
#
# Needs: pip install -e '.[train]', espeak-ng, ffmpeg, scripts/fetch_resources.sh, WANDB_API_KEY (or WANDB_MODE=offline),
# and the MuQ weights (OpenMuQ/MuQ-large-msd-iter, downloaded from the Hugging Face Hub on first use).
#
# Usage: DATASET=/datasets sh scripts/train_v3k_sung_lora32.sh --resume
#
# Longer schedule or more windows per step (a bigger GPU): set BATCH / ACCUM / MAX_STEPS / DECAY_STEPS and add
# --extend-schedule. Only those four may change; the data position carries over by window count, so training still
# continues with the next unseen window. Keep the learning rates. E.g. from checkpoint-34000, 16 windows per step,
# constant to 134k, then decay to 234k (about 8 more passes over the data):
#   DATASET=/datasets BATCH=16 ACCUM=1 MAX_STEPS=234000 DECAY_STEPS=100000 WORKERS=16 \
#     sh scripts/train_v3k_sung_lora32.sh --resume --extend-schedule
set -eu
: "${DATASET:?Set DATASET to the dataset root (audio/ and karaoke/ inside)}"
RUNS=/runs
BATCH=${BATCH:-4} ACCUM=${ACCUM:-2} MAX_STEPS=${MAX_STEPS:-70000} DECAY_STEPS=${DECAY_STEPS:-20000} WORKERS=${WORKERS:-6}
name=fa-300m-muq-v3k-sung-lora32
init=$RUNS/fa-300m-muq-v3k-sung-cont/checkpoint-22000
# The text head was trained on song-text v1 (carried over from the base checkpoint); allow it explicitly.
export FA_DIAG_ALLOW_SONG_TEXT_V1=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec python -u -m echora_kara.train.train \
  --data "$DATASET/karaoke/forced_alignment_apple_100k_v3b_sung/compact_s4_005" --root "$DATASET" \
  --model "$init" --init-optimizer-from "$init" --backbone wav2vec2 --output "$RUNS/$name" --run-name "$name" \
  --loss constrained_alignment --boundary-tolerance-frames 1 --window-seconds 25 --window-policy snap \
  --batch-size "$BATCH" --gradient-accumulation "$ACCUM" --learning-rate 8e-6 \
  --fusion-lr 8e-5 --warmup-steps 200 --freeze-encoder-steps 0 --head-warmup-lr 1e-5 \
  --max-steps "$MAX_STEPS" --lr-schedule wsd --decay-steps "$DECAY_STEPS" --eval-steps 2000 --save-steps 2000 \
  --save-total-limit 20 --workers "$WORKERS" --index-workers 16 \
  --edge-loss-weight 1.0 --edge-positive-weight 20 --edge-emd-weight 1.0 --edge-tolerance-frames 1 \
  --fusion muq --fusion-layer 11 --fusion-model OpenMuQ/MuQ-large-msd-iter --fusion-gate scalar \
  --fusion-lora-rank 32 --fusion-lora-alpha 64 --fusion-lora-layers 8 --fusion-lora-targets qv_ffn --fusion-lora-lr 8e-5 \
  --fusion-lora-grow \
  --region-fallback ignore \
  --background-head-weight 0.5 --background-blank-weight 0.2 --background-blank-bias 5.0 --background-head-lr 8e-4 \
  --song-text-version 1 --text-aligner-weight 0.5 --text-aligner-lr 1e-4 --text-aligner-dim 512 --text-aligner-heads 8 \
  --text-aligner-text-layers 6 --text-aligner-cross-layers 4 --text-aligner-fs-weight 0.02 --text-aligner-back-weight 0.5 \
  --text-aligner-qk-norm --text-aligner-weight-decay 0.05 \
  --decoded-eval-ids "$DATASET/karaoke/forced_alignment_apple_100k_v2/harness/heldout_clean_80.txt" --decoded-eval-songs 40 \
  --best-metric decoded_bonus_mean_ms \
  "$@"
