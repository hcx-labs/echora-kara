#!/bin/sh
# One clean run with the decoder margin, fa-300m-muq-clean100k-dm: everything trained from the omniASR CTC 300M v2
# conversion in a single schedule (no carried optimizer, no data replay): v3b-sung labels, MuQ fusion + LoRA rank 32
# (8 blocks, q/v + FFN), background head and text aligner (QK-norm, weight decay) from step 0. 1000 steps with the
# encoder frozen (fresh heads at --head-warmup-lr), a 1000-step warm-up, then one cosine decay to zero at 100k.
#
# On top: the decoder margin (train/decoder_margin.py). Every window runs the production decoder (vocal-activity
# penalty, plain pass, paced floors, blank-free fragments, in-line gap cost, text bonus) and its pick, charged
# 0.1 nats per 20 ms of start error (1 frame free, capped at 500 ms), is pushed below the measured path. Weight 0.5,
# off for the first 3000 steps (the decoder's pick on an untrained model is noise), then ramped to full over 2000.
# The same command without the three --decoder-margin-* flags is the baseline it is compared against.
#
# GPUS=N (1, 2, 4 or 8) runs N torchrun processes and keeps 8 windows per optimizer step (8/N per GPU, no
# accumulation), so the recipe and step count do not depend on N; only wall-clock does. world_size is part of the
# recorded identity: resume with the same N. Add --no-gradient-checkpointing on 80 GB GPUs.
#
# Layout:
#   $DATASET/audio/<ISRC>.m4a
#   $DATASET/karaoke/forced_alignment_apple_100k_v3b_sung/compact_s4_005/{train,validation,test}.jsonl, vocab.json
#   $DATASET/karaoke/forced_alignment_apple_100k_v2/harness/heldout_clean_80.txt   (decoded-eval song ids)
#   /runs/omniasr-ctc-300m-v2-fa-kara-init                                         (train/convert_omniasr.py output)
#
# Needs: pip install -e '.[train]', espeak-ng, ffmpeg, scripts/fetch_resources.sh, WANDB_API_KEY (or WANDB_MODE=offline),
# and the MuQ weights (OpenMuQ/MuQ-large-msd-iter, downloaded from the Hugging Face Hub on first use).
#
# Usage: DATASET=/datasets GPUS=8 sh scripts/train_clean100k_dm.sh --no-gradient-checkpointing
#        (resume: the same command with --resume)
set -eu
: "${DATASET:?Set DATASET to the dataset root (audio/ and karaoke/ inside)}"
RUNS=/runs
GPUS=${GPUS:-1} WORKERS=${WORKERS:-6}
name=fa-300m-muq-clean100k-dm
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
  --fusion-lora-rank 32 --fusion-lora-alpha 64 --fusion-lora-layers 8 --fusion-lora-targets qv_ffn --fusion-lora-lr 1e-4 \
  --region-fallback ignore \
  --background-head-weight 0.5 --background-blank-weight 0.2 --background-blank-bias 5.0 --background-head-lr 5e-4 \
  --song-text-version 1 --text-aligner-weight 0.5 --text-aligner-lr 1e-4 --text-aligner-dim 512 --text-aligner-heads 8 \
  --text-aligner-text-layers 6 --text-aligner-cross-layers 4 --text-aligner-fs-weight 0.02 --text-aligner-back-weight 0.5 \
  --text-aligner-qk-norm --text-aligner-weight-decay 0.05 \
  --decoded-eval-ids "$DATASET/karaoke/forced_alignment_apple_100k_v2/harness/heldout_clean_80.txt" --decoded-eval-songs 40 \
  --best-metric decoded_bonus_mean_ms \
  --decoder-margin-weight 0.5 --decoder-margin-start-step 3000 --decoder-margin-ramp-steps 2000 \
  "$@"
