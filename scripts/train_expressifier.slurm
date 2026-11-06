#!/bin/bash
#SBATCH -J expressifier
#SBATCH -A hai-res
#SBATCH -p hai-res
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 23:00:00
#SBATCH --qos=hai-res-main
#SBATCH -o /data/scratch-oc40/joel/logs/expressifier_train_%j.log
#SBATCH -e /data/scratch-oc40/joel/logs/expressifier_train_%j.log
#SBATCH --export=ALL

# Train the expressifier (bidirectional dense-AMT-medium score encoder + causal performance decoder).
#   STAGE=1 sbatch --job-name=expressifier_s1 .../scripts/train_expressifier.slurm
#   STAGE=2 sbatch --job-name=expressifier_s2 .../scripts/train_expressifier.slurm   # from stage-1 best
#   STAGE=1 DRYRUN=1 sbatch .../scripts/train_expressifier.slurm
#   STAGE=1 RESUME_FROM=.../expressifier_stage1/checkpoint-XXXX sbatch ...
# Options: DATA_ROOT, OUTPUT_DIR, INIT_FROM (stage 2; 'none' = fresh), ENCODER_CHECKPOINT, DECODER_LAYERS.

set -x
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$(readlink -f "$0")")/expressifier_env.sh" 2>/dev/null \
    || source /data/scratch-oc40/joel/orchestration_v2/scripts/expressifier_env.sh

extra_args=(--stage "${STAGE:?set STAGE=1 or 2}")
[ -n "${DATA_ROOT:-}" ] && extra_args+=(--data-root "${DATA_ROOT}")
[ -n "${OUTPUT_DIR:-}" ] && extra_args+=(--output-dir "${OUTPUT_DIR}")
[ -n "${INIT_FROM:-}" ] && extra_args+=(--init-from "${INIT_FROM}")
[ -n "${ENCODER_CHECKPOINT:-}" ] && extra_args+=(--encoder-checkpoint "${ENCODER_CHECKPOINT}")
[ -n "${DECODER_LAYERS:-}" ] && extra_args+=(--decoder-layers "${DECODER_LAYERS}")
[ -n "${RESUME_FROM:-}" ] && extra_args+=(--resume-from-checkpoint "${RESUME_FROM}")
[ "${DRYRUN:-0}" = "1" ] && extra_args+=(--dryrun)

unset PET_REDIRECTS PET_TEE PET_LOG_DIR PET_LOCAL_RANKS_FILTER 2>/dev/null || true
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
NPROC="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-2}}"
MASTER_PORT=$((15000 + SLURM_JOB_ID % 20000))

echo "=== expressifier stage=${STAGE} nproc=${NPROC} dryrun=${DRYRUN:-0} args=${extra_args[*]} ==="
srun torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" --master_addr=127.0.0.1 \
    --master_port="${MASTER_PORT}" -m orchestration_v2.train.expressifier_train "${extra_args[@]}"
echo "expressifier training finished."
