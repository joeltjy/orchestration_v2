#!/bin/bash
#SBATCH -J expressify_snet
#SBATCH -A hai-res
#SBATCH -p hai-res-l40s
#SBATCH --qos=hai-res-main
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 23:00:00
#SBATCH --exclude=huang-l40s-2
#SBATCH -o /data/scratch-oc40/joel/logs/expressify_symphonynet_%j.log
#SBATCH -e /data/scratch-oc40/joel/logs/expressify_symphonynet_%j.log
#SBATCH --export=ALL

# Expressify the piano parts of SymphonyNet's classical piano concertos (EXPRESSIFIER.md).
#   sbatch scripts/expressify_symphonynet.slurm                 # all pieces
#   LIMIT=2 OUT_TAG=dryrun sbatch scripts/expressify_symphonynet.slurm
# Pass variables as env (LIMIT=2 sbatch ...), not via sbatch --export=ALL,LIMIT=2 (that gets jobs cancelled at
# start on this cluster). Reruns skip pieces whose parquet already exists.
# Options: CHECKPOINT, OUT_TAG, LIMIT, BATCH_SIZE, NUM_SAMPLES, TEMPERATURE, TOP_P, SEED.

set -x
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$(readlink -f "$0")")/expressifier_env.sh" 2>/dev/null \
    || source /data/scratch-oc40/joel/orchestration_v2/scripts/expressifier_env.sh

CHECKPOINT="${CHECKPOINT:-${SCRATCH_ROOT}/checkpoints/expressifier_stage2/best}"
OUT_TAG="${OUT_TAG:-expressifier_stage2_best}"
OUT="${SCRATCH_ROOT}/expressified/symphonynet_piano_concertos/${OUT_TAG}"
args=(--checkpoint "${CHECKPOINT}" --output-dir "${OUT}" --batch-size "${BATCH_SIZE:-32}"
      --num-samples "${NUM_SAMPLES:-1}" --temperature "${TEMPERATURE:-1.0}" --top-p "${TOP_P:-0.95}"
      --seed "${SEED:-0}")
[ -n "${LIMIT:-}" ] && args+=(--limit "${LIMIT}")

python -u -m orchestration_v2.sampling.expressify_symphonynet "${args[@]}"
echo "expressify finished."
