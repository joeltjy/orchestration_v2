#!/bin/bash
#SBATCH -J expressifier_pairs
#SBATCH -A hai-res
#SBATCH -p hai-res
#SBATCH --array=0-17
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --qos=hai-res-extended
#SBATCH -o /data/scratch-oc40/joel/logs/expressifier_pairs_%A_%a.log
#SBATCH -e /data/scratch-oc40/joel/logs/expressifier_pairs_%A_%a.log
#SBATCH --export=ALL

# Stage-1 expressifier pairs: one array task per PianoCoRe parquet shard (18 shards). Each task saves a batch
# of pairs as it goes (resumable: resubmit to continue). After all tasks finish, merge:
#   sbatch .../scripts/preprocess_expressifier_pairs.slurm
#   MERGE_ONLY=1 sbatch --array=0 .../scripts/preprocess_expressifier_pairs.slurm

set -x
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$(readlink -f "$0")")/expressifier_env.sh" 2>/dev/null \
    || source /data/scratch-oc40/joel/orchestration_v2/scripts/expressifier_env.sh

args=(--num-workers "${SLURM_CPUS_PER_TASK:-16}")
if [ "${MERGE_ONLY:-0}" = "1" ]; then
    args+=(--merge-only)
else
    args+=(--shard-index "${SLURM_ARRAY_TASK_ID}")
fi
python -u -m orchestration_v2.preprocessing.expressifier_pairs "${args[@]}"
