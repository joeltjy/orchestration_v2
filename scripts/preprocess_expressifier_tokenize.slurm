#!/bin/bash
#SBATCH -J expressifier_tokenize
#SBATCH -A hai-res
#SBATCH -p hai-res
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --qos=hai-res-extended
#SBATCH -o /data/scratch-oc40/joel/logs/expressifier_tokenize_%j.log
#SBATCH -e /data/scratch-oc40/joel/logs/expressifier_tokenize_%j.log
#SBATCH --export=ALL

# Chunk + tokenize expressifier pairs (saves shard by shard; resubmit to continue).
#   STAGE=1 sbatch .../scripts/preprocess_expressifier_tokenize.slurm   # PianoCoRe A*, no augmentation
#   STAGE=2 sbatch .../scripts/preprocess_expressifier_tokenize.slurm   # aligned MAESTRO crops, 16x aug
#   STAGE=2 AUG=32 OUTPUT_DIR=... sbatch ...                            # more augmentation

set -x
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$(readlink -f "$0")")/expressifier_env.sh" 2>/dev/null \
    || source /data/scratch-oc40/joel/orchestration_v2/scripts/expressifier_env.sh

args=(--stage "${STAGE:?set STAGE=1 or 2}")
[ -n "${AUG:-}" ] && args+=(--augmentation-factor "${AUG}")
[ -n "${OUTPUT_DIR:-}" ] && args+=(--output-dir "${OUTPUT_DIR}")
[ -n "${SOURCE:-}" ] && args+=(--source "${SOURCE}")
python -u -m orchestration_v2.preprocessing.expressifier_tokenize "${args[@]}"
