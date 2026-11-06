#!/bin/bash
# Shared environment for the expressifier slurm scripts (sourced, not run).
# Same conda env, scratch layout and wandb setup as ~/orchestration's train scripts.

OC40_ENV=/data/scratch-oc40/joel/conda_envs/experiments
if [ -f "$OC40_ENV/.copy_complete" ]; then
    # scratch-oc40 copy of the experiments env (the huang nodes cannot see /data/scratch since 2026-10-19)
    export CONDA_PREFIX="$OC40_ENV" CONDA_DEFAULT_ENV="$OC40_ENV" PATH="$OC40_ENV/bin:$PATH"
else
    for CONDA_SH in /opt/miniconda3/etc/profile.d/conda.sh /data/scratch/joeltjy1/miniconda3/etc/profile.d/conda.sh; do
        [ -f "$CONDA_SH" ] && break
    done
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    for EXPERIMENTS_ENV in /data/scratch/joeltjy1/miniconda3/envs/experiments /data/scratch/joeltjy1/conda_envs/experiments; do
        [ -x "$EXPERIMENTS_ENV/bin/python" ] && break
    done
    conda activate "$EXPERIMENTS_ENV"
fi

export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
SCRATCH_ROOT="${SCRATCH_ROOT:-/data/scratch-oc40/joel}"
REPO_ROOT="${REPO_ROOT:-${SCRATCH_ROOT}/orchestration_v2}"
if [ ! -f "${REPO_ROOT}/src/orchestration_v2/constants.py" ]; then
    echo >&2 "FATAL: ${REPO_ROOT}/src/orchestration_v2 not found (pull orchestration_v2 on scratch first)"
    exit 1
fi
unset PYTHONPATH
export PYTHONPATH="${REPO_ROOT}/src"
cd "${REPO_ROOT}"

mkdir -p "${SCRATCH_ROOT}/logs" "${SCRATCH_ROOT}/tmp" "${SCRATCH_ROOT}/checkpoints"
export TMPDIR="${SCRATCH_ROOT}/tmp"
export HF_HOME="${SCRATCH_ROOT}/huggingface"
export HF_HUB_CACHE="${SCRATCH_ROOT}/huggingface/hub"
export HF_DATASETS_CACHE="${SCRATCH_ROOT}/huggingface/datasets"

WANDB_SECRETS="${SCRATCH_ROOT}/secrets/wandb_api_key"
if [[ -r "${WANDB_SECRETS}" ]]; then
    WANDB_API_KEY="$(cat "${WANDB_SECRETS}")"
    export WANDB_API_KEY
    export WANDB_ENTITY="${WANDB_ENTITY:-joeltjy17-massachusetts-institute-of-technology}"
    export WANDB_NAME="${WANDB_NAME:-orchestration_v2_${SLURM_JOB_NAME:-local}_${SLURM_JOB_ID:-0}}"
    export WANDB_RUN_ID="${WANDB_RUN_ID:-${SLURM_JOB_ID:-}}"
    export WANDB_DIR="${SCRATCH_ROOT}/wandb"
    export WANDB_START_METHOD="thread"
    export WANDB_SILENT="true"
    export WANDB_MODE="${WANDB_MODE:-online}"
else
    echo >&2 "WARN: ${WANDB_SECRETS} not readable; WANDB_MODE=offline"
    export WANDB_MODE="offline"
    export WANDB_DIR="${SCRATCH_ROOT}/wandb"
fi
mkdir -p "${WANDB_DIR}"
