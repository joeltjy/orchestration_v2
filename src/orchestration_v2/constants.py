"""Shared constants for orchestration_v2 training pipelines.

Dense-AMT token layout is copied from ~/orchestration's ``constants.py`` (same ids, so checkpoints trained
there load here): one note = 4 tokens (onset, duration, note, velocity), 10 ms time steps.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

SCRATCH_ROOT = Path("/data/scratch-oc40/joel")
CHECKPOINT_ROOT = SCRATCH_ROOT / "checkpoints"
LABEL_IGNORE_INDEX = -100

# ---------------------------------------------------------------------------
# wandb (same projects as ~/orchestration)
# ---------------------------------------------------------------------------
WANDB_ENTITY = "joeltjy17-massachusetts-institute-of-technology"
WANDB_PROJECT_DRYRUN = "orchestration_dryruns"


def wandb_project_full_train(*, today: date | None = None) -> str:
    del today
    return "orchestration_nodouble"


# ---------------------------------------------------------------------------
# Dense AMT (copied from ~/orchestration constants.py)
# ---------------------------------------------------------------------------
AMT_DENSE_ONSET_OFFSET = 0
AMT_DENSE_DURATION_OFFSET = 10_000
AMT_DENSE_NOTE_OFFSET = 11_000
AMT_DENSE_NUM_INSTRUMENTS = 5
AMT_DENSE_VELOCITY_OFFSET = AMT_DENSE_NOTE_OFFSET + AMT_DENSE_NUM_INSTRUMENTS * 128  # 11640
AMT_DENSE_VOCAB_SIZE = AMT_DENSE_VELOCITY_OFFSET + 128  # 11768
AMT_DENSE_MAX_ONSET = 9_999
AMT_DENSE_MAX_DURATION = 999
assert AMT_DENSE_VELOCITY_OFFSET == 11_640 and AMT_DENSE_VOCAB_SIZE == 11_768

# Dense-AMT medium (24 x 1024, 16 heads) checkpoints
AMT_DENSE_BIDIR_CHECKPOINT = CHECKPOINT_ROOT / "amt_dense" / "hf_trainer_pianocore_bidir"  # PianoCoRe MLM
AMT_DENSE_CAUSAL_CHECKPOINT = CHECKPOINT_ROOT / "amt_dense" / "hf_trainer_30k" / "best_eval_loss_note"

# ---------------------------------------------------------------------------
# Expressifier: score -> performance encoder-decoder
#
# Encoder = dense-AMT medium (bidirectional) over a score chunk; decoder = causal nn.TransformerDecoder over
# the performance of the same notes. Both sides are dense-AMT quads, 4 tokens per note, piano only
# (instrument 0). Chunks hold <= 128 notes in score-onset order; the output keeps the score's note order, so
# performance onsets need not be monotonic. Onsets are relative to the chunk's earliest onset on each side.
# ---------------------------------------------------------------------------
EXPRESSIFIER_TOKENS_PER_NOTE = 4
EXPRESSIFIER_MAX_NOTES = 128
EXPRESSIFIER_MAX_TOKENS = EXPRESSIFIER_MAX_NOTES * EXPRESSIFIER_TOKENS_PER_NOTE  # 512
EXPRESSIFIER_CHUNK_LENGTHS = (128, 64, 32, 16)
EXPRESSIFIER_CHUNK_PROBS = (0.5, 0.25, 0.125, 0.125)
EXPRESSIFIER_MIN_CHUNK_NOTES = 4  # trailing chunks shorter than this are dropped
EXPRESSIFIER_SCORE_VELOCITY = 80  # score notes carry no dynamics: one fixed velocity token
EXPRESSIFIER_VOCAB_SIZE = AMT_DENSE_VOCAB_SIZE  # shared by encoder input and decoder output
EXPRESSIFIER_BOS = AMT_DENSE_VOCAB_SIZE  # decoder-only start token (11768)
EXPRESSIFIER_DECODER_INPUT_VOCAB = AMT_DENSE_VOCAB_SIZE + 1
EXPRESSIFIER_PAIRS_DIRNAME = "expressifier_pairs_pianocore_astar"
EXPRESSIFIER_ALIGNED_DIRNAME = "aligned_pianocore_maestro"
EXPRESSIFIER_STAGE1_DATA_DIRNAME = "expressifier_stage1_pianocore_astar"
EXPRESSIFIER_STAGE2_DATA_DIRNAME = "expressifier_stage2_aligned_maestro_aug16"
EXPRESSIFIER_STAGE1_CHECKPOINT_DIRNAME = "expressifier_stage1"
EXPRESSIFIER_STAGE2_CHECKPOINT_DIRNAME = "expressifier_stage2"
EXPRESSIFIER_ENCODER_CHECKPOINT = AMT_DENSE_BIDIR_CHECKPOINT / "best_eval_loss"
