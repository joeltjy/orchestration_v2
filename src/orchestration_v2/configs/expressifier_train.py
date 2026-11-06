"""Configuration for the expressifier (score -> performance), stages 1 and 2.

Stage 1: PianoCoRe A* pairs, no augmentation. Stage 2: aligned_pianocore_maestro (MAESTRO crops), 16x
materialised augmentation, no replay, initialised from the stage-1 checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from orchestration_v2.constants import (
    CHECKPOINT_ROOT,
    EXPRESSIFIER_ENCODER_CHECKPOINT,
    EXPRESSIFIER_STAGE1_CHECKPOINT_DIRNAME,
    EXPRESSIFIER_STAGE1_DATA_DIRNAME,
    EXPRESSIFIER_STAGE2_CHECKPOINT_DIRNAME,
    EXPRESSIFIER_STAGE2_DATA_DIRNAME,
    SCRATCH_ROOT,
    WANDB_PROJECT_DRYRUN,
    wandb_project_full_train,
)

TMPDIR = SCRATCH_ROOT / "tmp"

ENCODER_CHECKPOINT = EXPRESSIFIER_ENCODER_CHECKPOINT
DECODER_LAYERS = 6
DECODER_DROPOUT = 0.1

# Per-parameter-group learning rates: the new decoder learns faster than the pretrained encoder.
DECODER_LR_SCALE = 10.0
DECODER_PARAM_PREFIXES = ("dec_wte", "dec_wpe", "dec_field", "decoder", "dec_ln", "head")

COMMON: dict[str, Any] = dict(
    batch_size=8,
    gradient_accumulation_steps=2,
    weight_decay=0.1,
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    seed=42,
    logging_steps=10,
    save_total_limit=5,
    dataloader_num_workers=4,
    bf16=False,  # repo convention: fp32
    early_stopping_patience=20,
)

STAGES: dict[int, dict[str, Any]] = {
    1: dict(
        data_root=SCRATCH_ROOT / EXPRESSIFIER_STAGE1_DATA_DIRNAME,
        output_dir=CHECKPOINT_ROOT / EXPRESSIFIER_STAGE1_CHECKPOINT_DIRNAME,
        run_name="expressifier_stage1_pianocore",
        learning_rate=3e-5,
        max_steps=200_000,
        save_steps=2_000,
        eval_steps=2_000,
        num_eval_samples=2_000,
    ),
    2: dict(
        data_root=SCRATCH_ROOT / EXPRESSIFIER_STAGE2_DATA_DIRNAME,
        output_dir=CHECKPOINT_ROOT / EXPRESSIFIER_STAGE2_CHECKPOINT_DIRNAME,
        run_name="expressifier_stage2_maestro_aug16",
        learning_rate=1e-5,
        max_steps=30_000,
        save_steps=1_000,
        eval_steps=1_000,
        num_eval_samples=2_000,
        init_from=CHECKPOINT_ROOT / EXPRESSIFIER_STAGE1_CHECKPOINT_DIRNAME / "best",
    ),
}

DRYRUN = dict(max_steps=20, eval_steps=10, num_train_samples=64, num_eval_samples=16)


def stage_config(stage: int) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"stage must be 1 or 2, got {stage}")
    return {**COMMON, **STAGES[stage]}


def build_training_arguments(stage: int, output_dir: str | Path | None = None, *, with_eval: bool = True,
                             dryrun: bool = False, **overrides: Any):
    """HF TrainingArguments for a stage; wandb on (dry runs go to the dry-run project)."""
    from transformers import TrainingArguments

    cfg = stage_config(stage)
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT_DRYRUN if dryrun else wandb_project_full_train()
    eval_steps = DRYRUN["eval_steps"] if dryrun else cfg["eval_steps"]
    kwargs: dict[str, Any] = dict(
        output_dir=str(output_dir or (cfg["output_dir"] / "dryrun" if dryrun else cfg["output_dir"])),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        max_grad_norm=cfg["max_grad_norm"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        max_steps=DRYRUN["max_steps"] if dryrun else cfg["max_steps"],
        seed=cfg["seed"],
        logging_steps=cfg["logging_steps"],
        save_strategy="no" if dryrun else "steps",
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        eval_strategy="steps" if with_eval else "no",
        eval_steps=eval_steps if with_eval else None,
        load_best_model_at_end=bool(with_eval and not dryrun),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_num_workers=cfg["dataloader_num_workers"],
        dataloader_drop_last=True,
        bf16=cfg["bf16"],
        ddp_find_unused_parameters=False,
        report_to="wandb",
        run_name=f"{cfg['run_name']}_dryrun" if dryrun else cfg["run_name"],
    )
    kwargs.update(overrides)
    # transformers < 4.41 calls it evaluation_strategy (the cluster env has 4.40)
    import inspect

    if "eval_strategy" not in inspect.signature(TrainingArguments.__init__).parameters:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return TrainingArguments(**kwargs)


__all__ = ["COMMON", "DECODER_DROPOUT", "DECODER_LAYERS", "DECODER_LR_SCALE", "DECODER_PARAM_PREFIXES",
           "DRYRUN", "ENCODER_CHECKPOINT", "STAGES", "TMPDIR", "build_training_arguments", "stage_config"]
