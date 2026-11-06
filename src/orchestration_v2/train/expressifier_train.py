"""Train the expressifier (score -> performance). HF Trainer, wandb logging, torchrun-ready.

    torchrun --standalone --nproc_per_node=2 -m orchestration_v2.train.expressifier_train --stage 1
    torchrun ... -m orchestration_v2.train.expressifier_train --stage 2   # starts from stage-1 best

Checkpoints: HF ``checkpoint-N`` dirs (resumable) each also hold the expressifier format
(``encoder/`` + ``decoder.pt``); at the end the best model (lowest eval loss) is saved to ``<output>/best``.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from orchestration_v2.configs.expressifier_train import (
    DECODER_DROPOUT,
    DECODER_LAYERS,
    DECODER_LR_SCALE,
    DECODER_PARAM_PREFIXES,
    DRYRUN,
    ENCODER_CHECKPOINT,
    TMPDIR,
    build_training_arguments,
    stage_config,
)


def configure_environment() -> None:
    TMPDIR.mkdir(parents=True, exist_ok=True)
    for var in ("TMPDIR", "TEMP", "TMP"):
        os.environ.setdefault(var, str(TMPDIR))


def make_trainer_class():
    import torch
    from transformers import Trainer

    class ExpressifierTrainer(Trainer):
        """Two AdamW groups: pretrained encoder at lr, new decoder at lr * DECODER_LR_SCALE."""

        decoder_lr_scale: float = DECODER_LR_SCALE

        def create_optimizer(self):
            if self.optimizer is not None:
                return self.optimizer
            lr, wd = float(self.args.learning_rate), float(self.args.weight_decay)
            groups: dict[tuple[bool, bool], list] = {}
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                bare = name.removeprefix("module.")
                is_dec = bare.startswith(DECODER_PARAM_PREFIXES)
                no_decay = p.ndim < 2 or "ln" in bare or "norm" in bare
                groups.setdefault((is_dec, no_decay), []).append(p)
            param_groups = [dict(params=ps, lr=lr * (self.decoder_lr_scale if is_dec else 1.0),
                                 weight_decay=0.0 if nd else wd) for (is_dec, nd), ps in groups.items()]
            opt_cls, opt_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            opt_kwargs.pop("lr", None)
            self.optimizer = opt_cls(param_groups, lr=lr, **opt_kwargs)
            return self.optimizer

    del torch
    return ExpressifierTrainer


def make_save_callback():
    from transformers import TrainerCallback
    from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

    from orchestration_v2.models.expressifier import save_expressifier

    class SaveExpressifierCallback(TrainerCallback):
        def on_save(self, args, state, control, model=None, **kwargs):  # noqa: ANN001
            if model is not None and state.is_world_process_zero:
                dest = Path(args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
                if dest.is_dir():
                    save_expressifier(model, dest)
            return control

        def on_train_end(self, args, state, control, model=None, **kwargs):  # noqa: ANN001
            if model is not None and state.is_world_process_zero:
                save_expressifier(model, Path(args.output_dir) / "best")  # best model is loaded at end
            return control

    return SaveExpressifierCallback()


def build_model(stage: int, *, init_from: Path | None, encoder_checkpoint: Path, decoder_layers: int,
                dropout: float):
    from orchestration_v2.models.expressifier import (
        is_expressifier_checkpoint,
        load_expressifier_checkpoint,
        load_expressifier_model,
    )

    if init_from is not None:
        if not is_expressifier_checkpoint(init_from):
            raise FileNotFoundError(f"{init_from}: not an expressifier checkpoint (encoder/ + decoder.pt)")
        print(f"stage {stage}: initialising from {init_from}", flush=True)
        return load_expressifier_checkpoint(init_from, train=True)
    print(f"stage {stage}: encoder from {encoder_checkpoint}, new {decoder_layers}-layer decoder", flush=True)
    return load_expressifier_model(encoder_checkpoint, decoder_layers=decoder_layers, dropout=dropout)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train the expressifier (score -> performance).")
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--encoder-checkpoint", type=Path, default=ENCODER_CHECKPOINT)
    ap.add_argument("--init-from", type=str, default=None,
                    help="expressifier checkpoint to start from (stage 2 default: stage-1 best); 'none' = fresh")
    ap.add_argument("--decoder-layers", type=int, default=DECODER_LAYERS)
    ap.add_argument("--dropout", type=float, default=DECODER_DROPOUT)
    ap.add_argument("--resume-from-checkpoint", type=Path, default=None)
    ap.add_argument("--num-train-samples", type=int, default=None)
    ap.add_argument("--dryrun", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    configure_environment()
    from transformers import EarlyStoppingCallback

    from orchestration_v2.dataloaders.expressifier_dataloader import ExpressifierDataset, collate_expressifier_batch

    cfg = stage_config(args.stage)
    data_root = args.data_root or cfg["data_root"]
    init = args.init_from if args.init_from is not None else cfg.get("init_from")
    init_from = None if init in (None, "none") else Path(init)
    n_train = DRYRUN["num_train_samples"] if args.dryrun else args.num_train_samples
    n_eval = DRYRUN["num_eval_samples"] if args.dryrun else cfg["num_eval_samples"]
    train_ds = ExpressifierDataset(data_root, "train", max_samples=n_train, seed=cfg["seed"])
    eval_ds = ExpressifierDataset(data_root, "test", max_samples=n_eval, seed=cfg["seed"])
    print(f"expressifier stage {args.stage}: data={data_root} train={len(train_ds)} eval={len(eval_ds)} chunks",
          flush=True)

    model = build_model(args.stage, init_from=init_from, encoder_checkpoint=args.encoder_checkpoint,
                        decoder_layers=args.decoder_layers, dropout=args.dropout)
    training_args = build_training_arguments(args.stage, args.output_dir, with_eval=True, dryrun=args.dryrun)
    callbacks = [make_save_callback()]
    if not args.dryrun:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"]))
    trainer = make_trainer_class()(model=model, args=training_args, train_dataset=train_ds, eval_dataset=eval_ds,
                                   data_collator=collate_expressifier_batch, callbacks=callbacks)
    print(f"run={training_args.run_name} max_steps={training_args.max_steps} lr={training_args.learning_rate} "
          f"(decoder x{DECODER_LR_SCALE}) out={training_args.output_dir}", flush=True)
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)


if __name__ == "__main__":
    main()
