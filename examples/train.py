from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from liquid_audio.data.dataloader import LFM2DataLoader, TaskBalancedDataset
from liquid_audio.trainer import Trainer
from liquid_audio.training import TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune LFM2-Audio from a reproducible JSON recipe.")
    parser.add_argument("--config", default="configs/omni_lora.json", help="TrainerConfig JSON recipe.")
    parser.add_argument(
        "--data",
        action="append",
        required=True,
        help="Preprocessed training dataset; repeat for a task-balanced omni mixture.",
    )
    parser.add_argument("--data-weight", action="append", type=float, help="One sampling weight per --data.")
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--val-data", help="Optional preprocessed validation dataset.")
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--model-id", help="Override model_id from the recipe.")
    parser.add_argument("--revision", help='Override the model revision; use "none" for an unpinned/local model.')
    parser.add_argument("--output-dir", help="Override output_dir from the recipe.")
    parser.add_argument("--resume", nargs="?", const="latest", help="Resume a checkpoint path, or latest when omitted.")
    parser.add_argument(
        "--strategy",
        choices=("full", "asr_adapter", "asr_lora", "omni_lora"),
        help="Override the fine-tuning strategy.",
    )
    return parser.parse_args()


def require_dataset(path: str) -> Path:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Preprocessed dataset not found at {dataset_path}. Run preprocessing first.")
    return dataset_path


if __name__ == "__main__":
    args = parse_args()
    config = TrainerConfig.from_json(args.config)
    if args.model_id:
        config.model_id = args.model_id
        config.revision = None
    if args.revision:
        config.revision = None if args.revision.casefold() in {"none", "null"} else args.revision
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.resume:
        config.resume_from_checkpoint = args.resume
    if args.strategy:
        config.fine_tune = replace(config.fine_tune, strategy=args.strategy)
    config.validate()

    task_datasets = [
        LFM2DataLoader(
            dataset_path=str(require_dataset(path)),
            context_length=args.context_length,
            dynamic_padding=True,
            augmentation_seed=config.seed,
        )
        for path in args.data
    ]
    if args.data_weight is not None and len(args.data_weight) != len(task_datasets):
        raise ValueError("provide exactly one --data-weight for each --data")
    train_data = (
        task_datasets[0]
        if len(task_datasets) == 1 and args.data_weight is None and args.samples_per_epoch is None
        else TaskBalancedDataset(
            task_datasets,
            weights=args.data_weight,
            samples_per_epoch=args.samples_per_epoch,
            seed=config.seed,
        )
    )
    val_data = (
        LFM2DataLoader(
            dataset_path=str(require_dataset(args.val_data)),
            context_length=args.context_length,
            dynamic_padding=True,
            augmentation_seed=config.seed,
        )
        if args.val_data
        else None
    )
    Trainer(config=config, train_data=train_data, val_data=val_data).train()
