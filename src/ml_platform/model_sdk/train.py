"""Command-line entry point for training the MNIST Lightning example."""

import argparse
from pathlib import Path

import lightning.pytorch as pl

from ml_platform.model_sdk.mnist import MNISTClassifier, MNISTDataModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Lightning MLP on MNIST.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run one train and validation batch to verify the pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    data = MNISTDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = MNISTClassifier(learning_rate=args.learning_rate)
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=args.max_epochs,
        deterministic=True,
        fast_dev_run=args.fast_dev_run,
    )
    trainer.fit(model, datamodule=data)
    if not args.fast_dev_run:
        trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
