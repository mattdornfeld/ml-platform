"""A minimal Lightning module and data module for MNIST classification."""

from pathlib import Path

import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTDataModule(pl.LightningDataModule):
    """Download MNIST once and provide normalized training and test loaders."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> None:
        """Store loader configuration and define the shared image transform."""
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.train_dataset: datasets.MNIST | None = None
        self.test_dataset: datasets.MNIST | None = None

    def prepare_data(self) -> None:
        """Download the training and test MNIST files when they are absent."""
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        """Create datasets required by Lightning's requested training stage."""
        if stage in ("fit", None):
            self.train_dataset = datasets.MNIST(
                self.data_dir, train=True, transform=self.transform
            )
        if stage in ("fit", "test", None):
            self.test_dataset = datasets.MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self) -> DataLoader:
        """Return shuffled batches from the MNIST training dataset."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting the training loader.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Return ordered batches from the MNIST test dataset for validation."""
        if self.test_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting the validation loader.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Return ordered batches from the MNIST test dataset for final testing."""
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') before requesting the test loader.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class MNISTClassifier(pl.LightningModule):
    """A two-layer MLP classifier for 28 by 28 grayscale digit images."""

    def __init__(self, learning_rate: float = 1e-3) -> None:
        """Initialize the MLP layers and persist the optimizer learning rate."""
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Convert a batch of images into unnormalized class scores."""
        return self.model(images)

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        """Calculate loss and accuracy for one batch and log stage-specific metrics."""
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run one optimization batch and return its cross-entropy loss."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Evaluate and log metrics for one validation batch."""
        self._shared_step(batch, "val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Evaluate and log metrics for one final test batch."""
        self._shared_step(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Create the Adam optimizer configured with the saved learning rate."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
