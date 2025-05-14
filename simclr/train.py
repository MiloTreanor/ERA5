from datasets import ContrastiveDataset
from transform import TemporalAugmentation
from simclr_model.simclr import SimCLR
from simclr_model.simclr_onesided import SimCLROneSide
import os
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping


def train_simclr():
    # Load full dataset
    full_dataset = ContrastiveDataset(
        '/scratch-shared/tmp.Udl4HYbZtd/dataset_normalized_2018-2021.h5',
        transform=TemporalAugmentation()
    )

    # Split into train/val (80/20 split)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    num_workers = 4  # Increased from 4
    persistent_workers = True
    prefetch_factor = 2  # Prefetch 2 batches per worker
    pin_memory = True
    batch_size = 16
    epochs = 100

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        drop_last=False  # Keep all samples for validation
    )

    # Model setup remains identical
    model = SimCLR(
        hidden_dim=128,
        lr=5e-4 * 16,
        temperature=0.07,
        weight_decay=1e-4,
        max_epochs=epochs
    )

    # Trainer setup with validation
    trainer = pl.Trainer(
        precision='16-mixed',
        accelerator="auto",
        devices=1,
        max_epochs=epochs,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True,
                mode="max",
                monitor="val_acc_top1"
            ),
            LearningRateMonitor("epoch"),
            EarlyStopping(
                monitor="val_loss",  # Monitor validation loss
                patience=10,  # Wait 10 epochs without improvement
                mode="max"  # Stop when metric stops increasing
            )
        ],
        default_root_dir='/scratch-shared/tmp.Udl4HYbZtd/models/simclr'

    )
    #os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    #os.environ["NCCL_ALGO"] = "Tree"
    # Train with both loaders
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    train_simclr()