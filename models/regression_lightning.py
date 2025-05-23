import lightning.pytorch as pl
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from create_dataset.dataset_multivariate import SingleStepPrecipDataset
import argparse
import numpy as np


class UNetBase(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--model",
            type=str,
            default="UNet",
            choices=["UNet", "UNetDS", "UNetAttention", "UNetDSAttention", "PersistenceModel"],
        )
        parser.add_argument("--n_channels", type=int, default=12)
        parser.add_argument("--n_classes", type=int, default=1)
        parser.add_argument("--kernels_per_layer", type=int, default=1)
        parser.add_argument("--bilinear", type=bool, default=True)
        parser.add_argument("--reduction_ratio", type=int, default=16)
        parser.add_argument("--lr_patience", type=int, default=5)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

    def forward(self, x):
        pass

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.1, patience=self.hparams.lr_patience
            ),
            "monitor": "val_loss",  # Default: val_loss
        }
        return [opt], [scheduler]

    def loss_func(self, y_pred, y_true):
        # Ensure consistent shapes before computing loss
        if y_pred.dim() > y_true.dim():
            y_pred = y_pred.squeeze(1)
        elif y_true.dim() > y_pred.dim():
            y_pred = y_pred.unsqueeze(1)

        # reduction="mean" is average of every pixel, but I want average of image
        return nn.functional.mse_loss(y_pred, y_true, reduction="sum") / y_true.size(0)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred, y)
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred, y)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Calculate the loss (MSE per default) on the test set normalized and denormalized."""
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred, y)
        factor = 47.83
        loss_denorm = self.loss_func(y_pred * factor, y * factor)
        self.log("MSE", loss)
        self.log("MSE_denormalized", loss_denorm)


class PrecipRegressionBase(UNetBase):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = UNetBase.add_model_specific_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_input_images", type=int, default=12)
        parser.add_argument("--num_output_images", type=int, default=6)
        parser.add_argument("--valid_size", type=float, default=0.1)
        parser.add_argument("--use_oversampled_dataset", type=bool, default=True)
        parser.n_channels = parser.parse_args().num_input_images
        parser.n_classes = 1
        return parser

    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        self.train_dataset = None
        self.valid_dataset = None
        self.train_sampler = None
        self.valid_sampler = None

    def prepare_data(self):
        # train_transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip()]
        # )
        train_transform = None
        valid_transform = None
        # Instantiate datasets directly from the HDF5 splits
        self.train_dataset = SingleStepPrecipDataset(
            h5_file=self.hparams.train_dataset,
            transform=train_transform,
        )
        self.valid_dataset = SingleStepPrecipDataset(
            h5_file=self.hparams.val_dataset,
            transform=valid_transform,
        )
    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            sampler=self.train_sampler,
            pin_memory=True,
            # The following can/should be tweaked depending on the number of CPU cores
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
            prefetch_factor=self.hparams.prefetch_factor
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.batch_size,
            sampler=self.valid_sampler,
            pin_memory=True,
            # The following can/should be tweaked depending on the number of CPU cores
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
            prefetch_factor=self.hparams.prefetch_factor
        )
        return valid_loader


class PersistenceModel(UNetBase):
    def forward(self, x):
        return x[:, -1:, :, :]
