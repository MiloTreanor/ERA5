import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from lightning.pytorch import loggers
import argparse
from models import unet_precip_regression_lightning as unet_regr
from lightning.pytorch.tuner import Tuner
import torch

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
def train_regression(hparams, find_batch_size_automatically: bool = False):
    if hparams.model == "UNetDSAttention":
        net = unet_regr.UNetDSAttention(hparams=hparams)
    elif hparams.model == "UNetAttention":
        net = unet_regr.UNetAttention(hparams=hparams)
    elif hparams.model == "UNet":
        net = unet_regr.UNet(hparams=hparams)
    elif hparams.model == "UNetDS":
        net = unet_regr.UNetDS(hparams=hparams)
    else:
        raise NotImplementedError(f"Model '{hparams.model}' not implemented")

    # HPC-specific change: Use HPC paths instead of ROOT_DIR
    default_save_path = "Smat_test"

    checkpoint_callback = ModelCheckpoint(
        dirpath=default_save_path,
        filename=net.__class__.__name__ + "_rain_threshold_50_{epoch}-{val_loss:.6f}",
        save_top_k=-1,
        verbose=False,
        monitor="val_loss",
        mode="min",
    )
    lr_monitor = LearningRateMonitor()
    tb_logger = loggers.TensorBoardLogger(save_dir=default_save_path, name=net.__class__.__name__)
    earlystopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=hparams.es_patience,
    )

    # HPC optimization: Enable mixed precision
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        fast_dev_run=hparams.fast_dev_run,
        max_epochs=hparams.epochs,
        default_root_dir=default_save_path,
        logger=tb_logger,
        callbacks=[checkpoint_callback, earlystopping_callback, lr_monitor],
        val_check_interval=hparams.val_check_interval,
        accumulate_grad_batches=1,
    )

    if find_batch_size_automatically:
        tuner = Tuner(trainer)
        tuner.scale_batch_size(net, mode="binsearch")

    trainer.fit(model=net, ckpt_path=hparams.resume_from_checkpoint)

if __name__ == "__main__":


    # Create the parser and add base arguments
    parser = argparse.ArgumentParser()
    parser = unet_regr.PrecipRegressionBase.add_model_specific_args(parser)

    # Add additional arguments (ensure these are added AFTER the base arguments)

    parser.add_argument(
        "--train_dataset",
        type=str,
        default="C:/Users/milot/PycharmProjects/ERA5/create_dataset/test_dataset_normalized.h5",
        help="Path to the training H5 file"
    )
    parser.add_argument(
        "--val_dataset",
        type=str,
        default="C:/Users/milot/PycharmProjects/ERA5/create_dataset/test_dataset_normalized.h5",
        help="Path to the validation H5 file"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    # Use store_true so that --fast_dev_run sets the flag to True when provided
    parser.add_argument("--fast_dev_run", action="store_true", help="Run a fast development run")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Checkpoint path to resume training")
    parser.add_argument("--val_check_interval", type=float, default=None, help="Validation check interval")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--prefetch_factor", type=int, default=4)

    parser.add_argument(
        "--pretrained_encoder_path",
        type=str,
        default=None,
        help="Path to SimCLR encoder checkpoint"
    )
    #"C:/Users/milot/PycharmProjects/ERA5/inspect_encoder/simclr_logs/version_6/checkpoints/epoch=9-step=550.ckpt"
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze encoder weights during training"
    )

    args = parser.parse_args()

    # Add additional parameters that are not passed from the command line
    args.n_channels = 12
    args.lr_patience = 4       # Required for learning rate scheduler
    args.es_patience = 15      # Required for EarlyStopping callback
    args.kernels_per_layer = 2
    args.use_oversampled_dataset = True
    args.model = "UNetDSAttention"

    print(f"Start training model: {args.model}")
    train_regression(args, find_batch_size_automatically=False)