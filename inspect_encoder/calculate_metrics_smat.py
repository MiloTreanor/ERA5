import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Metric
import numpy as np
import matplotlib.pyplot as plt
from models.unet_precip_regression_lightning import UNetDSAttention
from simclr.datasets import ContrastiveDataset
from simclr.transform import TemporalAugmentation

# Original normalization parameters
ORIGINAL_MEAN = 6.971789036885477e-05 * 1000
ORIGINAL_STD = 0.0002507412914788022 * 1000


class TimeSeriesDataset(Dataset):
    """Custom dataset to handle temporal sequencing"""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset) - 1  # Ensure we have targets for all samples

    def __getitem__(self, idx):
        # Get current input and next target
        input, _ = self.base_dataset[idx]
        target, _ = self.base_dataset[idx + 1]
        return input, target[0]  # Return only 0th channel of target


class PrecipitationMetrics(Metric):
    """Adapted metrics for our normalization scheme"""

    def __init__(self, threshold=0.5, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold = threshold

        # Register states
        self.add_state("sum_mse", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_mae", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

        # Classification metrics
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    def _denormalize(self, tensor):
        """Apply our specific denormalization and clamp negatives"""
        denorm = tensor * ORIGINAL_STD + ORIGINAL_MEAN
        return torch.clamp(denorm, min=0.0)

    def update(self, preds, targets):
        # Denormalize and convert to physical units
        preds_denorm = self._denormalize(preds)
        targets_denorm = self._denormalize(targets)

        # Regression metrics
        self.sum_mse += torch.nn.functional.mse_loss(preds_denorm, targets_denorm, reduction='sum')
        self.sum_mae += torch.nn.functional.l1_loss(preds_denorm, targets_denorm, reduction='sum')
        self.total_samples += targets.numel()

        # Classification metrics
        pred_mask = preds_denorm > self.threshold
        target_mask = targets_denorm > self.threshold

        self.tp += torch.sum(pred_mask & target_mask)
        self.fp += torch.sum(pred_mask & ~target_mask)
        self.fn += torch.sum(~pred_mask & target_mask)

    def compute(self):
        # Regression results
        mse = self.sum_mse / self.total_samples
        mae = self.sum_mae / self.total_samples

        # Classification results
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        csi = self.tp / (self.tp + self.fn + self.fp + 1e-8)
        far = self.fp / (self.tp + self.fp + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        return {
            'MSE': mse,
            'MAE': mae,
            'Precision': precision,
            'Recall': recall,
            'CSI': csi,
            'FAR': far,
            'F1': f1
        }


def load_model():
    """Load trained model with proper device handling"""
    checkpoint_path = "smat_multiv2/UNetDSAttention_rain_threshold_50_epoch=25-val_loss=13727.683594.ckpt"
    hparams = {
        "n_channels": 12,
        "n_classes": 1,
        "bilinear": True,
        "reduction_ratio": 16,
        "kernels_per_layer": 2
    }

    model = UNetDSAttention(hparams=hparams)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def evaluate_model():
    """Main evaluation function with batch processing"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model().to(device)

    # Create dataset and loader
    base_dataset = ContrastiveDataset(
        "C:/Users/milot/PycharmProjects/ERA5/obtain_data/test_dataset_normalized.h5",
        transform=TemporalAugmentation()
    )
    dataset = TimeSeriesDataset(base_dataset)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=1)

    metrics = PrecipitationMetrics(threshold=0.5).to(device)

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Get predictions and update metrics
            preds = model(inputs)
            metrics.update(preds.squeeze(1), targets)  # Remove channel dim if needed

    results = metrics.compute()

    # Print results with units
    print("\nModel Performance Evaluation:")
    print(f"MSE: {results['MSE'].item():.4f} (mm^2/h^2)")
    print(f"MAE: {results['MAE'].item():.4f} (mm/h)")
    print(f"Precision: {results['Precision'].item():.4f}")
    print(f"Recall: {results['Recall'].item():.4f}")
    print(f"CSI: {results['CSI'].item():.4f}")
    print(f"FAR: {results['FAR'].item():.4f}")
    print(f"F1 Score: {results['F1'].item():.4f}")


if __name__ == "__main__":
    evaluate_model()