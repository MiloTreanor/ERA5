from models.unet_precip_regression_lightning import UNetDSAttention
from simclr.datasets import ContrastiveDataset
from simclr.transform import TemporalAugmentation
import torch
import matplotlib.pyplot as plt

ORIGINAL_MEAN = 6.971789036885477e-05
ORIGINAL_STD = 0.0002507412914788022

def load_model():
    """Explicit model loading without fixtures"""
    checkpoint_path = "smat_multiv2/UNetDSAttention_rain_threshold_50_epoch=25-val_loss=13727.683594.ckpt"
    hparams = {
        "n_channels": 12,
        "n_classes": 1,
        "bilinear": True,
        "reduction_ratio": 16,
        "kernels_per_layer": 2
    }

    model = UNetDSAttention(hparams=hparams)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def test_samples():
    """Test with explicit data loading"""
    # 1. Load model
    model = load_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # 2. Load dataset
    full_dataset = ContrastiveDataset(
        "C:/Users/milot/PycharmProjects/ERA5/obtain_data/test_dataset_normalized.h5",
        transform=TemporalAugmentation()
    )

    # 3. Get sample and target
    input_idx = 60
    input_tensor, _ = full_dataset[input_idx]
    target_tensor, _ = full_dataset[61]  # Assuming second return value is target
    target_tensor = target_tensor[0]
    # Add batch dimension and send to device
    input_batch = input_tensor.unsqueeze(0).to(device)

    # 4. Run inference
    with torch.no_grad():
        prediction = model(input_batch)

        prediction = torch.clamp((prediction * ORIGINAL_STD + ORIGINAL_MEAN), min=0.0)
        target_tensor = torch.clamp((target_tensor * ORIGINAL_STD + ORIGINAL_MEAN), min=0.0)

        # Convert to numpy for visualization
        prediction_np = prediction.squeeze().cpu().numpy()  # [288, 288]
        target_np = target_tensor.squeeze().cpu().numpy()  # [288, 288]

        # Print value statistics
        print("\nValue Comparison:")
        print(
            f"Prediction - Min: {prediction_np.min():.4f}, Max: {prediction_np.max():.4f}, Mean: {prediction_np.mean():.4f}")
        print(f"Target     - Min: {target_np.min():.4f}, Max: {target_np.max():.4f}, Mean: {target_np.mean():.4f}")

        # Visualization
        plot_comparison(input_tensor, prediction_np, target_np)


def plot_comparison(input_tensor, prediction, target):
    """Enhanced visualization with all three elements"""
    plt.figure(figsize=(18, 6))

    # Input (first channel)
    plt.subplot(1, 3, 1)
    plt.imshow(input_tensor[0].cpu().numpy(), cmap='viridis')
    plt.title('Input Channel 0')
    plt.colorbar()

    # Prediction
    plt.subplot(1, 3, 2)
    plt.imshow(prediction, cmap='viridis')
    plt.title(f'Prediction\nMax: {prediction.max():.4f}')
    plt.colorbar()

    # Target
    plt.subplot(1, 3, 3)
    plt.imshow(target, cmap='viridis')
    plt.title(f'Target\nMax: {target.max():.4f}')
    plt.colorbar()

    plt.tight_layout()
    plt.show()