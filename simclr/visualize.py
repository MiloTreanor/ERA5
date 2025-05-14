# visualize_transforms.py
import matplotlib.pyplot as plt
from datasets import ContrastiveDataset
from transform import ApplyMeanFilter
import torch
import torchvision.transforms as transforms
from transform import TemporalAugmentation
import numpy as np
import random

def calculate_spatial_correlation(view1: torch.Tensor, view2: torch.Tensor, eps: float = 1e-8) -> list:
    """
    Calculates spatial correlation for each layer between two views

    Args:
    view1: Tensor of shape (12, H, W)
    view2: Tensor of shape (12, H, W)
    eps: Small epsilon to avoid division by zero

    Returns:
    List of 12 correlation coefficients (float) for corresponding layers
    """
    # Input validation
    assert view1.shape == view2.shape, "Views must have same shape"
    assert view1.dim() == 3 and view1.size(0) == 12, "Input must be (12, H, W)"

    correlations = []

    for layer in range(12):
        # Extract layer from both views
        v1 = view1[layer].flatten()
        v2 = view2[layer].flatten()

        # Calculate means
        mean1 = v1.mean()
        mean2 = v2.mean()

        # Center the data
        v1_centered = v1 - mean1
        v2_centered = v2 - mean2

        # Calculate covariance
        covariance = (v1_centered * v2_centered).sum()

        # Calculate standard deviations
        std1 = torch.sqrt((v1_centered ** 2).sum() + eps)
        std2 = torch.sqrt((v2_centered ** 2).sum() + eps)

        # Pearson correlation coefficient
        corr = covariance / (std1 * std2)
        correlations.append(corr.item())

    return correlations

def visualize_transformations():
    # Load dataset without transformations
    dataset = ContrastiveDataset("../dataset_normalized_2018-2021.h5", transform=TemporalAugmentation())
    averages = np.zeros(12)
    averages_negative = np.zeros(12)
    neg_count = 0
    pos_count = 0

    for _ in range(1000):
        random_integer = random.sample(range(1, 35000), 3)
        original1, cropped1 = dataset[random_integer[0]]
        original2, cropped2 = dataset[random_integer[1]]
        original3, cropped3 = dataset[random_integer[2]]

        averages += np.array(calculate_spatial_correlation(original1, cropped1))
        averages += np.array(calculate_spatial_correlation(original2, cropped2))
        averages += np.array(calculate_spatial_correlation(original3, cropped3))

        averages_negative += np.array(calculate_spatial_correlation(original1, cropped2))
        averages_negative += np.array(calculate_spatial_correlation(original1, cropped3))
        averages_negative += np.array(calculate_spatial_correlation(original2, cropped1))
        averages_negative += np.array(calculate_spatial_correlation(original2, cropped3))
        averages_negative += np.array(calculate_spatial_correlation(original3, cropped1))
        averages_negative += np.array(calculate_spatial_correlation(original3, cropped2))


        averages_negative += np.array(calculate_spatial_correlation(original1, original2)) *2
        averages_negative += np.array(calculate_spatial_correlation(original1, original3)) *2
        #averages_negative += np.array(calculate_spatial_correlation(original2, original1))
        averages_negative += np.array(calculate_spatial_correlation(original2, original3)) *2
        #averages_negative += np.array(calculate_spatial_correlation(original3, original1))
        #averages_negative += np.array(calculate_spatial_correlation(original3, original2))
        neg_count += 12
        pos_count +=3

    print(averages_negative/neg_count)
    print(averages/pos_count)

def plot_transformations():
    dataset = ContrastiveDataset("../dataset_normalized_2018-2021.h5", transform=TemporalAugmentation())
    original, cropped = dataset[5]
    variables = 2
    # Plotting
    fig, axes = plt.subplots(2, variables, figsize=(15, 5))
    # 5th channel (0-indexed)

    for channel_idx in range(variables):
        # Original image
        axes[1, channel_idx].imshow(original[channel_idx], cmap='viridis')

        axes[1, channel_idx].axis('off')

        # Cropped image

        axes[0, channel_idx].imshow(cropped[channel_idx], cmap='viridis')
        axes[0, channel_idx].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    #visualize_transformations()
    plot_transformations()