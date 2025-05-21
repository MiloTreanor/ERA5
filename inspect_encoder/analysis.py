import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split


from simclr.datasets import ContrastiveDataset
from simclr.transform import TemporalAugmentation
#from simclr.train import train_simclr  # Import just for dataset/loader setup
from simclr.simclr_model.simclr import SimCLR  # Your model class
from similarities import (analyze_similarities, check_embedding_variance,
                         temporal_consistency, augmentation_sensitivity)
from activation_maps import visualize_all_layers
from models.unet_precip_regression_lightning import UNetDSAttention

def main():
    # 1. Load trained model
    checkpoint_path = "smat_multiv2/UNetDSAttention_rain_threshold_50_epoch=32-val_loss=13749.474609.ckpt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimCLR.load_from_checkpoint(checkpoint_path, map_location='cpu', weights_only=True).to(device)

    model = model.to(device)
    model.eval()
    #print(model.encoder)
    # 2. Get dataloaders (reuse your training setup)
    full_dataset = ContrastiveDataset(
        "C:/Users/milot/PycharmProjects/ERA5/obtain_data/test_dataset_normalized.h5",
        transform=TemporalAugmentation()
    )
    _, val_dataset = random_split(full_dataset, [0.8, 0.2])

    # 3. Use smaller batch size for stability
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=0)  # num_workers=0 for Windows

    # 4. Run analysis with memory limits
    with torch.inference_mode():
        #analyze_similarities(model, val_loader, device=device)
        #check_embedding_variance(model, val_loader, device=device)
        #temporal_consistency(model, full_dataset, device=device)
        #augmentation_sensitivity(model, full_dataset, device=device)
        visualize_all_layers(model, full_dataset, device=device)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()  # Windows-specific fix
    main()
