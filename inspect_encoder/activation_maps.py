import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import interpolate


def visualize_all_layers(model, dataset, device='cuda', max_layers=10):
    sample, _ = dataset[np.random.randint(len(dataset))]

    sample = sample.unsqueeze(0).to(device)

    activations = []
    layer_info = []

    # Hook to capture activations
    def hook_fn(module, input, output):
        activations.append(output.detach())

    # Register hooks for all layers
    handles = []
    for idx, layer in enumerate(model.encoder[:max_layers]):
        handles.append(layer.register_forward_hook(hook_fn))
        layer_info.append(f"Layer {idx}: {type(layer).__name__}")

    # Forward pass
    with torch.inference_mode():
        _ = model.encoder(sample)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Visualize each layer's activations
    for idx, (act, info) in enumerate(zip(activations, layer_info)):
        try:
            # Handle different activation dimensions
            if act.ndim == 4:
                # Channel-first format [batch, channels, height, width]
                act_map = act.mean(1).squeeze().cpu().numpy()

                # Upsample small activations for visibility
                if act_map.shape[0] < 32:
                    act_map = interpolate(act[None, None], scale_factor=32 / act_map.shape[0], mode='nearest')[
                        0, 0].cpu().numpy()

                plt.figure(figsize=(12, 4))
                plt.suptitle(f"{info}\nShape: {act.shape}")

                # Input channel (first in sample)
                plt.subplot(131)
                plt.imshow(sample[0, 0].cpu().numpy())
                plt.title("Input Channel 0")

                # Activation map
                plt.subplot(132)
                plt.imshow(act_map, cmap='viridis')
                plt.colorbar()
                plt.title("Channel-Averaged Activation")

                # Max activation channel
                plt.subplot(133)
                max_channel = act.mean((2, 3)).argmax().item()
                plt.imshow(act[0, max_channel].cpu().numpy(), cmap='viridis')
                plt.title(f"Max Activation Channel {max_channel}")

                plt.tight_layout()
                plt.show()

            elif act.ndim == 2:
                print(f"Skipping {info} - 2D output (likely pooled)")
            else:
                print(f"Skipping {info} - Unhandled dimensions {act.shape}")

        except Exception as e:
            print(f"Failed to visualize {info}: {str(e)}")
            continue