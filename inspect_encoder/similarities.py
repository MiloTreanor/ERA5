import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


def analyze_similarities(model, dataloader, device='cuda'):
    model.eval()
    pos_similarities = []
    neg_similarities = []

    with torch.inference_mode():
        for batch in dataloader:
            x1, x2 = batch[0], batch[1]

            # Get embeddings
            z1 = model.projection(model.encoder(x1.to(device)).mean((2, 3)))
            z2 = model.projection(model.encoder(x2.to(device)).mean((2, 3)))
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)

            # Positive pairs (diagonal)
            pos_sim = (z1 * z2).sum(dim=1)
            pos_similarities.extend(pos_sim.cpu().numpy())

            # Negative pairs (off-diagonal)
            sim_matrix = torch.mm(z1, z2.T)  # [batch_size x batch_size]
            mask = ~torch.eye(x1.size(0), dtype=torch.bool)  # Remove diagonals
            neg_sim = sim_matrix[mask]
            neg_similarities.extend(neg_sim.cpu().numpy())

    # Plotting
    plt.figure(figsize=(10, 4))

    # Positive distribution
    plt.subplot(121)
    plt.hist(pos_similarities, bins=50)
    plt.title(f"Positive Similarities\n(μ={np.mean(pos_similarities):.2f})")

    # Comparison plot
    plt.subplot(122)
    plt.hist(neg_similarities, bins=50, alpha=0.5, label='Negatives')
    plt.hist(pos_similarities, bins=50, alpha=0.5, label='Positives')
    plt.legend()
    plt.title("Similarity Comparison")
    plt.show()


def check_embedding_variance(model, dataloader, device='cuda'):
    # Expected Results:
    # - Good: Slow variance decay (high-dimensional info)
    # - Bad: First few components explain >90% variance

    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            z = model.projection(model.encoder(x).mean((2, 3)))
            embeddings.append(z.cpu())

    embeddings = torch.cat(embeddings)
    cov_matrix = torch.cov(embeddings.T)
    explained_var = torch.linalg.eigvalsh(cov_matrix).flip(0)

    plt.plot(explained_var.cumsum(0) / explained_var.sum())
    plt.title("Cumulative Explained Variance")
    plt.xlabel("Principal Components")
    plt.show()


def temporal_consistency(model, dataset, num_samples=1000, device='cuda'):
    idxs = np.random.choice(len(dataset) - 10, num_samples)
    similarities = []

    # Process temporal pairs
    with torch.inference_mode():
        for i in idxs:
            x1, _ = dataset[i]
            x2, _ = dataset[i + 1]

            # Add batch dimension and move to device
            x1 = x1.unsqueeze(0).to(device)
            x2 = x2.unsqueeze(0).to(device)

            z1 = model.projection(model.encoder(x1).mean((2, 3)))
            z2 = model.projection(model.encoder(x2).mean((2, 3)))

            sim = F.cosine_similarity(z1, z2).item()
            similarities.append(sim)


    plt.hist(similarities, bins=50, alpha=0.5, label='Temporal Neighbors')
    plt.legend()
    plt.show()

def augmentation_sensitivity(model, dataset, num_samples=100, device='cuda'):
    # Expected Results:
    # - Good: Augmented >> Random
    # - Bad: Augmented ≈ Random
    orig_sims, aug_sims = [], []
    for _ in range(num_samples):
        x, x_aug = dataset[np.random.randint(len(dataset))]
        with torch.no_grad():
            z = model.projection(model.encoder(x.unsqueeze(0).to(device)).mean((2, 3)))
            z_aug = model.projection(model.encoder(x_aug.unsqueeze(0).to(device)).mean((2, 3)))
            sim = F.cosine_similarity(z, z_aug).item()

        # Compare with different sample
        x_other = dataset[np.random.randint(len(dataset))][0]
        z_other = model.projection(model.encoder(x_other.unsqueeze(0).to(device)).mean((2, 3)))
        sim_other = F.cosine_similarity(z, z_other).item()

        orig_sims.append(sim)
        aug_sims.append(sim_other)

    print(f"Augmented Similarity: {np.mean(orig_sims):.3f} ± {np.std(orig_sims):.3f}")
    print(f"Random Similarity: {np.mean(aug_sims):.3f} ± {np.std(aug_sims):.3f}")




