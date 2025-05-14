import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from .encoder import load_encoder

class SimCLR(pl.LightningModule):
    def __init__(self, hidden_dim=128, lr=5e-4, temperature=0.07,
                 weight_decay=1e-4, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()

        # Load encoder and determine feature dimension via pooling
        self.encoder = load_encoder()
        with torch.no_grad():
            dummy = torch.randn(2, 12, 288, 288)
            out = self.encoder(dummy)           # [B, C, H, W]
            pooled = out.mean(dim=(2, 3))      # [B, C]
            encoder_dim = pooled.shape[1]

        # Projection head: Linear -> ReLU -> Linear -> LayerNorm
        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, 4 * hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def info_nce_loss(self, batch, mode='train'):
        x1, x2 = batch
        imgs = torch.cat([x1, x2], dim=0)      # [2N, C, H, W]
        N = x1.size(0)

        # Encode + pool + project + normalize
        feats = self.encoder(imgs)             # [2N, C, H, W]
        feats = feats.mean(dim=(2, 3))         # [2N, C]
        z = self.projection(feats)             # [2N, D]
        z = F.normalize(z, dim=1)

        # Similarity matrix
        sim = torch.mm(z, z.T) / self.hparams.temperature
        mask = torch.eye(2 * N, device=sim.device).bool()
        # Use a half-safe negative fill value
        neg_val = torch.tensor(-1e4, dtype=sim.dtype, device=sim.device)
        sim.masked_fill_(mask, neg_val)

        # Construct labels: positive of i is i+N mod 2N
        labels = torch.arange(2 * N, device=sim.device)
        labels = (labels + N) % (2 * N)

        # Use cross-entropy as NT-Xent loss
        loss = F.cross_entropy(sim, labels)


        # Retrieval metrics: rank of positive in logits
        # sim already scaled and masked


        if mode == 'val':
            pos_sim = sim[torch.arange(2 * N), labels].unsqueeze(1)  # [2N, 1]
            neg_sim = sim.clone()
            neg_sim[torch.arange(2 * N), labels] = neg_val  # mask out positives
            all_sim = torch.cat([pos_sim, neg_sim], dim=1)  # [2N, 2N]
            ranks = all_sim.argsort(dim=1, descending=True).argmin(dim=1)
            sorted_indices = sim.argsort(dim=1, descending=True)
            ranks = (sorted_indices == labels.unsqueeze(1)).nonzero()[:, 1]
            top1_acc = (ranks == 0).float().mean()
            top5_acc = (ranks < 5).float().mean()
            self.log(f'{mode}_acc_top1', top1_acc, prog_bar=True)
            self.log(f'{mode}_acc_top5', top5_acc, prog_bar=True)
            self.log(f'{mode}_loss', loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, 'val')

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay
        )
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=10
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=self.hparams.lr / 50
        )
        return [optimizer], [warmup, cosine]
