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
        )

    def info_nce_loss(self, batch, mode='train'):
        x1, x2 = batch
        imgs = torch.cat([x1, x2], dim=0)  # [2N, C, H, W]
        N = x1.size(0)

        # Encode + project + normalize
        feats = self.encoder(imgs)
        feats = feats.mean(dim=(2, 3))  # [2N, C]
        z = self.projection(feats)
        z = F.normalize(z, dim=1)  # [2N, D]

        # Similarity matrix
        sim = torch.mm(z, z.T) / self.hparams.temperature
        mask_self = torch.eye(2 * N, dtype=torch.bool, device=sim.device)
        neg_value = torch.finfo(sim.dtype).min
        sim.masked_fill_(mask_self, neg_value)  # More stable than -1e4

        # Positive pairs: (x1_i, x2_i) and (x2_i, x1_i)
        labels = torch.arange(2 * N, device=sim.device)
        labels = (labels + N) % (2 * N)
        mask_pos = torch.zeros_like(sim, dtype=torch.bool)
        mask_pos[torch.arange(2 * N), labels] = True

        # NT-Xent loss calculation
        pos_sim = sim[mask_pos]
        neg_sim = sim.masked_fill(mask_pos | mask_self, neg_value)
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        loss = -logits[:, 0] + torch.logsumexp(logits, dim=1)
        loss = loss.mean()

        # Validation metrics only when needed
        #if mode == 'val':
        comb_sim = logits  # [2N, 2N-1]
        sorted_indices = comb_sim.argsort(dim=1, descending=True)
        pos_rank = torch.nonzero(sorted_indices == 0, as_tuple=True)[1]
        top1_acc = (pos_rank == 0).float().mean()
        top5_acc = (pos_rank < 5).float().mean()

        self.log(f'{mode}_acc_top1', top1_acc, prog_bar=True)
        self.log(f'{mode}_acc_top5', top5_acc, prog_bar=True)

        self.log(f'{mode}_loss', loss, prog_bar=(mode == 'val'))
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
            total_iters=480000
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=self.hparams.lr / 50
        )
        return [optimizer], [warmup, cosine]
