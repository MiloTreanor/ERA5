import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from .encoder import load_encoder

class SimCLROneSide(pl.LightningModule):
    def __init__(self, hidden_dim=128, lr=5e-4, temperature=0.07,
                 weight_decay=1e-4, max_epochs=500):
        super().__init__()
        self.save_hyperparameters()

        # 1) Load your backbone encoder
        self.encoder = load_encoder()
        with torch.no_grad():
            dummy = torch.randn(2, 12, 288, 288)
            feat_map = self.encoder(dummy)           # [B, C, H, W]
            feat_dim = feat_map.mean(dim=(2, 3)).shape[1]

        # 2) Projection head MLP
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, 4 * hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def info_nce_one_sided(self, batch, mode='train'):
        x_a, x_b = batch                     # each is [N, C, H, W]
        N = x_a.size(0)

        # 1) Embed both views
        imgs = torch.cat([x_a, x_b], dim=0)  # [2N, C, H, W]
        feats = self.encoder(imgs)           # [2N, C, H, W]
        feats = feats.mean(dim=(2, 3))       # [2N, C]
        z = self.projection(feats)           # [2N, D]
        z = F.normalize(z, dim=1)            # unit vectors

        # 2) Full 2N×2N similarity logits
        logits = torch.mm(z, z.T) / self.hparams.temperature

        # 3) Mask out self-similarities
        eye = torch.eye(2 * N, device=logits.device, dtype=torch.bool)
        neg_inf = torch.tensor(-1e4, dtype=logits.dtype, device=logits.device)
        logits = logits.masked_fill(eye, neg_inf)

        # 4) Only keep first N rows (one-sided anchors)
        logits_onesided = logits[:N]         # [N, 2N]

        # 5) Build labels: for anchor i, positive is at index i+N
        labels = torch.arange(N, device=logits.device) + N  # [N]

        # 6) NT-Xent via cross-entropy
        loss = F.cross_entropy(logits_onesided, labels)
        self.log(f'{mode}_loss', loss)

        # 7) Retrieval metrics on those N anchors
        #    rank of the true-positive logit among all 2N
        with torch.no_grad():
            # Gather pos sim and mask them in a competitor matrix
            pos_sim = logits_onesided[torch.arange(N), labels].unsqueeze(1)  # [N,1]
            neg_sim = logits_onesided.clone()
            neg_sim[torch.arange(N), labels] = neg_inf                      # mask pos
            all_sim = torch.cat([pos_sim, neg_sim], dim=1)                  # [N, 2N]
            ranks = all_sim.argsort(dim=1, descending=True).argmin(dim=1)   # [N]

            self.log(f'{mode}_acc_top1', (ranks == 0).float().mean())
            self.log(f'{mode}_acc_top5', (ranks < 5).float().mean())

        return loss

    def training_step(self, batch, batch_idx):
        return self.info_nce_one_sided(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self.info_nce_one_sided(batch, mode='val')

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