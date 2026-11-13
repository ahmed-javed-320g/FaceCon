"""FaceCon advanced models (report Section 5.4, models 6-9) in PyTorch.

All predict the FLAME-stand-in shape vector beta (K=50) from the SNP features
(SNP-PCA tokens) and optionally demographics. They are meant to be compared,
baselines-first, against ridge / MLP / demographic baselines: the honest
expectation under the 20% planted genetic share is that they do NOT beat
demographics, which is itself a valid RQ5 result.

  MLPReg            - extended MLP baseline (model 6)
  TransformerSNP    - transformer SNP encoder (model 7)
  Contrastive       - contrastive SNP<->face representation learning (model 8)
  Diffusion         - conditional diffusion over beta given SNPs (model 9)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class MLPReg(nn.Module):
    """Extended MLP: SNP(+demo) -> beta. Dropout + weight decay via optimiser."""
    def __init__(self, p, d, k, hidden=(512, 256, 128), use_demo=True, drop=0.2):
        super().__init__()
        self.use_demo = use_demo
        din = p + (d if use_demo else 0)
        layers, prev = [], din
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(drop)]
            prev = h
        layers += [nn.Linear(prev, k)]
        self.net = nn.Sequential(*layers)

    def forward(self, snp, demo):
        x = torch.cat([snp, demo], 1) if self.use_demo else snp
        return self.net(x)


class TransformerSNP(nn.Module):
    """Transformer SNP encoder: each SNP-PC is a token -> self-attention -> beta."""
    def __init__(self, p, d, k, d_model=128, nhead=4, layers=3, use_demo=True, drop=0.1):
        super().__init__()
        self.use_demo = use_demo
        self.tok = nn.Linear(1, d_model)                       # per-token (scalar PC) embed
        self.pos = nn.Parameter(torch.randn(1, p, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, drop,
                                         batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers)
        head_in = d_model + (d if use_demo else 0)
        self.head = nn.Sequential(nn.Linear(head_in, d_model), nn.GELU(),
                                  nn.Dropout(drop), nn.Linear(d_model, k))

    def forward(self, snp, demo):
        t = self.tok(snp.unsqueeze(-1)) + self.pos             # (B,P,d)
        z = self.enc(t).mean(1)                                # (B,d) pooled
        if self.use_demo:
            z = torch.cat([z, demo], 1)
        return self.head(z)


class Contrastive(nn.Module):
    """CLIP-style: align SNP and face (beta) embeddings; regress beta from z_snp."""
    def __init__(self, p, d, k, zdim=128, use_demo=True):
        super().__init__()
        self.use_demo = use_demo
        din = p + (d if use_demo else 0)
        self.snp_enc = nn.Sequential(nn.Linear(din, 256), nn.GELU(),
                                     nn.Linear(256, zdim))
        self.face_enc = nn.Sequential(nn.Linear(k, 128), nn.GELU(),
                                      nn.Linear(128, zdim))
        self.reg = nn.Sequential(nn.Linear(zdim, 128), nn.GELU(), nn.Linear(128, k))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_snp(self, snp, demo):
        x = torch.cat([snp, demo], 1) if self.use_demo else snp
        return nn.functional.normalize(self.snp_enc(x), dim=1)

    def encode_face(self, beta):
        return nn.functional.normalize(self.face_enc(beta), dim=1)

    def contrastive_loss(self, snp, demo, beta):
        zs = self.encode_snp(snp, demo); zf = self.encode_face(beta)
        logits = self.logit_scale.exp() * zs @ zf.t()
        tgt = torch.arange(len(zs), device=zs.device)
        return 0.5 * (nn.functional.cross_entropy(logits, tgt) +
                      nn.functional.cross_entropy(logits.t(), tgt))

    def forward(self, snp, demo):                              # beta prediction
        x = torch.cat([snp, demo], 1) if self.use_demo else snp
        return self.reg(nn.functional.normalize(self.snp_enc(x), dim=1))


class Diffusion(nn.Module):
    """Conditional DDPM over beta given a SNP(+demo) condition embedding."""
    def __init__(self, p, d, k, T=200, cond_dim=128, use_demo=True):
        super().__init__()
        self.use_demo = use_demo; self.k = k; self.T = T
        din = p + (d if use_demo else 0)
        self.cond = nn.Sequential(nn.Linear(din, 256), nn.GELU(), nn.Linear(256, cond_dim))
        self.net = nn.Sequential(
            nn.Linear(k + cond_dim + 1, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(), nn.Linear(256, k))
        betas = torch.linspace(1e-4, 0.02, T)
        a = torch.cumprod(1 - betas, 0)
        self.register_buffer("betas", betas)
        self.register_buffer("abar", a)

    def cond_emb(self, snp, demo):
        x = torch.cat([snp, demo], 1) if self.use_demo else snp
        return self.cond(x)

    def loss(self, snp, demo, beta):
        c = self.cond_emb(snp, demo); B = beta.size(0)
        t = torch.randint(0, self.T, (B,), device=beta.device)
        ab = self.abar[t].unsqueeze(1)
        noise = torch.randn_like(beta)
        xt = ab.sqrt() * beta + (1 - ab).sqrt() * noise
        inp = torch.cat([xt, c, t.float().unsqueeze(1) / self.T], 1)
        return nn.functional.mse_loss(self.net(inp), noise)

    @torch.no_grad()
    def sample(self, snp, demo):
        c = self.cond_emb(snp, demo); B = c.size(0)
        x = torch.randn(B, self.k, device=c.device)
        for t in reversed(range(self.T)):
            ab = self.abar[t]; beta_t = self.betas[t]
            inp = torch.cat([x, c, torch.full((B, 1), t / self.T, device=c.device)], 1)
            eps = self.net(inp)
            mean = (x - beta_t / (1 - ab).sqrt() * eps) / (1 - beta_t).sqrt()
            x = mean + (beta_t.sqrt() * torch.randn_like(x) if t > 0 else 0)
        return x
