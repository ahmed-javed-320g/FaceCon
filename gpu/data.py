"""Data loading + evaluation metrics for the FaceCon GPU models.

Reads the matched synthetic dataset (outputs/real_match/dataset.npz) and the
face model (outputs/real_match/face_model.npz, or assets/ict/ict_face_model.npz).
Everything is leak-safe: SNP standardisation and PCA are fit on the TRAIN split
only, then applied to val/test. The prediction target is the FLAME-stand-in
shape vector beta (50-dim); metrics mirror the CPU pipeline (param R2, per-vertex
mesh error in nominal mm, retrieval top-k) so GPU models are directly comparable
to the ridge / MLP / demographic baselines.
"""
from __future__ import annotations

import os
import numpy as np
import torch

from sklearn.decomposition import PCA


def _find(path_candidates):
    for p in path_candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"None of these exist: {path_candidates}")


class FaceConData:
    def __init__(self, root=".", snp_pca=256, device="cpu", seed=20261016,
                 identity_scale=None, features="pca", data_dir="outputs/real_match",
                 train_n=None, permute_snps=False, split_seed=None):
        self.device = device
        if identity_scale is None:                      # read it from the config
            import yaml
            cfgp = os.path.join(root, "config/real_match.yaml")
            identity_scale = (yaml.safe_load(open(cfgp))["face_model"].get("identity_scale", 1.0)
                              if os.path.isfile(cfgp) else 1.0)
        ds_path = _find([os.path.join(root, data_dir, "dataset.npz"),
                         os.path.join(root, "dataset.npz")])
        fm_path = _find([os.path.join(root, "outputs/real_match/face_model.npz"),
                         os.path.join(root, "assets/ict/ict_face_model.npz"),
                         os.path.join(root, "face_model.npz")])
        d = np.load(ds_path, allow_pickle=True)
        self.geno = d["genotypes"].astype(np.float32)          # (N, M) with NaN
        self.demo = d["demographics"].astype(np.float32)       # (N, D)
        self.beta = d["face_params"].astype(np.float32)        # (N, K) target
        self.tr = d["split_train"]; self.va = d["split_val"]; self.te = d["split_test"]
        if split_seed is not None:
            # Re-draw train/val/test exactly as scripts/generate_matched._make_splits does,
            # from the named "splits" stream. split_seed == the cohort's own seed reproduces
            # the stored split bit-for-bit (tests/test_seeds.py); other values resample it.
            import yaml
            from src.common import rng_for
            cfgp = os.path.join(root, "config/real_match.yaml")
            sp = yaml.safe_load(open(cfgp))["split"] if os.path.isfile(cfgp) else {"train": .7, "val": .15}
            n = len(self.beta)
            idx = rng_for(int(split_seed), "splits").permutation(n)
            ntr, nva = int(sp["train"] * n), int(sp["val"] * n)
            self.tr, self.va, self.te = (np.sort(idx[:ntr]), np.sort(idx[ntr:ntr + nva]),
                                         np.sort(idx[ntr + nva:]))
        if permute_snps:        # NEGATIVE CONTROL: break the genotype<->face pairing
            self.geno = self.geno[np.random.default_rng(seed + 1).permutation(len(self.geno))]
        if train_n and train_n < len(self.tr):   # learning curve: nested train subsets,
            order = np.random.default_rng(seed).permutation(len(self.tr))   # val/test fixed
            self.tr = np.sort(self.tr[order[:int(train_n)]])
        self.N, self.M = self.geno.shape
        self.K = self.beta.shape[1]

        # ---- impute + standardise SNPs on TRAIN only ----
        # Done in place on one float32 array: with real genotype panels
        # (e.g. 2,573 x 100,000) every extra full copy costs about 1 GB.
        g = self.geno
        self.geno = None
        col_mean = np.nanmean(g[self.tr], axis=0)
        inds = np.where(np.isnan(g))
        g[inds] = np.take(col_mean, inds[1])                   # impute with train mean
        gt = g[self.tr]
        mu = gt.mean(0); sd = gt.std(0) + 1e-6
        del gt
        g -= mu
        g /= sd

        # ---- SNP features, fit on TRAIN only ----
        # "pca"   : top principal components. On unlinked SNPs these keep ancestry
        #           structure but DISCARD most per-SNP causal signal (gpu/controls.py
        #           measures this: ~+0.004 R2 kept of ~+0.17 available).
        # "screen": the snp_pca SNPs most associated with shape on the TRAIN split,
        #           after removing demographic effects (a train-only GWAS-style
        #           screen; val/test never influence the selection).
        # "screen_na": the same idea with a 2-df genotypic score (additive dosage AND
        #           heterozygote indicator per SNP), keeping snp_pca // 2 SNPs and giving
        #           the network BOTH codes, interleaved [a1, h1, a2, h2, ...], so it sees
        #           the same number of columns as "screen". Like "screen" it residualises
        #           on the full demographics, whereas gpu/controls.py residualises on
        #           cov + ancestry PCs; that existing difference is kept rather than
        #           switching conventions for one variant only.
        self.pca = None
        self.features = features
        if features == "screen" and snp_pca and snp_pca < self.M:
            C = np.c_[np.ones(len(self.tr)), self.demo[self.tr]].astype(np.float64)
            def _res(M):
                b, *_ = np.linalg.lstsq(C, M, rcond=None)
                R = M - C @ b
                return (R - R.mean(0)) / (R.std(0) + 1e-9)
            Yr = _res(self.beta[self.tr].astype(np.float64))
            score = np.empty(self.M)
            for s in range(0, self.M, 5000):                     # column chunks: bounded memory
                blk = g[self.tr, s:s + 5000].astype(np.float64)
                score[s:s + 5000] = ((_res(blk).T @ Yr) ** 2).sum(1)
            self.screen_idx = np.sort(np.argsort(-score)[:snp_pca])
            self.snp_feat = g[:, self.screen_idx].astype(np.float32)
        elif features == "screen_na" and snp_pca and snp_pca // 2 < self.M:
            C = np.c_[np.ones(len(self.tr)), self.demo[self.tr]].astype(np.float64)
            def _res(M):
                b, *_ = np.linalg.lstsq(C, M, rcond=None)
                R = M - C @ b
                return (R - R.mean(0)) / (R.std(0) + 1e-9)
            het = lambda raw: (np.abs(raw - 1.0) < 0.5)        # band: imputed cells are non-integer
            Yr = _res(self.beta[self.tr].astype(np.float64))
            ntr = len(self.tr)
            score = np.full(self.M, -np.inf)
            for s in range(0, self.M, 5000):
                e = min(self.M, s + 5000)
                blk = g[self.tr, s:e].astype(np.float64)
                hraw = het(blk * sd[s:e] + mu[s:e])              # back to dosages for the het code
                a, h = _res(blk), _res(hraw.astype(np.float64))
                Sah = np.clip((a * h).mean(0), -0.99, 0.99)
                r_a, r_h = a.T @ Yr / ntr, h.T @ Yr / ntr
                sc = ((r_a ** 2 - 2 * Sah[:, None] * r_a * r_h + r_h ** 2)
                      / (1 - Sah ** 2 + 1e-9)[:, None]).sum(1)
                sc[hraw.sum(0) < 10] = -np.inf                   # too few heterozygotes to trust
                score[s:e] = sc
            self.screen_idx = np.sort(np.argsort(-score)[:snp_pca // 2])
            A = g[:, self.screen_idx]
            H = het(A * sd[self.screen_idx] + mu[self.screen_idx]).astype(np.float32)
            H = (H - H[self.tr].mean(0)) / (H[self.tr].std(0) + 1e-6)   # train-only scaling
            self.snp_feat = np.empty((self.N, 2 * len(self.screen_idx)), np.float32)
            self.snp_feat[:, 0::2] = A; self.snp_feat[:, 1::2] = H
        elif snp_pca and snp_pca < self.M:
            self.pca = PCA(n_components=snp_pca, random_state=seed).fit(g[self.tr])
            self.snp_feat = self.pca.transform(g).astype(np.float32)
        else:
            self.snp_feat = g
        del g
        self.P = self.snp_feat.shape[1]
        self.D = self.demo.shape[1]

        # ---- face model (for mesh-error metric) ----
        f = np.load(fm_path, allow_pickle=True)
        self.mean = f["mean"].astype(np.float32)               # (V,3)
        # The generator scales the identity basis by face_model.identity_scale
        # (src/faces/morphable.py). Apply the SAME factor here, otherwise mesh
        # errors are under-reported by that factor (was 9.5x too small).
        self.basis = f["basis"].astype(np.float32) * float(identity_scale)   # (K,3V)
        self.mm_per_unit = float(f["mm_per_unit"]) if "mm_per_unit" in f.files else 100.0
        self.V = self.mean.shape[0]

    def tensors(self, split):
        idx = {"train": self.tr, "val": self.va, "test": self.te}[split]
        snp = torch.tensor(self.snp_feat[idx], device=self.device)
        demo = torch.tensor(self.demo[idx], device=self.device)
        beta = torch.tensor(self.beta[idx], device=self.device)
        return snp, demo, beta

    # -------- metrics --------
    def decode_mm(self, beta_np):
        """beta (B,K) -> vertices (B,V,3) in nominal mm."""
        disp = (beta_np @ self.basis).reshape(-1, self.V, 3)
        return (self.mean[None] + disp) * self.mm_per_unit

    def evaluate(self, beta_true, beta_pred):
        """Return param R2, mesh mean mm, landmark-free vertex median, retrieval top1."""
        bt = np.asarray(beta_true, np.float32); bp = np.asarray(beta_pred, np.float32)
        # param R2 (uniform average over dims)
        ss_res = ((bt - bp) ** 2).sum(0)
        ss_tot = ((bt - bt.mean(0)) ** 2).sum(0) + 1e-9
        param_r2 = float((1 - ss_res / ss_tot).mean())
        # mesh error (decode in chunks to save memory)
        errs = []
        for s in range(0, len(bt), 256):
            vt = self.decode_mm(bt[s:s + 256]); vp = self.decode_mm(bp[s:s + 256])
            errs.append(np.linalg.norm(vp - vt, axis=2).mean(1))
        mesh_mm = float(np.concatenate(errs).mean())
        # retrieval top-1 by beta distance (gallery = all test)
        d2 = ((bp[:, None, :] - bt[None, :, :]) ** 2).sum(-1)
        top1 = float((d2.argmin(1) == np.arange(len(bt))).mean())
        return {"param_r2": round(param_r2, 4), "mesh_mm": round(mesh_mm, 3),
                "retrieval_top1": round(top1, 4)}
