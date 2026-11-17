"""Unified training harness for the FaceCon GPU models.

Examples (run on the GPU box from the project root):
    python -m gpu.train --model transformer --epochs 150 --log-every 10
    python -m gpu.train --model all --epochs 150 --out gpu/results.json

Trains the selected advanced model(s), streams progress every --log-every
epochs (train loss + held-out validation R2), evaluates on the TEST split, and
prints a comparison table against the mean-face / demographics / ridge
baselines. Honest framing: under the 20% planted genetic share, the advanced
models are expected NOT to beat the demographics baseline -- that is a valid
RQ5 finding, not a failure.
"""
from __future__ import annotations

import argparse, copy, json, sys, time
import numpy as np, torch
from sklearn.linear_model import Ridge, LinearRegression

from .data import FaceConData
from .models import MLPReg, TransformerSNP, Contrastive, Diffusion

MODELS = ["mlp", "transformer", "contrastive", "diffusion"]


def _param_r2(bt, bp):
    bt = np.asarray(bt, np.float32); bp = np.asarray(bp, np.float32)
    ss_res = ((bt - bp) ** 2).sum(0)
    ss_tot = ((bt - bt.mean(0)) ** 2).sum(0) + 1e-9
    return float((1 - ss_res / ss_tot).mean())


def baselines(data: FaceConData):
    snp_tr, demo_tr, b_tr = [t.cpu().numpy() for t in data.tensors("train")]
    snp_te, demo_te, b_te = [t.cpu().numpy() for t in data.tensors("test")]
    rows = {}
    rows["mean_face"] = data.evaluate(b_te, np.repeat(b_tr.mean(0, keepdims=True), len(b_te), 0))
    lr = LinearRegression().fit(demo_tr, b_tr)
    rows["demographics"] = data.evaluate(b_te, lr.predict(demo_te))
    rg = Ridge(alpha=10.0).fit(snp_tr, b_tr)
    rows["ridge_snp"] = data.evaluate(b_te, rg.predict(snp_te))
    # Linear model given EXACTLY what the deep nets get (demographics + SNP
    # features), alpha picked on the validation split. This is the baseline the
    # deep models have to beat to justify their complexity; ridge_snp above
    # withholds the covariates and so understates the linear option.
    # Two-stage, as in the controls ladder: covariates unpenalised by OLS, then
    # ridge on the SNP features against what the covariates left over. A plain
    # ridge on [demo, snp] penalises the covariates too, so at the large alpha
    # that noise features demand it falls BELOW the demographics baseline and
    # understates the linear option exactly where the comparison matters.
    snp_va, demo_va, b_va = [t.cpu().numpy() for t in data.tensors("val")]
    r_tr = b_tr - lr.predict(demo_tr)
    best, best_r2 = None, -np.inf
    for alpha in (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6):
        m = Ridge(alpha=alpha).fit(snp_tr, r_tr)
        r2 = _param_r2(b_va, lr.predict(demo_va) + m.predict(snp_va))
        if r2 > best_r2:
            best, best_r2 = m, r2
    rows["ridge_demo_snp"] = data.evaluate(b_te, lr.predict(demo_te) + best.predict(snp_te))
    rows["ridge_demo_snp"]["alpha"] = float(best.alpha)
    return rows


def _predict(net, name, snp, demo):
    return net.sample(snp, demo) if name == "diffusion" else net(snp, demo)


def train_torch(name, data, epochs, lr, batch, use_demo, device,
                log_every=10, patience=0, quiet=False):
    """Train one model with best-on-validation model selection.

    The TEST metrics are always computed on the checkpoint with the best
    held-out val_R2 -- never the final (possibly overfit) epoch. This is the
    standard way to report deep-model results fairly: without it, a model that
    peaks at epoch 40 and then overfits for 110 more epochs is judged on its
    worst self. ``patience`` > 0 also stops training early once val_R2 has not
    improved for that many consecutive evaluations (saving GPU time); patience
    == 0 trains the full schedule but still reports the best checkpoint.
    """
    snp_tr, demo_tr, b_tr = data.tensors("train")
    snp_va, demo_va, b_va = data.tensors("val")
    snp_te, demo_te, b_te = data.tensors("test")
    P, D, K = data.P, data.D, data.K
    net = {"mlp": MLPReg, "transformer": TransformerSNP,
           "contrastive": Contrastive, "diffusion": Diffusion}[name](
        P, D, K, use_demo=use_demo).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    n = snp_tr.size(0); t0 = time.time()
    b_va_np = b_va.cpu().numpy()
    best_r2, best_state, best_ep, no_improve = -1e30, None, 0, 0
    _print = (lambda *x, **k: None) if quiet else print
    _print(f"[{name}] training {epochs} epochs on {device} "
          f"({sum(p.numel() for p in net.parameters())/1e3:.0f}k params"
          f"{'; patience='+str(patience) if patience else ''})", flush=True)
    for ep in range(1, epochs + 1):
        net.train(); perm = torch.randperm(n, device=device); tot = 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            sn, dm, bt = snp_tr[idx], demo_tr[idx], b_tr[idx]
            opt.zero_grad()
            if name == "contrastive":
                loss = net.contrastive_loss(sn, dm, bt) + \
                    torch.nn.functional.mse_loss(net(sn, dm), bt)
            elif name == "diffusion":
                loss = net.loss(sn, dm, bt)
            else:
                loss = torch.nn.functional.mse_loss(net(sn, dm), bt)
            loss.backward(); opt.step(); tot += float(loss.detach()) * len(idx)
        if ep % log_every == 0 or ep == epochs:
            net.eval()
            with torch.no_grad():
                vp = _predict(net, name, snp_va, demo_va).cpu().numpy()
            vr2 = _param_r2(b_va_np, vp)
            if vr2 > best_r2:
                best_r2, best_ep, no_improve = vr2, ep, 0
                best_state = copy.deepcopy(net.state_dict())
                flag = " *best"
            else:
                no_improve += 1
                flag = ""
            _print(f"[{name}] epoch {ep:>4}/{epochs}  train_loss={tot/n:.4f}  "
                  f"val_R2={vr2:+.4f}  ({time.time()-t0:.0f}s){flag}", flush=True)
            if patience and no_improve >= patience:
                _print(f"[{name}] early stop at epoch {ep} "
                      f"(no val gain for {patience} evals; "
                      f"best val_R2={best_r2:+.4f} @ep{best_ep})", flush=True)
                break
    if best_state is not None:
        net.load_state_dict(best_state)       # restore best-on-val checkpoint
    net.eval()
    with torch.no_grad():
        pred = _predict(net, name, snp_te, demo_te).cpu().numpy()
    m = data.evaluate(b_te.cpu().numpy(), pred)
    m["train_sec"] = round(time.time() - t0, 1)
    m["best_epoch"] = best_ep
    m["best_val_r2"] = round(best_r2, 4)
    _print(f"[{name}] TEST on best-val checkpoint (ep{best_ep}): "
          f"paramR2={m['param_r2']}  mesh_mm={m['mesh_mm']}", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=["all"] + MODELS)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--snp-pca", type=int, default=256, help="number of SNP features")
    ap.add_argument("--features", default="pca", choices=["pca", "screen"],
                    help="pca = SNP principal components (discards most causal signal on "
                         "unlinked SNPs); screen = train-only association screen")
    ap.add_argument("--log-every", type=int, default=10, help="print status every N epochs")
    ap.add_argument("--patience", type=int, default=0,
                    help="early stop after N evals (each = log-every epochs) with no "
                         "val gain; 0 = run full schedule but still report best-val model")
    ap.add_argument("--no-demo", action="store_true")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="gpu/results.json")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  snp_pca={a.snp_pca}  use_demo={not a.no_demo}  "
          f"log_every={a.log_every}", flush=True)
    data = FaceConData(root=a.root, snp_pca=a.snp_pca, device=device, features=a.features)
    print(f"features={a.features}", flush=True)
    print(f"N={data.N} M={data.M} P={data.P} D={data.D} K={data.K} "
          f"splits={len(data.tr)}/{len(data.va)}/{len(data.te)}", flush=True)

    results = baselines(data)
    print("baselines: demographics R2=%.3f  ridge R2=%.3f" % (
        results["demographics"]["param_r2"], results["ridge_snp"]["param_r2"]), flush=True)
    todo = MODELS if a.model == "all" else [a.model]
    for m in todo:
        results[m] = train_torch(m, data, a.epochs, a.lr, a.batch,
                                 not a.no_demo, device, a.log_every, a.patience)

    order = ["mean_face", "demographics", "ridge_snp"] + todo
    print("\n%-14s %8s %9s %10s %8s %9s" % (
        "model", "paramR2", "mesh_mm", "top1", "bestEp", "sec"))
    for k in order:
        r = results.get(k, {})
        print("%-14s %8s %9s %10s %8s %9s" % (
            k, r.get("param_r2", "-"), r.get("mesh_mm", "-"),
            r.get("retrieval_top1", "-"), r.get("best_epoch", "-"),
            r.get("train_sec", "-")))
    json.dump(results, open(a.out, "w"), indent=2)
    print(f"\nsaved {a.out}", flush=True)


if __name__ == "__main__":
    main()
