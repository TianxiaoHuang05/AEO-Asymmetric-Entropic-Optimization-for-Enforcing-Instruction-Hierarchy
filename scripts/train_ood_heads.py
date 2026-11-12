#!/usr/bin/env python3
"""Train the IHEval detector heads used for the JBB transfer experiment (paper Table 10).

``--loss aeo`` uses cross-entropy on compliant rows plus the entropy term that
pushes violation predictions toward ln 2; ``--loss ce`` is the Plain-CE baseline
(standard cross-entropy on both classes).  Both use the same MLP, Adam
(lr 5e-4, weight decay 1e-5), batch size 16, and 20 epochs, and keep the epoch
with the best validation accuracy.  The released heads use ``--seed 1``.

Feature files are JSON/JSONL in the repository format (``hd``, ``hd3``,
``response_label``) or NPZ with arrays ``hd``, ``hd3``, and ``label``.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval_ood_jbb import EntropyClassifier, load_features


def loss_fn(logits: torch.Tensor, y: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "ce":
        return F.cross_entropy(logits, y)
    loss = logits.new_zeros(())
    if (y == 0).any():
        loss = loss + F.cross_entropy(logits[y == 0], y[y == 0])
    if (y == 1).any():
        p = F.softmax(logits[y == 1], dim=1)
        loss = loss + (0.693 - (-(p * torch.log(p + 1e-10)).sum(1)).mean())
    return loss


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_features", type=Path, required=True)
    ap.add_argument("--val_features", type=Path, required=True)
    ap.add_argument("--loss", choices=("aeo", "ce"), required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--output_ckpt", type=Path, required=True)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    x_tr, y_tr = load_features(args.train_features)
    x_va, y_va = load_features(args.val_features)
    X, Y = torch.from_numpy(x_tr).float().to(dev), torch.from_numpy(y_tr).long().to(dev)
    Xv, Yv = torch.from_numpy(x_va).float().to(dev), torch.from_numpy(y_va).long().to(dev)

    model = EntropyClassifier(X.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    best_acc, best = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), 16):
            b = perm[i:i + 16]
            loss = loss_fn(model(X[b]), Y[b], args.loss)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.inference_mode():
            acc = float((model(Xv).argmax(1) == Yv).float().mean())
        if acc > best_acc:
            best_acc, best = acc, (epoch, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}))

    epoch, state = best
    meta = {"method": "AEO" if args.loss == "aeo" else "Plain-CE", "seed": args.seed, "epochs": args.epochs,
            "batch_size": 16, "lr": 5e-4, "weight_decay": 1e-5, "selection": "best validation accuracy",
            "best_epoch": epoch, "best_val_acc": best_acc, "input_dim": int(X.shape[1]), "feature_order": "hd||hd3"}
    args.output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": state, **meta}, args.output_ckpt)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
