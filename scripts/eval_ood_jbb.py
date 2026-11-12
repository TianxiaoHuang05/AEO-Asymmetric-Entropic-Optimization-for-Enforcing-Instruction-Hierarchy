#!/usr/bin/env python3
"""Out-of-distribution transfer from IHEval to JBB attack families (paper Table 10).

Scores JBB attack responses with frozen IHEval-trained detectors and reports
AUC/FPR95 (%) for AEO, Plain-CE, MSP, kNN, and one-class.  Positives are
jailbroken attack responses; negatives are the remaining attack responses.

Feature files are JSON/JSONL in the repository format (``hd``, ``hd3``, and a
label) or NPZ with arrays ``hd``, ``hd3``, and ``label``.  The detector input is
``hd || hd3``.  IHEval train features are used only by kNN and one-class, which
fit on the compliant (``follow_system``) training rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


class EntropyClassifier(nn.Module):
    def __init__(self, input_size: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(input_size, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def label(row) -> int | None:
    for key in ("jailbroken", "violation_label"):
        if row.get(key) in (0, 1, False, True):
            return int(row[key])
    raw = str(row.get("response_label", row.get("label", ""))).lower()
    if raw in {"1", "true", "jailbroken", "violate", "violate_system"} or "violate" in raw:
        return 1
    if raw in {"0", "false", "follow", "follow_system", "normal", "ok"} or "follow" in raw:
        return 0
    return None


def load_features(path: Path):
    if path.suffix == ".npz":
        d = np.load(path)
        x = np.concatenate([d["hd"], d["hd3"]], axis=1).astype(np.float32)
        return x, d["label"].astype(int)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
        rows = rows if isinstance(rows, list) else [rows]
    xs, ys = [], []
    for row in rows:
        y = label(row)
        if y is None or row.get("hd") is None or row.get("hd3") is None:
            continue
        xs.append(np.asarray(list(row["hd"]) + list(row["hd3"]), dtype=np.float32)); ys.append(y)
    if not xs:
        raise ValueError(f"No labeled feature rows in {path}")
    return np.stack(xs), np.asarray(ys)


def load_detector(path: Path, dim: int) -> EntropyClassifier:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    state = obj.get("model_state_dict", obj)
    if int(state["net.1.weight"].shape[1]) != dim:
        raise ValueError(f"{path} expects input dim {state['net.1.weight'].shape[1]}, features have {dim}")
    model = EntropyClassifier(dim)
    model.load_state_dict(state)
    return model.eval()


def probs(model: EntropyClassifier, x: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        return torch.softmax(model(torch.from_numpy(x)), dim=1).numpy()


def metrics(y: np.ndarray, s: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(y, s)
    return {"auc": round(100 * float(roc_auc_score(y, s)), 2),
            "fpr95": round(100 * float(fpr[np.flatnonzero(tpr >= 0.95)[0]]), 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_features", type=Path, required=True, help="IHEval train features (kNN / one-class reference).")
    ap.add_argument("--jbb_features", type=Path, required=True, help="JBB attack-response features with jailbroken labels.")
    ap.add_argument("--aeo_checkpoint", type=Path, required=True)
    ap.add_argument("--plain_ce_checkpoint", type=Path, required=True)
    ap.add_argument("--k", type=int, default=15, help="Neighbour rank used as the kNN distance score.")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    x_tr, y_tr = load_features(args.train_features)
    x, y = load_features(args.jbb_features)
    if len(set(y.tolist())) != 2:
        raise ValueError("JBB features need both jailbroken and non-jailbroken responses")
    compliant = x_tr[y_tr == 0]

    p_aeo = probs(load_detector(args.aeo_checkpoint, x.shape[1]), x)
    p_ce = probs(load_detector(args.plain_ce_checkpoint, x.shape[1]), x)
    knn = NearestNeighbors(n_neighbors=args.k, metric="cosine").fit(compliant)
    scaler = StandardScaler().fit(compliant)
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1).fit(scaler.transform(compliant))
    scores = {
        "AEO": p_aeo[:, 1],
        "Plain-CE": p_ce[:, 1],
        "kNN": knn.kneighbors(x)[0][:, -1],
        "one-class": -ocsvm.decision_function(scaler.transform(x)),
        "MSP": 1 - p_ce.max(axis=1),
    }
    result = {"n_attack": int(len(y)), "n_jailbroken": int(y.sum()),
              "metrics": {name: metrics(y, s) for name, s in scores.items()}}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
