#!/usr/bin/env python3
"""CPU scoring for curated AEO detector checkpoints.

The checkpoint input dimension is inferred from ``net.1.weight`` and must
match ``len(hd) + len(hd3)`` in every record.  Score 1 means violation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve


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


def records(path: Path):
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else [data]


def label(row):
    if row.get("violation_label") in (0, 1):
        return int(row["violation_label"])
    raw = str(row.get("response_label", row.get("label", ""))).lower()
    if raw in {"1", "violate", "violate_system", "conflict"} or "violate" in raw:
        return 1
    if raw in {"0", "follow", "follow_system", "normal", "ok"} or "follow" in raw:
        return 0
    return None


def checkpoint_state(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        obj = obj["model_state_dict"]
    if not isinstance(obj, dict) or "net.1.weight" not in obj:
        raise ValueError(f"Unsupported detector checkpoint: {path}")
    return obj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--features", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--label-key", choices=("response_label", "label", "violation_label"), default=None,
                    help="Select the response-level label field when the input has multiple label fields.")
    args = ap.parse_args()

    state = checkpoint_state(args.checkpoint)
    dim = int(state["net.1.weight"].shape[1])
    model = EntropyClassifier(dim)
    model.load_state_dict(state)
    model.eval()

    xs, ys, ids = [], [], []
    for path in args.features:
        for row in records(path):
            y = label({args.label_key: row.get(args.label_key)}) if args.label_key else label(row)
            if y is None or row.get("hd") is None or row.get("hd3") is None:
                continue
            x = np.asarray(list(row["hd"]) + list(row["hd3"]), dtype=np.float32)
            if x.shape != (dim,) or not np.isfinite(x).all():
                raise ValueError(f"Feature shape/non-finite value mismatch in {path}: {x.shape}, expected {(dim,)}")
            xs.append(x); ys.append(y); ids.append(row.get("sample_id", len(ids)))
    if not xs:
        raise ValueError("No labeled feature rows found")

    with torch.inference_mode():
        score = F.softmax(model(torch.from_numpy(np.stack(xs))), dim=1)[:, 1].numpy()
    result = {
        "checkpoint": str(args.checkpoint),
        "input_dim": dim,
        "score_definition": "softmax(logits)[:,1]; higher means response-level violation",
        "n": len(ys),
        "n_violation": int(sum(y == 1 for y in ys)),
        "n_compliant": int(sum(y == 0 for y in ys)),
        "scores": [{"sample_id": i, "label": int(y), "score": float(s)} for i, y, s in zip(ids, ys, score)],
    }
    if len(set(ys)) == 2:
        fpr, tpr, thresholds = roc_curve(np.asarray(ys), score, pos_label=1, drop_intermediate=False)
        idx = int(np.flatnonzero(tpr >= 0.95)[0])
        result.update({
            "roc_auc": float(roc_auc_score(ys, score)),
            "fpr95": float(fpr[idx]),
            "fpr95_threshold": float(thresholds[idx]),
            "fpr95_tpr": float(tpr[idx]),
        })
    else:
        result.update({"roc_auc": None, "fpr95": None, "fpr95_threshold": None, "fpr95_tpr": None})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("input_dim", "n", "n_violation", "n_compliant", "roc_auc", "fpr95")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
