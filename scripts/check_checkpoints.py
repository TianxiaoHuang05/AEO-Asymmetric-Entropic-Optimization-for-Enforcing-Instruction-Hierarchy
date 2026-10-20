#!/usr/bin/env python3
"""Load every curated detector checkpoint without executing serialized code."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    rows = []
    for path in sorted((args.root / "checkpoints").rglob("*.pt")):
        obj = torch.load(path, map_location="cpu", weights_only=True)
        state = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj
        weight = state.get("net.1.weight") if isinstance(state, dict) else None
        if weight is None or len(weight.shape) != 2:
            raise RuntimeError(f"No detector first layer in {path}")
        rows.append({"path": str(path.relative_to(args.root)), "bytes": path.stat().st_size,
                     "input_dim": int(weight.shape[1]), "output_dim": int(weight.shape[0])})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
