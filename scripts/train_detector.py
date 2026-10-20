import argparse
import datetime
import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_label(item: Dict) -> Optional[int]:
    if "response_label" in item:
        rl = str(item["response_label"]).lower()
        if "violate" in rl or rl == "1":
            return 1
        if "follow" in rl or rl == "0":
            return 0
    if "label" in item:
        ll = str(item["label"]).lower()
        if ll in {"conflict", "violate", "1"}:
            return 1
        if ll in {"normal", "ok", "0"}:
            return 0
    return None


def expand_json_files(paths: List[str]) -> List[str]:
    files: List[str] = []
    for path in paths:
        if not os.path.exists(path):
            print(f"[WARN] Path not found, skip: {path}", flush=True)
            continue
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                if name.endswith(".json"):
                    files.append(os.path.join(path, name))
        elif os.path.isfile(path) and path.endswith(".json"):
            files.append(path)
    return files


def read_hd_data(paths: List[str]) -> List[Dict]:
    items: List[Dict] = []
    files = expand_json_files(paths)
    print(f"[INFO] Loading {len(files)} json files.", flush=True)

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"[WARN] Failed to read {file_path}: {exc}", flush=True)
            continue

        if isinstance(data, dict):
            data = [data]

        kept = 0
        for it in data:
            hd = it.get("hd")
            hd3 = it.get("hd3")
            if hd is None or hd3 is None:
                continue
            lab = parse_label(it)
            if lab is None:
                continue
            vec = np.asarray(hd + hd3, dtype=np.float32)
            items.append({"input": vec, "label": int(lab), "source_file": file_path})
            kept += 1
        print(f"[INFO] {os.path.basename(file_path)} -> kept {kept} samples", flush=True)

    return items


class MindDataset(Dataset):
    def __init__(self, rows: List[Dict]):
        if not rows:
            raise ValueError("Empty dataset.")
        x_np = np.stack([r["input"] for r in rows], axis=0)
        y_np = np.asarray([r["label"] for r in rows], dtype=np.int64)
        self.x = torch.from_numpy(x_np).float()
        self.y = torch.from_numpy(y_np).long()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class EntropyClassifier(nn.Module):
    def __init__(self, input_size: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


def safe_roc_auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    if labels.size == 0 or len(np.unique(labels)) <= 1:
        return None
    return float(roc_auc_score(labels, scores) * 100.0)


def fpr95(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    if labels.size == 0 or len(np.unique(labels)) <= 1:
        return None
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    idx = np.where(tpr >= 0.95)[0]
    if idx.size == 0:
        return None
    return float(fpr[idx[0]] * 100.0)


def tpr_at_fpr5(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    if labels.size == 0 or len(np.unique(labels)) <= 1:
        return None
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    if fpr.size == 0:
        return None
    return float(np.interp(0.05, fpr, tpr) * 100.0)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict:
    model.eval()
    all_scores: List[float] = []
    all_labels: List[int] = []
    all_preds: List[int] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        all_scores.extend(probs.detach().cpu().tolist())
        all_labels.extend(y.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    labels_np = np.asarray(all_labels, dtype=np.int64)
    scores_np = np.asarray(all_scores, dtype=np.float64)
    preds_np = np.asarray(all_preds, dtype=np.int64)

    if labels_np.size == 0:
        return {
            "val_acc": None,
            "val_auc": None,
            "val_fpr95": None,
            "val_tpr95": None,
            "num_samples": 0,
            "num_pos": 0,
            "num_neg": 0,
        }

    acc = float((preds_np == labels_np).mean() * 100.0)
    return {
        "val_acc": acc,
        "val_auc": safe_roc_auc(labels_np, scores_np),
        "val_fpr95": fpr95(labels_np, scores_np),
        "val_tpr95": tpr_at_fpr5(labels_np, scores_np),
        "num_samples": int(labels_np.size),
        "num_pos": int((labels_np == 1).sum()),
        "num_neg": int((labels_np == 0).sum()),
    }


def format_optional(v: Optional[float], precision: int = 2) -> str:
    if v is None:
        return "NA"
    return f"{v:.{precision}f}"


def build_loss(logits: torch.Tensor, labels: torch.Tensor, entropy_weight: float) -> torch.Tensor:
    mask0 = labels == 0
    mask1 = labels == 1
    loss = torch.tensor(0.0, dtype=logits.dtype, device=logits.device)

    if mask0.any():
        loss = loss + F.cross_entropy(logits[mask0], labels[mask0])
    if mask1.any():
        probs1 = F.softmax(logits[mask1], dim=1)
        entropy1 = -torch.sum(probs1 * torch.log(probs1 + 1e-10), dim=1).mean()
        loss = loss + entropy_weight * (0.693 - entropy1)
    return loss


def main():
    parser = argparse.ArgumentParser(description="NeurIPS full retrain for AEO detector with per-epoch AUC/FPR95 logging.")
    parser.add_argument("--train_paths", type=str, nargs="+", required=True)
    parser.add_argument("--val_paths", type=str, nargs="+", required=True)
    parser.add_argument("--output_ckpt", type=str, required=True)
    parser.add_argument("--metrics_out", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--entropy_weight", type=float, default=1.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_seed(args.seed)
    run_start = time.perf_counter()

    os.makedirs(os.path.dirname(args.output_ckpt), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}", flush=True)
    print(f"[INFO] train_paths={args.train_paths}", flush=True)
    print(f"[INFO] val_paths={args.val_paths}", flush=True)

    train_rows = read_hd_data(args.train_paths)
    val_rows = read_hd_data(args.val_paths)
    if not train_rows:
        raise RuntimeError("No valid training samples loaded.")
    if not val_rows:
        raise RuntimeError("No valid validation samples loaded.")

    input_dim = int(train_rows[0]["input"].shape[0])
    if any(int(row["input"].shape[0]) != input_dim for row in train_rows + val_rows):
        raise RuntimeError("Input feature dimensions are inconsistent across samples.")

    train_loader = DataLoader(MindDataset(train_rows), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(MindDataset(val_rows), batch_size=args.batch_size, shuffle=False)

    model = EntropyClassifier(input_size=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    detector_params = int(sum(p.numel() for p in model.parameters()))

    print(
        f"[INFO] input_dim={input_dim} detector_params={detector_params} "
        f"train={len(train_rows)} val={len(val_rows)}",
        flush=True,
    )

    history: List[Dict] = []
    best_auc = -1.0
    best_acc = -1.0
    best_epoch = -1
    train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = build_loss(logits, y, args.entropy_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / max(1, num_batches)
        val_stats = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_stats,
        }
        history.append(row)

        print(
            "[EPOCH] "
            f"epoch={epoch} "
            f"loss={train_loss:.6f} "
            f"val_auc={format_optional(val_stats['val_auc'])} "
            f"val_fpr95={format_optional(val_stats['val_fpr95'])} "
            f"val_tpr95={format_optional(val_stats['val_tpr95'])} "
            f"val_acc={format_optional(val_stats['val_acc'])}",
            flush=True,
        )

        auc_key = val_stats["val_auc"] if val_stats["val_auc"] is not None else -1.0
        acc_key = val_stats["val_acc"] if val_stats["val_acc"] is not None else -1.0
        if (auc_key > best_auc) or (auc_key == best_auc and acc_key >= best_acc):
            best_auc = auc_key
            best_acc = acc_key
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "detector_params": detector_params,
                    "best_epoch": best_epoch,
                    "best_val_auc": None if best_auc < 0 else best_auc,
                    "best_val_acc": None if best_acc < 0 else best_acc,
                    "timestamp": datetime.datetime.now().isoformat(),
                },
                args.output_ckpt,
            )
            print(
                f"[CHECKPOINT] New best saved at epoch={epoch} "
                f"val_auc={format_optional(val_stats['val_auc'])} val_acc={format_optional(val_stats['val_acc'])}",
                flush=True,
            )

    train_time_sec = time.perf_counter() - train_start
    total_runtime_sec = time.perf_counter() - run_start
    ckpt_size_mb = round(os.path.getsize(args.output_ckpt) / 1024 / 1024, 6) if os.path.exists(args.output_ckpt) else None

    metrics = {
        "output_ckpt": args.output_ckpt,
        "input_dim": input_dim,
        "detector_params": detector_params,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "entropy_weight": args.entropy_weight,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_auc": None if best_auc < 0 else best_auc,
        "best_val_acc": None if best_acc < 0 else best_acc,
        "train_time_sec": train_time_sec,
        "total_runtime_sec": total_runtime_sec,
        "checkpoint_size_mb": ckpt_size_mb,
        "epoch_history": history,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[DONE] checkpoint={args.output_ckpt}", flush=True)
    print(f"[DONE] metrics={args.metrics_out}", flush=True)
    print(
        f"[DONE] best_epoch={best_epoch} best_val_auc={format_optional(metrics['best_val_auc'])} "
        f"best_val_acc={format_optional(metrics['best_val_acc'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
