"""
Train the UQ head MLP from collected (iou_token, mask_token, gt_iou) records.

Usage (run from UncSANSA/UncSANSA/):
    python tools/train_uq_head.py \
        --token_files data/uq_tokens/coco_fold0.pt data/uq_tokens/coco_fold1.pt \
        --output models/uq_heads/coco.pth \
        --epochs 60 --lr 5e-5 --batch_size 128

The resulting .pth file can be passed to inference via:
    python inference_fss.py ... --uq_head_path models/uq_heads/coco.pth
"""

import argparse
import os
import sys
from os.path import dirname
sys.path.insert(0, dirname(dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

import importlib.util as _ilu, pathlib as _pl
_spec = _ilu.spec_from_file_location(
    "uq_head",
    _pl.Path(__file__).resolve().parent.parent / "models" / "sansa" / "uq_head.py",
)
_mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
UQHead = _mod.UQHead


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TokenDataset(Dataset):
    """Wraps a list of {iou_token, mask_token, gt_iou} dicts."""

    def __init__(self, records):
        self.iou_tokens  = torch.cat([r["iou_token"]  for r in records], dim=0)   # [N, 256]
        self.mask_tokens = torch.cat([r["mask_token"] for r in records], dim=0)   # [N, 256]
        self.gt_ious     = torch.stack([r["gt_iou"]   for r in records], dim=0)   # [N]

    def __len__(self):
        return self.iou_tokens.size(0)

    def __getitem__(self, idx):
        return (
            self.iou_tokens[idx],
            self.mask_tokens[idx],
            self.gt_ious[idx],
        )


# ---------------------------------------------------------------------------
# Train / validate helpers
# ---------------------------------------------------------------------------

def run_epoch(loader, model, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    preds_all, targets_all = [], []

    for iou_tok, mask_tok, gt_iou in loader:
        iou_tok  = iou_tok.cuda()
        mask_tok = mask_tok.cuda()
        gt_iou   = gt_iou.cuda()

        pred = model(iou_tok, mask_tok).squeeze(-1)   # [B]
        loss = loss_fn(pred, gt_iou)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * iou_tok.size(0)
        preds_all.append(pred.detach().cpu())
        targets_all.append(gt_iou.cpu())

    n = len(loader.dataset)
    preds   = torch.cat(preds_all).numpy()
    targets = torch.cat(targets_all).numpy()
    mae = float(np.mean(np.abs(preds - targets)))
    return total_loss / n, mae


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # Load and merge all token files
    all_records = []
    for path in args.token_files:
        records = torch.load(path, map_location='cpu')
        all_records.extend(records)
        print(f"  loaded {len(records)} records from {path}")

    print(f"Total records: {len(all_records)}")
    if len(all_records) == 0:
        sys.exit("No records found — run collect_uq_tokens.py first.")

    dataset = TokenDataset(all_records)

    # 80/20 split
    n_val   = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=256,             shuffle=False, num_workers=4)

    model = UQHead().cuda()
    loss_fn = nn.MSELoss()
    # SGD with momentum, matching UncertainSAM training protocol
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.25,
        weight_decay=0.001,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    best_state    = None

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_mae = run_epoch(train_loader, model, loss_fn, optimizer)
        va_loss, va_mae = run_epoch(val_loader,   model, loss_fn)
        scheduler.step()

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:3d}/{args.epochs}  "
                  f"train loss={tr_loss:.5f} MAE={tr_mae:.4f}  "
                  f"val loss={va_loss:.5f} MAE={va_mae:.4f}")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    os.makedirs(dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(best_state, args.output)
    print(f"\nBest val loss {best_val_loss:.5f} — saved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train UQ head MLP')
    parser.add_argument('--token_files', nargs='+', required=True,
                        help='.pt files produced by collect_uq_tokens.py')
    parser.add_argument('--output', type=str, default='models/uq_heads/uq_head.pth',
                        help='Path to save the trained UQ head weights')
    parser.add_argument('--epochs',     type=int,   default=60)
    parser.add_argument('--lr',         type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int,   default=128)
    args = parser.parse_args()
    main(args)
