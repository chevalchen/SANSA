"""
Offline token collection for UQ head training.

Runs a trained SANSA adapter over the training split of a dataset and records
(iou_token, mask_token, gt_iou) triplets from every query-frame decode.
Tokens are read from the mask decoder side-channel attributes set by predict_masks().

Usage (run from UncSANSA/UncSANSA/):
    python tools/collect_uq_tokens.py \
        --resume pretrain/adapter_coco_fold0.pth \
        --dataset_file coco --fold 0 \
        --adaptformer_stages 2 3 \
        --prompt mask \
        --output data/uq_tokens/coco_fold0.pt

To collect across all COCO folds, run once per fold and concatenate with
--token_files in train_uq_head.py.
"""

import argparse
import sys
import os
from os.path import dirname
sys.path.insert(0, dirname(dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import opts
from models.sansa.sansa import build_sansa
from datasets import build_dataset
from util.commons import make_deterministic, resume_from_checkpoint
from util.promptable_utils import build_prompt_dict


def compute_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """Binary IoU between two boolean/float masks of the same spatial size."""
    pred = (pred_mask.sigmoid() > 0.5).float()
    gt   = gt_mask.float()
    inter = (pred * gt).sum().item()
    union = ((pred + gt) > 0).float().sum().item()
    return inter / max(union, 1e-6)


def main(args: argparse.Namespace) -> None:
    make_deterministic(args.seed)

    model = build_sansa(
        args.sam2_version,
        args.adaptformer_stages,
        args.channel_factor,
        args.device,
    )
    device = torch.device(args.device)
    model.to(device)

    if args.resume:
        resume_from_checkpoint(args.resume, model)
    model.eval()

    # Build training dataset (image_set='train')
    ds = build_dataset(args.dataset_file, image_set='train', args=args)
    dataloader = DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=args.num_workers
    )

    records = []
    dec = model.sam.sam_mask_decoder

    pbar = tqdm(dataloader, desc="Collecting tokens", ncols=80, file=sys.stderr)
    for batch in pbar:
        query_img   = batch['query_img']           # [1, C, H, W]
        query_mask  = batch['query_mask']          # [1, H, W]
        support_imgs  = batch['support_imgs']      # [1, S, C, H, W]
        support_masks = batch['support_masks']     # [1, S, H, W]

        imgs = torch.cat([support_imgs[0], query_img], dim=0).unsqueeze(0)  # [1, T, C, H, W]
        imgs = imgs.to(device)

        prompt_dict = build_prompt_dict(
            support_masks, args.prompt, n_shots=args.shots,
            train_mode=False, device=model.device,
        )

        with torch.no_grad():
            outputs = model(imgs, prompt_dict)

        # Side-channel is None when support used _use_mask_as_output (mask prompt)
        if dec._uq_iou_token is None:
            continue

        # Last frame = query; compute GT IoU
        gt_iou = compute_iou(
            outputs["pred_masks"][-1],          # logit, [H', W']
            query_mask[0].to(device),           # [H, W]
        )

        records.append({
            "iou_token":  dec._uq_iou_token.cpu().clone(),   # [1, 256]
            "mask_token": dec._uq_mask_token.cpu().clone(),  # [1, 256]
            "gt_iou":     torch.tensor(gt_iou, dtype=torch.float32),
        })

        if len(records) % 500 == 0:
            pbar.set_postfix({"collected": len(records)})

    os.makedirs(dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(records, args.output)
    print(f"Saved {len(records)} records → {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Collect UQ tokens from SANSA forward passes',
        parents=[opts.get_args_parser()],
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Path to save collected token records (.pt file)',
    )
    args = parser.parse_args()
    # shots=1 for token collection (match standard 1-shot eval setup)
    args.shots = 1
    main(args)
