"""
Offline token collection for UQ head training.

Runs a trained SANSA adapter over the training split of a dataset and records
(iou_token, mask_token, gt_iou) triplets from every query-frame decode.
Tokens are read from the mask decoder side-channel attributes set by predict_masks().

Usage (run from UncSANSA/UncSANSA/):

  # COCO fold 0, subsample 5000 episodes (~40 min)
  python tools/collect_uq_tokens.py \
    --resume pretrain/adapter_coco_fold0.pth \
    --dataset_file coco --fold 0 \
    --adaptformer_stages 2 3 --channel_factor 0.3 --prompt mask \
    --max_samples 5000 \
    --output data/uq_tokens/coco_fold0.pt

  # FSS-1000 (~1 h, no fold needed)
  python tools/collect_uq_tokens.py \
    --resume pretrain/adapter_fss.pth \
    --dataset_file fss \
    --adaptformer_stages 2 3 --channel_factor 0.3 --prompt mask \
    --output data/uq_tokens/fss.pt

  # Generalist adapter
  python tools/collect_uq_tokens.py \
    --resume pretrain/adapter_generalist.pth \
    --dataset_file coco --fold -1 --multi_train \
    --adaptformer_stages 2 3 --channel_factor 0.8 --prompt mask \
    --max_samples 5000 \
    --output data/uq_tokens/generalist.pt

Notes:
  - --channel_factor must match the adapter being loaded.
  - --max_samples caps collection after N *recorded* samples (skipped frames
    not counted), keeping the collection time manageable. 3000-8000 is enough
    for the 3-layer MLP to converge.
  - For mask-prompted support frames SANSA calls _use_mask_as_output()
    (bypasses the decoder), so no side-channel tokens are produced for those
    frames; only query-frame tokens are recorded.
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
    """Binary IoU between predicted logit mask and ground-truth binary mask."""
    pred  = (pred_mask.sigmoid() > 0.5).float()
    gt    = gt_mask.float()
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

    ds = build_dataset(args.dataset_file, image_set='train', args=args)
    dataloader = DataLoader(
        ds, batch_size=1, shuffle=True,   # shuffle for representative subsample
        num_workers=args.num_workers,
    )

    max_samples = args.max_samples  # None = collect everything
    records = []
    dec = model.sam.sam_mask_decoder

    total_hint = f"/{max_samples}" if max_samples else ""
    pbar = tqdm(dataloader, desc="Collecting tokens", ncols=90, file=sys.stderr)

    for batch in pbar:
        if max_samples and len(records) >= max_samples:
            break

        query_img     = batch['query_img']    # [1, C, H, W]
        query_mask    = batch['query_mask']   # [1, H, W]
        support_imgs  = batch['support_imgs'] # [1, S, C, H, W]
        support_masks = batch['support_masks']# [1, S, H, W]

        imgs = torch.cat([support_imgs[0], query_img], dim=0).unsqueeze(0)
        imgs = imgs.to(device)

        prompt_dict = build_prompt_dict(
            support_masks, args.prompt, n_shots=args.shots,
            train_mode=False, device=model.device,
        )

        with torch.no_grad():
            outputs = model(imgs, prompt_dict)

        # mask-prompted support frames bypass the decoder → no tokens
        if dec._uq_iou_token is None:
            continue

        gt_iou = compute_iou(
            outputs["pred_masks"][-1],   # query frame logit [H', W']
            query_mask[0].to(device),    # GT [H, W]
        )

        records.append({
            "iou_token":  dec._uq_iou_token.cpu().clone(),    # [1, 256]
            "mask_token": dec._uq_mask_token.cpu().clone(),   # [1, 256]
            "gt_iou":     torch.tensor(gt_iou, dtype=torch.float32),
        })

        pbar.set_postfix({"collected": f"{len(records)}{total_hint}",
                          "iou": f"{gt_iou:.3f}"})

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
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
    parser.add_argument(
        '--max_samples', type=int, default=None,
        help='Stop after collecting this many records. Recommended: 5000 for COCO, '
             'None (all) for FSS-1000. Default: collect everything.',
    )
    args = parser.parse_args()
    args.shots = 1  # always 1-shot collection
    main(args)
