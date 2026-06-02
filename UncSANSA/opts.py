import argparse

def get_args_parser() -> argparse.ArgumentParser:
    """Argument parser for SANSA training and inference (parent parser)."""
    parser = argparse.ArgumentParser("SANSA training and inference", add_help=False)

    # General
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device.")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from.")

    # Experiment I/O
    parser.add_argument("--output_dir", type=str, default="output", help="Root directory for outputs.")
    parser.add_argument("--name_exp", type=str, default="prova", help="Experiment name (subfolder in output_dir).")

    # Data
    parser.add_argument("--data_root", type=str, default="../../datasets", help="Root directory for datasets.")
    parser.add_argument("--dataset_file", type=str, default="coco", choices=["coco", "lvis", "fss", "pascal_voc", "pascal_voc_cd", "pascal_part", "paco_part", "deepglobe", "isic",
                 "lung", "ade20k", "multi"], help="Dataset name. Use 'multi' for training the generalist model.")
    parser.add_argument("--multi_train", nargs="+", type=str, default=["lvis", "coco", "ade20k", "paco_part"], help="Datasets to mix when dataset_file='multi'.")
    parser.add_argument("--ds_weight", nargs="+", type=float, default=[0.4, 0.45, 0.1, 0.05], help="Sampling weights for datasets in --multi_train.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")

    # Prompting / Shots / Folds
    parser.add_argument("--prompt", type=str, default="mask", choices=["mask", "scribble", "box", "point", "multi"], help="Prompt type for support frames; 'multi' samples a type at random.")
    parser.add_argument("--shots", type=int, default=1, help="Number of support frames per episode.")
    parser.add_argument("--J", type=int, default=3, help="Number of unlabeled target images per training episode.")
    parser.add_argument("--fold", type=int, default=0, help="Training fold (if applicable).")

    # SAM2 / Backbone
    parser.add_argument("--sam2_version", type=str, default="large", choices=["tiny", "base", "large"], help="Version of SAM2 image encoder.")
    parser.add_argument("--adaptformer_stages", nargs="+", type=int, default=[-1], help="Adapter/AdaptFormer stages to enable (model-specific).")
    parser.add_argument("--channel_factor", type=float, default=0.3, help="Adapter channel scaling factor (model-specific).")

    # Optimization
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument('--start_epoch', default=0, type=int, help="Epoch to start training from (for resuming)")
    parser.add_argument("--clip_max_norm", type=float, default=0.1, help="Gradient clipping max norm (0 disables clipping).")
    parser.add_argument("--batch_size", type=int, default=2, help="Global batch size (may be split per GPU).")

    # Logging / Runtime
    parser.add_argument("--no_distributed", action="store_true", default=False, help="Force single-process training.")

    # Inference
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold to binarize masks at eval.")
    parser.add_argument("--visualize", action="store_true", default=False, help="Save qualitative results.")
    parser.add_argument("--hflip_tta", action="store_true", default=False,
                        help="Enable horizontal-flip test-time augmentation for query frames: "
                             "re-encode the flipped image and ensemble masks in logit space. "
                             "Adds one backbone forward per query frame. (+0.43 mIoU on PACO-Part.)")
    parser.add_argument("--mem_feedback", action="store_true", default=False,
                        help="Memory closed-loop feedback: second decode using first prediction "
                             "as soft mask_inputs prompt; fuse two logits with equal weight. "
                             "No extra backbone cost (reuses memory-conditioned features).")
    parser.add_argument("--boundary_refine", action="store_true", default=False,
                        help="Boundary Refinement Module (BRM): lightweight residual correction "
                             "at uncertain boundary pixels (sigmoid in (0.4,0.6)). "
                             "Requires stage-2 training with --freeze_adapter.")
    parser.add_argument("--freeze_adapter", action="store_true", default=False,
                        help="Freeze adapter params during training; only BRM params are updated. "
                             "Use with --boundary_refine for 2-stage training.")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max training iterations per epoch (default: full dataset). "
                             "Useful for fast BRM training: --max_steps 1000 gives ~1h/epoch "
                             "with frozen backbone on SAM2-large.")
    parser.add_argument("--uq_head_path", type=str, default=None,
                        help="Path to a trained UQ head (.pth). When provided, inference_fss.py "
                             "records per-episode confidence scores (observe-only; no decoder changes).")
    parser.add_argument("--save_uq_log", type=str, default=None,
                        help="Path to save per-episode (uq_score, actual_iou, class_id) log (.pt). "
                             "Requires --uq_head_path. Used by tools/eval_uq_quality.py.")

    return parser
