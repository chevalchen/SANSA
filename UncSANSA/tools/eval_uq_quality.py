"""
Evaluate the discriminative quality of the UQ head.

Loads a per-episode log produced by inference_fss.py --save_uq_log and computes:
  1. Pearson correlation  (uq_score vs actual IoU)
  2. Spearman correlation (rank-based)
  3. Relative AUC        (correction curve, matching UncertainSAM Fig. 5 / Table 4)
  4. Optional scatter plot saved as PNG

The "correction AUC" metric works as follows (UncertainSAM §4.5):
  - Sort episodes by uncertainty (= 1 - uq_score), most-uncertain first.
  - Progressively "correct" the most uncertain r% of episodes by replacing
    their IoU with 1.0 (simulating what would happen if a user fixed them).
  - Plot mIoU vs correction ratio r ∈ [0, R_max].
  - AUC of this curve ↑ = the UQ head correctly identifies bad predictions.
  - Relative AUC = (AUC_method - AUC_random) / (AUC_oracle - AUC_random) × 100
    where oracle = sort by actual IoU ascending (perfect ranking),
          random = no sorting (random correction order).

Usage (run from UncSANSA/UncSANSA/):
    python tools/eval_uq_quality.py --log output/eval_fss_uq/uq_log.pt

    # also save scatter plot
    python tools/eval_uq_quality.py --log output/eval_fss_uq/uq_log.pt --plot
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


# ---------------------------------------------------------------------------
# AUC helpers
# ---------------------------------------------------------------------------

def correction_curve(uq_scores: np.ndarray, iou: np.ndarray,
                     sort_idx: np.ndarray, r_max: float = 0.5,
                     n_steps: int = 100) -> np.ndarray:
    """
    Compute mIoU at each correction ratio when correcting in the order given
    by sort_idx (most-uncertain / worst-first order).

    Returns: miou_curve of shape [n_steps+1]
    """
    N = len(iou)
    corrected = iou.copy()
    ratios = np.linspace(0, r_max, n_steps + 1)
    curve  = np.zeros(n_steps + 1)

    for i, r in enumerate(ratios):
        n_correct = int(r * N)
        corrected_iou = iou.copy()
        corrected_iou[sort_idx[:n_correct]] = 1.0
        curve[i] = corrected_iou.mean()

    return curve


def compute_relative_auc(uq_scores: np.ndarray, iou: np.ndarray,
                          r_max: float = 0.5, n_steps: int = 100) -> dict:
    N = len(iou)
    # Most uncertain first = ascending uq_score
    uq_order     = np.argsort(uq_scores)               # our UQ method
    oracle_order = np.argsort(iou)                      # worst actual IoU first
    random_order = np.random.default_rng(0).permutation(N)

    curve_uq     = correction_curve(uq_scores, iou, uq_order,     r_max, n_steps)
    curve_oracle = correction_curve(uq_scores, iou, oracle_order, r_max, n_steps)
    curve_random = correction_curve(uq_scores, iou, random_order, r_max, n_steps)

    auc_uq     = np.trapezoid(curve_uq,     dx=r_max / n_steps)
    auc_oracle = np.trapezoid(curve_oracle, dx=r_max / n_steps)
    auc_random = np.trapezoid(curve_random, dx=r_max / n_steps)

    rel_auc = (auc_uq - auc_random) / max(auc_oracle - auc_random, 1e-8) * 100

    return {
        'auc_uq':     auc_uq,
        'auc_oracle': auc_oracle,
        'auc_random': auc_random,
        'rel_auc':    rel_auc,         # % (higher is better; oracle = 100%)
        'curve_uq':     curve_uq,
        'curve_oracle': curve_oracle,
        'curve_random': curve_random,
        'ratios': np.linspace(0, r_max, n_steps + 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    log = torch.load(args.log, map_location='cpu')
    uq_scores = log['uq_scores'].numpy()   # [N]
    iou       = log['iou'].numpy()         # [N]
    dataset   = log.get('dataset', '?')
    fold      = log.get('fold', '?')
    shot      = log.get('shot', '?')
    N         = len(uq_scores)

    print(f"\n{'='*55}")
    print(f"  UQ Quality Evaluation")
    print(f"  Dataset: {dataset}  fold: {fold}  shot: {shot}  N={N}")
    print(f"{'='*55}")

    # ---- 1. Correlation ----
    pearson_r,  pearson_p  = pearsonr(uq_scores, iou)
    spearman_r, spearman_p = spearmanr(uq_scores, iou)

    print(f"\n[Correlation: UQ confidence vs actual IoU]")
    print(f"  Pearson  r = {pearson_r:+.4f}   p = {pearson_p:.2e}")
    print(f"  Spearman r = {spearman_r:+.4f}   p = {spearman_p:.2e}")
    print(f"  (positive = higher confidence → higher IoU, as expected)")

    # ---- 2. Correction AUC ----
    auc_res = compute_relative_auc(uq_scores, iou, r_max=args.r_max)

    print(f"\n[Correction AUC  (r_max={args.r_max*100:.0f}% corrected)]")
    print(f"  UQ method  AUC = {auc_res['auc_uq']:.4f}")
    print(f"  Oracle     AUC = {auc_res['auc_oracle']:.4f}  (perfect ranking)")
    print(f"  Random     AUC = {auc_res['auc_random']:.4f}  (random baseline)")
    print(f"  Relative AUC   = {auc_res['rel_auc']:.1f}%  "
          f"(0% = random, 100% = oracle)")

    # ---- 3. Bucket analysis ----
    print(f"\n[Confidence bucket analysis]")
    thresholds = [0.4, 0.6, 0.7, 0.8, 0.9]
    prev = 0.0
    for t in thresholds + [1.01]:
        mask = (uq_scores >= prev) & (uq_scores < t)
        if mask.sum() > 0:
            print(f"  conf [{prev:.1f}, {t:.1f})  "
                  f"n={mask.sum():4d}  avg_iou={iou[mask].mean():.4f}")
        prev = t

    # ---- 4. Optional plot ----
    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f'UQ Quality — {dataset} fold {fold} {shot}-shot', fontsize=13)

            # Scatter: UQ confidence vs actual IoU
            ax1.scatter(uq_scores, iou, s=4, alpha=0.3, color='steelblue')
            ax1.set_xlabel('UQ confidence score')
            ax1.set_ylabel('Actual IoU')
            ax1.set_title(f'Scatter  (Pearson r={pearson_r:.3f})')
            ax1.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.4)

            # Correction curve
            ratios = auc_res['ratios'] * 100  # to %
            ax2.plot(ratios, auc_res['curve_oracle'] * 100, 'g--',  lw=1.5, label=f"Oracle  (AUC={auc_res['auc_oracle']:.3f})")
            ax2.plot(ratios, auc_res['curve_uq']     * 100, 'b-',   lw=2,   label=f"UQ head (AUC={auc_res['auc_uq']:.3f}, rel={auc_res['rel_auc']:.1f}%)")
            ax2.plot(ratios, auc_res['curve_random'] * 100, 'r:',   lw=1.5, label=f"Random  (AUC={auc_res['auc_random']:.3f})")
            ax2.set_xlabel('Ratio of corrected samples (%)')
            ax2.set_ylabel('mIoU (%)')
            ax2.set_title('Correction curve')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)

            out_png = args.log.replace('.pt', '_uq_quality.png')
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            print(f'\nPlot saved → {out_png}')
        except ImportError:
            print('\n[warn] matplotlib not installed, skipping plot.')

    print(f"\n{'='*55}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate UQ head discriminative quality')
    parser.add_argument('--log',   type=str, required=True,
                        help='.pt file produced by inference_fss.py --save_uq_log')
    parser.add_argument('--r_max', type=float, default=0.5,
                        help='Max correction ratio for AUC (default 0.5 = top-50%%)')
    parser.add_argument('--plot',  action='store_true',
                        help='Save scatter + correction curve PNG (requires matplotlib)')
    args = parser.parse_args()
    main(args)
