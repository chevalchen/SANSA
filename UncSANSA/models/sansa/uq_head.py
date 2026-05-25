import torch
import torch.nn as nn


class UQHead(nn.Module):
    """
    Lightweight MLP estimating per-query expected IoU from mask-decoder tokens.

    Inputs:  iou_token  [B, 256]  +  mask_token [B, 256]  →  concat [B, 512]
    Output:  confidence score [B, 1] in (0, 1)
             High score = high confidence (low uncertainty).

    Architecture matches UncertainSAM (ICML 2025): 3 layers, hidden 512, Sigmoid.
    Trained post-hoc with MSE loss against ground-truth IoU values collected from
    a frozen SANSA adapter (see tools/collect_uq_tokens.py / train_uq_head.py).
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 512) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, iou_token: torch.Tensor, mask_token: torch.Tensor) -> torch.Tensor:
        """
        Args:
            iou_token:  [B, 256] — iou_token_out from mask decoder transformer
            mask_token: [B, 256] — mask_tokens_out[:, 0, :] (single-mask token)
        Returns:
            score: [B, 1]
        """
        x = torch.cat([iou_token, mask_token], dim=-1)  # [B, 512]
        return self.layers(x)
