import torch
import torch.nn as nn
from torch import Tensor


class BoundaryRefinementModule(nn.Module):
    """
    Lightweight residual correction for uncertain boundary pixels.

    Inputs:
      logits  [B, 1, H, W]          -- SAM2 low_res_masks (raw logits, not sigmoid)
      feat    [B, feat_channels, H, W] -- highest-resolution backbone feature (conv_s0, 32ch)

    Output:
      corrected logits [B, 1, H, W]

    Only pixels where sigmoid(logit) ∈ (0.4, 0.6) receive a non-zero residual; others
    pass through unchanged. This boundary gate prevents the module from disturbing
    confidently-predicted foreground / background regions.

    Parameters: ~16 K  (feat_channels=32, hidden=16)

    The final Conv2d output layer is zero-initialised so the module starts as an exact
    identity — loading an existing adapter checkpoint works without any quality loss.
    """

    def __init__(self, feat_channels: int = 32, hidden: int = 16) -> None:
        super().__init__()
        # Input: concat(logits [1], feat [feat_channels]) → [feat_channels+1, H, W]
        self.refine = nn.Sequential(
            nn.Conv2d(feat_channels + 1, hidden, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1, bias=True),
        )
        # Zero-init the output layer → identity at init (stable residual learning)
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, logits: Tensor, feat: Tensor) -> Tensor:
        """
        Args:
            logits: [B, 1, H, W]          low-res mask logits from SAM2 decoder
            feat:   [B, feat_channels, H, W]  conv_s0 backbone feature (same spatial size)
        Returns:
            corrected logits [B, 1, H, W]
        """
        # Boundary zone mask — no gradient; acts as spatial gate on the residual
        with torch.no_grad():
            prob = torch.sigmoid(logits)
            b_mask = ((prob > 0.4) & (prob < 0.6)).float()   # [B, 1, H, W]

        x = torch.cat([logits, feat], dim=1)   # [B, feat_channels+1, H, W]
        offset = self.refine(x)                # [B, 1, H, W]
        return logits + b_mask * offset
