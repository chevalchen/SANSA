# UncSANSA Integration Plan
## Porting UncertainSAM UQ into SANSA

All edits happen exclusively inside `UncSANSA/UncSANSA/`. `SANSA/` and `UncertainSAM/` are read-only.

---

## Evaluation of GEMINI_DESIGN.md

### What the Gemini design got right
- Token source: concatenate `iou_token_out` (256-D) + `mask_token` (256-D) → 512-D MLP input ✓
- MLP architecture: 3 layers, hidden 512, Sigmoid output, MSE training ✓
- Post-hoc training strategy (freeze SANSA, collect tokens offline, train MLP) ✓
- Multi-shot confidence-weighted fusion concept ✓
- Minimal-intrusion philosophy ✓

### Critical errors that would cause implementation failure

**Error 1 — Wrong interception point (fatal)**
> "Intercept `mask_tokens` and `iou_predictions` at the final stage of SANSA's forward pass (typically after the decoder call in `engine.py` or `inference_fss.py`)."

`engine.py` and `inference_fss.py` only ever see `outputs["pred_masks"]` — the final interpolated mask tensor.
`iou_token_out` and `mask_tokens_out` are local variables inside `mask_decoder.predict_masks()` in
`models/sam2/modeling/sam/mask_decoder.py:214-215`. They are consumed entirely within that function and are
not returned to `_forward_sam_heads`, `DecoderOutput`, or anything upstream in `sansa.py`.
**Fix:** Add two side-channel attributes to the mask decoder's `predict_masks()` (exactly as UncertainSAM
does in `usam/patch_sam2.py:79-80`), then read them from `sansa.py` after each decoder call.

**Error 2 — Wrong description of SANSA's K-shot fusion (design mistake)**
> "Replace SANSA's default average-pooling feature fusion strategy."

SANSA does not average-pool support features. It uses a temporal memory bank with attention
(`_prepare_memory_conditioned_features` in `sam2_base.py`). The K independent 1-shot approach is
conceptually valid but is motivated differently: each support is scored independently so that
low-quality supports can be down-weighted when constructing the memory bank, not to replace
a pooling step that doesn't exist.

### Minor gaps that need filling in

**Gap 3 — Which mask token**
`mask_tokens_out` has shape `[B, num_mask_tokens=4, 256]`. The UQ head should use
`mask_tokens_out[:, 0, :]` — the single-mask output token, consistent with SANSA's default
single-mask mode. This matches UncertainSAM's usage and `sam_tokens_out` extraction in the decoder.

**Gap 4 — Checkpointing silently drops UQ weights**
`util/commons.py:adapter_state_dict()` keeps only keys containing `'adapter'`. Any UQ head
named `uq_head.*` will be silently discarded. The save filter must be updated.

**Gap 5 — Mask-prompted support frames bypass the decoder**
When `prompt_type == 'mask'`, `sansa.py` calls `_use_mask_as_output()` which never touches the
mask decoder transformer. No `iou_token_out` / `mask_tokens_out` are produced for those frames.
UQ scoring applies only to: (a) non-mask-prompted support frames and (b) all query frames.
In the standard training config (`--prompt mask`), **support frames produce no UQ tokens** —
UQ is purely a query-frame signal, which is actually the right target anyway.

---

## Final Implementation Plan

### File map

| Action | Path (relative to `UncSANSA/UncSANSA/`) |
|--------|------------------------------------------|
| Modify (2 lines) | `models/sam2/modeling/sam/mask_decoder.py` |
| Modify (extend filter) | `util/commons.py` |
| Modify (wire UQ head) | `models/sansa/sansa.py` |
| Create | `models/sansa/uq_head.py` |
| Create | `tools/collect_uq_tokens.py` |
| Create | `tools/train_uq_head.py` |

---

### Step 1 — Token side-channel in the mask decoder

**File:** `models/sam2/modeling/sam/mask_decoder.py`

Inside `predict_masks()`, immediately after line 215 where `mask_tokens_out` is computed, add:

```python
# UQ side-channel — readable by sansa.py after each decoder call
self._uq_iou_token  = iou_token_out.detach()          # [B, 256]
self._uq_mask_token = mask_tokens_out[:, 0, :].detach()  # [B, 256]  (single-mask token)
```

No other changes to the decoder. The attributes are overwritten on every forward call, so
they always reflect the most recent decode. Initialise both to `None` in `__init__` for safety:

```python
# in MaskDecoder.__init__, at end:
self._uq_iou_token = None
self._uq_mask_token = None
```

**Why not modify `forward()` return signature:** Changing `forward()` would require updating
`_forward_sam_heads` in `sam2_base.py`, `DecoderOutput` in `model_utils.py`, and every call
site. The side-channel attribute is a surgical two-line change that is invisible to all other code.

---

### Step 2 — UQ head module

**New file:** `models/sansa/uq_head.py`

```python
import torch
import torch.nn as nn


class UQHead(nn.Module):
    """
    Lightweight MLP that estimates per-query expected IoU from mask-decoder tokens.
    Input: cat(iou_token_out, mask_token_out) → 512-D
    Output: scalar in [0, 1]  (1 = high confidence, 0 = high uncertainty)
    Architecture matches UncertainSAM (ICML 2025) MLP design.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 512):
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
            iou_token:  [B, 256]
            mask_token: [B, 256]
        Returns:
            score: [B, 1]  expected IoU estimate
        """
        x = torch.cat([iou_token, mask_token], dim=-1)  # [B, 512]
        return self.layers(x)
```

---

### Step 3 — Wire UQ head into SANSA

**File:** `models/sansa/sansa.py`

#### 3a. `__init__`
```python
def __init__(self, sam: SAM2Base, device: torch.device, uq_head=None):
    super().__init__()
    self.sam = sam
    self.device = device
    self.uq_head = uq_head  # None during normal SANSA training; loaded for UQ inference
```

#### 3b. `forward` — capture UQ scores on query frames
After the `else` branch (`_compute_decoder_out_w_mem`), add score collection:

```python
else:
    decoder_out = self._compute_decoder_out_w_mem(backbone_output, absolute_idx, idx, self.memory_bank)
    if self.uq_head is not None:
        dec = self.sam.sam_mask_decoder
        if dec._uq_iou_token is not None:
            score = self.uq_head(
                dec._uq_iou_token.to(self.device),
                dec._uq_mask_token.to(self.device),
            )                              # [1, 1]
            outputs.setdefault("uq_scores", []).append(score)
```

After the mask interpolation at the end of `forward`, stack and return scores if present:

```python
if "uq_scores" in outputs:
    outputs["uq_scores"] = torch.cat(outputs["uq_scores"], dim=0)  # [B*J, 1]
```

UQ scores are only computed when `self.uq_head is not None`, so existing training and evaluation
pipelines are completely unaffected.

#### 3c. `build_sansa` — accept optional UQ checkpoint
```python
def build_sansa(sam2_version, adaptformer_stages, channel_factor, device,
                uq_head_path=None) -> SANSA:
    ...
    model = SANSA(sam=sam, device=torch.device(device))
    ...  # existing freeze logic unchanged

    if uq_head_path is not None:
        from models.sansa.uq_head import UQHead
        uq = UQHead().to(torch.device(device))
        uq.load_state_dict(torch.load(uq_head_path, map_location='cpu'))
        uq.eval()
        for p in uq.parameters():
            p.requires_grad_(False)
        model.uq_head = uq

    return model
```

---

### Step 4 — Fix checkpointing

**File:** `util/commons.py`

`adapter_state_dict` currently keeps only keys with `'adapter'`. Extend it:

```python
def adapter_state_dict(model) -> dict:
    sd = model.state_dict()
    adapter_sd = {k: v.cpu() for k, v in sd.items()
                  if 'adapter' in k or 'uq_head' in k}
    if not adapter_sd:
        print("[warn] no adapter keys found when saving!")
    return adapter_sd
```

This is backward-compatible: old checkpoints have no `uq_head.*` keys, and `resume_from_checkpoint`
already uses `strict=False`, so missing keys are silently tolerated.

---

### Step 5 — Token collection script (offline, Phase 1 prerequisite)

**New file:** `tools/collect_uq_tokens.py`

Purpose: run a trained SANSA adapter over a training split, intercept tokens from every
query-frame decode, record `(iou_token, mask_token, gt_iou)` triplets, save to disk.

Outline:
```
python tools/collect_uq_tokens.py \
  --resume pretrain/adapter_coco_fold0.pth \
  --dataset_file coco --fold 0 \
  --adaptformer_stages 2 3 \
  --prompt mask \
  --output_tokens data/uq_tokens/coco_fold0.pt
```

Key logic:
```python
model = build_sansa(...)  # uq_head=None; tokens via side-channel
model.eval()

records = []
for batch in dataloader:
    with torch.no_grad():
        outputs = model(imgs, prompt_dict)
    
    dec = model.sam.sam_mask_decoder
    if dec._uq_iou_token is None:
        continue

    # GT IoU: predicted query mask vs ground-truth query mask
    pred = (outputs["pred_masks"][-1].sigmoid() > 0.5).float()
    gt   = query_mask[0].to(pred.device).float()
    inter = (pred * gt).sum()
    union = ((pred + gt) > 0).float().sum()
    gt_iou = (inter / union.clamp(min=1e-6)).item()

    records.append({
        "iou_token":  dec._uq_iou_token.cpu(),   # [1, 256]
        "mask_token": dec._uq_mask_token.cpu(),  # [1, 256]
        "gt_iou":     gt_iou,
    })

torch.save(records, args.output_tokens)
```

One `.pt` file per dataset/fold. Collect from all training folds combined for maximum coverage.

---

### Step 6 — UQ head training script

**New file:** `tools/train_uq_head.py`

Purpose: load collected token files, train `UQHead` with MSE loss to predict `gt_iou`.

Outline:
```
python tools/train_uq_head.py \
  --token_files data/uq_tokens/coco_fold0.pt data/uq_tokens/coco_fold1.pt ... \
  --output models/uq_heads/coco.pth \
  --epochs 60 --lr 5e-5 --batch_size 128
```

Key design choices (following UncertainSAM):
- Optimizer: SGD with momentum 0.25, weight decay 0.001
- Loss: `nn.MSELoss()`
- Train/val split: 80/20 random split of collected records
- Epochs: 40–80 (very fast; dataset ~tens of thousands of samples)
- No data augmentation needed (tokens are already a distilled representation)

---

### Phase 2 — Multi-shot confidence weighting

This phase modifies `SANSA.forward()` for K-shot inference (`n_shots > 1`). It is independent
of Phase 1 and can be implemented after the UQ head is trained and validated.

**Mechanism:**
For K support images, instead of accumulating all K into the memory bank at once,
run K independent 1-shot mini-episodes to score each support, then build a confidence-weighted
memory bank before the final query decode.

**Modified `SANSA.forward()` for K-shot:**
```python
if n_shots > 1 and self.uq_head is not None:
    # --- Score each support independently ---
    support_scores = []
    support_mems   = []
    for k in range(n_shots):
        mini_bank = {}
        # 1-shot support k
        support_decoder_out = ... # process support k only (existing logic)
        mem_k = self._compute_memory_bank_dict(support_decoder_out, backbone_output, b*T+k)
        mini_bank[0] = mem_k
        # decode query frame with this single support
        q_abs = b * T + n_shots   # first query frame index
        qout  = self._compute_decoder_out_w_mem(backbone_output, q_abs, n_shots, mini_bank)
        dec   = self.sam.sam_mask_decoder
        score_k = self.uq_head(
            dec._uq_iou_token.to(self.device),
            dec._uq_mask_token.to(self.device),
        )                       # [1, 1]
        support_scores.append(score_k)
        support_mems.append(mem_k)

    # Softmax-weighted memory fusion
    weights = torch.softmax(torch.stack(support_scores, dim=0), dim=0)  # [K, 1, 1]
    fused_mem = {
        key: sum(w * m[key] for w, m in zip(weights, support_mems))
        for key in support_mems[0]
    }
    self.memory_bank = {k: support_mems[k] for k in range(n_shots)}
    # replace slot 0..n_shots-1 weights using the fused entry for the query
    # or simply: override memory_bank with weighted entries (implementation detail)
    ...
```

**Cost:** K extra query-frame decoder runs per episode (K=5 → 5 extra decoder calls per batch).
Encoder runs once (backbone is shared). Overhead is small vs the full backbone forward.

**Fallback:** If `self.uq_head is None` or `n_shots == 1`, the original sequential memory-bank
logic runs unchanged (zero regression vs baseline).

---

## Summary of all changes

### Modified files (minimal diffs)

| File | Change |
|------|--------|
| `models/sam2/modeling/sam/mask_decoder.py` | +4 lines: `__init__` init + `predict_masks` store side-channel |
| `util/commons.py` | +1 line: `adapter_state_dict` filter adds `'uq_head'` |
| `models/sansa/sansa.py` | `__init__` + `forward` + `build_sansa`: optional UQ head integration |

### New files

| File | Purpose |
|------|---------|
| `models/sansa/uq_head.py` | `UQHead` module (3-layer MLP, 512→512→1) |
| `tools/collect_uq_tokens.py` | Offline token collection over training set |
| `tools/train_uq_head.py` | Train UQ head from collected tokens |

### Files that do NOT need changes
- `engine.py` — training loop unchanged
- `inference_fss.py` — eval loop unchanged (UQ scores are extra keys in output dict, ignored if uq_head is absent)
- `main.py` — training entrypoint unchanged
- `opts.py` — no new required flags (Phase 2 can add `--uq_head_path`)
- Any file under `models/sam2/` except `mask_decoder.py`

---

## Execution order

```
Phase 1:
  Step 1  Modify mask_decoder.py       (2 min)
  Step 2  Create uq_head.py            (5 min)
  Step 3  Modify sansa.py              (15 min)
  Step 4  Modify commons.py            (1 min)
  Step 5  Create collect_uq_tokens.py  (30 min coding + hours of inference runtime)
  Step 6  Create train_uq_head.py      (20 min coding + minutes of training runtime)

Phase 2:
  Modify sansa.py forward for K-shot weighting
  Add --uq_head_path flag to opts.py
  Validate on COCO/LVIS multi-shot benchmarks
```
