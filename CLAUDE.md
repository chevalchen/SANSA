# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Layout & Edit Policy

This working directory holds **three sibling projects** that must be treated very differently:

```
UncSANSA/                  <- working root (D:\Desktop\ALL_Modify_SANSA\UncSANSA)
├── SANSA/                 <- READ-ONLY reference. Pristine upstream SANSA (NeurIPS '25).
├── UncertainSAM/          <- READ-ONLY reference. Source of the uncertainty-quantification module to port.
└── UncSANSA/              <- THE ONLY directory you may modify.
                              Started as a copy of SANSA and is being extended by porting the
                              UQ module from UncertainSAM into it.
```

**Rules — do not violate:**
- Never edit, create, or delete files inside `SANSA/` or `UncertainSAM/`. They exist only as references for diff/comparison and as the source for code to be ported.
- All modifications, new files, experiments, and outputs go inside `UncSANSA/`.
- When porting code from `UncertainSAM/usam/` into `UncSANSA/`, re-read the originals each time — do not assume they have been re-arranged.
- When unsure whether a behaviour came from upstream SANSA or from a local change, diff `UncSANSA/<file>` against `SANSA/<file>`.

## High-Level Architecture

### SANSA (baseline in `UncSANSA/UncSANSA/`)
A few-shot segmenter built on a **frozen SAM2** image encoder + mask decoder, with trainable **AdaptFormer-style adapters** injected into the last Hiera stages.

- `main.py` — distributed training entrypoint. Builds the model via `models.sansa.sansa.build_sansa`, sets up an `AdamW + CosineAnnealingLR` schedule, runs `engine.train_one_epoch`, then calls `inference_fss.eval_fss` after every epoch. Only adapter params are saved (`util.commons.adapter_state_dict`).
- `inference_fss.py` — evaluation entrypoint (also runnable standalone with `--resume`).
- `opts.py` — single shared argument parser used by both training and inference. Key flags: `--dataset_file`, `--fold`, `--shot`, `--prompt {mask|scribble|box|point|multi}`, `--adaptformer_stages`, `--channel_factor`, `--multi_train`, `--ds_weight`.
- `models/sam2/` — vendored SAM2 (modeling, configs, utils). Treat as a frozen backbone; weights are downloaded on first build via `util.path_utils.SAM2_WEIGHTS_URL`.
- `models/sansa/` — the SANSA wrapper.
  - `sansa.py` defines the `SANSA(nn.Module)` that wraps a `SAM2Base`. Its `forward` consumes `[B, T, C, H, W]` videos + a `prompt_dict`, runs the SAM2 backbone, then iterates over frames: support frames go through `_use_mask_as_output` or `_compute_decoder_out_no_mem`; query frames go through `_compute_decoder_out_w_mem` using an episode-local `memory_bank`.
  - `adapter.py` defines the AdaptFormer adapters injected into the SAM2 Hiera encoder.
- `datasets/` — one module per benchmark (`coco`, `lvis`, `fss`, `ade20k`, `paco_part`, `pascal_part`, `deepglobe`, `isic`, `lung`). `__init__.build_dataset` dispatches on `--dataset_file`; `samplers.py` provides the distributed sampler. For `--dataset_file multi`, the loader mixes datasets according to `--ds_weight`.
- `util/` — `commons` (logging / determinism / checkpoint resume / adapter-only state-dict extraction), `losses`, `metrics`, `misc` (distributed init), `promptable_utils` (prompt rescaling + `format_prompt` helper used by the TorchHub demo).
- `hubconf.py` — TorchHub entry exposing the "SANSA Universal Model" (`torch.hub.load('ClaudiaCuttano/SANSA', 'sansa', ...)`). Note the universal model is *not* the same as the strict-few-shot checkpoints.

### UncertainSAM (reference in `UncSANSA/UncertainSAM/`)
A post-hoc uncertainty-quantification patch over SAM2. The piece being ported is in `usam/`:

- `usam/patch_sam2.py` — monkey-patches a `SAM2ImagePredictor`/`SAM2Base` instance: replaces `predict`, `_predict`, `predict_masks`, and `forward` on the `sam_mask_decoder`. After the transformer runs inside `predict_masks`, it captures `iou_token_out` and `mask_tokens_out`, concatenates them, and feeds them through a dict of small `MLP` regressors loaded from `model_dir`. Outputs are stored in `decoder.scores` and returned alongside the masks (`predict` returns `(masks, iou, low_res, mlp_scores)`). It also supports injecting a `custom_token` to override the decoder's mask tokens — this is the integration hook.
- `usam/MLP.py` — the regressor architecture.
- `usam/training/` — standalone training pipeline for the MLP heads (`train.py`, `dataset.py`, `augmentations.py`, `metrics.py`, `scheduler.py`).
- `scripts/` — `demo.py` (live visualisation), `train_all_predictors.py`, `verify_sam2_installation.py`, `verify_usam_installation.py`.
- `models/sam/checkpoints_2.0` and `_2.1` — download scripts for SAM2 weights; `models/mlps/` is where trained UQ MLPs live.

UncertainSAM expects the *official* `sam2` package to be installed (it `import`s `from sam2.build_sam ...`). SANSA instead vendors SAM2 under `models/sam2/`. When porting into `UncSANSA/`, prefer the vendored copy and adapt the imports rather than adding a hard dependency on the external `sam2` package.

## Common Commands

All commands are run from inside `UncSANSA/UncSANSA/` (the editable copy), not from the repo root.

### Environment
```
conda create --name sansa python=3.10 -y
conda activate sansa
pip install -r requirements.txt
```

### Training (strict few-shot)
```
python main.py --batch_size 32 --name_exp train_coco_f0 \
  --dataset_file coco --fold 0 --adaptformer_stages 2 3 --prompt mask
```
- `--fold F` means *evaluate* on F, *train* on the other folds.
- Folds: COCO `0–3`, LVIS `0–9`, FSS-1000 omit `--fold`.
- `--prompt multi` randomises prompt type per episode (promptable variant).

### Training (generalist / multi-dataset)
```
python main.py --batch_size 32 --name_exp train_generalist --multi_train \
  --dataset_file lvis, coco, ade20k, paco_part \
  --ds_weight 0.4, 0.45, 0.1, 0.05 \
  --fold -1 --adaptformer_stages 2 3 --channel_factor 0.8 --prompt mask
```
`--fold -1` disables strict fold splitting.

### Inference / reproducing paper numbers
```
python inference_fss.py --dataset_file {coco|lvis|fss|pascal_part|paco_part} \
  --fold {FOLD} --resume /path/to/adapter.pth \
  --name_exp eval_xxx --shot {1|5} \
  --adaptformer_stages 2,3 --prompt mask
```
Add `--visualize` for qualitative outputs. For the generalist adapter, also pass `--channel_factor 0.8`.

Single-process run (no DDP): add `--no_distributed`.

### UncertainSAM-side reference commands (only run inside `UncertainSAM/`, never in `UncSANSA/`)
```
python scripts/verify_sam2_installation.py
python scripts/verify_usam_installation.py
python scripts/demo.py
```

## Porting Notes (UncertainSAM → UncSANSA)

When integrating UQ into SANSA:
- The UQ heads consume `iou_token_out` and `mask_tokens_out` *after* `self.transformer(src, pos_src, tokens)` runs in the SAM2 mask decoder. In SANSA, the mask decoder lives under `UncSANSA/UncSANSA/models/sam2/modeling/` — patch there (or wrap from `models/sansa/sansa.py`) rather than re-monkey-patching at runtime.
- SANSA always runs episodes through `models/sansa/sansa.py:SANSA.forward`, which calls into SAM2 multiple times per episode (one support frame, J query frames, with a memory bank). Decide whether UQ should fire on every call or only on query frames before wiring it in.
- Checkpointing in `main.py` saves only adapter params (`adapter_state_dict`). Any new trainable UQ heads must be added to that filter, or they will silently not be saved.
