# Modal Cloud Training — Splatfacto-DA2

This guide covers running `splatfacto-da2` training and evaluation on Modal cloud GPUs using `modal_train_splatfacto.py`.

## One-Time Setup

```bash
pip install modal
modal setup          # opens browser for authentication
modal volume create kitti-nerf-data
modal volume create nerf-outputs
```

## Dataset Upload

Upload once per dataset. Only needs to be repeated if the source data changes.

```bash
# Sparse KITTI sequence folder (contains images + depths_da2/)
modal volume put kitti-nerf-data \
  /path/to/kitti_select_static_5seq_sparse_every2 \
  kitti/kitti_select_static_5seq_sparse_every2

# Nerfstudio-formatted dataset (transforms.json + images)
modal volume put kitti-nerf-data \
  /path/to/nerfstudio/kitti_seq02_0034_sparse_every2 \
  nerfstudio/kitti_seq02_0034_sparse_every2
```

The training script will automatically build the depth-augmented dataset
(`kitti_seq02_0034_sparse_every2_da2`) inside the container on first run.

### Modal Volume Layout

```text
kitti-nerf-data/
  kitti/kitti_select_static_5seq_sparse_every2/
    KITTISeq02_...densegt/
      images/
      depths_da2/       ← DA2 depth PNGs (uint16, depth_m = value/256)
  nerfstudio/
    kitti_seq02_0034_sparse_every2/
      transforms.json
      images/

nerf-outputs/
  <exp_name>/
    splatfacto-da2/
      <timestamp>/
        config.yml
        nerfstudio_models/step-000049999.ckpt
        events.out.tfevents.*   ← tensorboard logs
        eval_output.json        ← written by run_eval
```

## Training

### Single Run (default: seq02, lambda=0.05, 50k steps)

```bash
modal run modal_train_splatfacto.py::main
```

### Detached Mode (recommended — terminal can be closed)

```bash
modal run --detach modal_train_splatfacto.py::main
```

Monitor at `https://modal.com/apps/waynechu/main`.

### Override Hyperparameters

```bash
modal run --detach modal_train_splatfacto.py::main \
  --lambda-depth 0.1 \
  --depth-loss-type l1 \
  --max-num-iterations 30000
```

### Lambda Sweep (parallel containers)

Runs all lambda values simultaneously, each on a separate A10G.

```bash
# Default: lambda in {0.0, 0.05, 0.1, 0.2}, loss=mse
modal run --detach modal_train_splatfacto.py::sweep

# Custom values
modal run --detach modal_train_splatfacto.py::sweep \
  --lambdas "0.0 0.05 0.1 0.2" \
  --loss-types "mse l1"
```

### Photometric Mask Threshold Sweep (two-stage)

Stage 1: train a base model without masks.

```bash
modal run --detach modal_train_splatfacto.py::main --lambda-depth 0.05
```

Stage 2: generate masks at multiple thresholds, then retrain — all in parallel on Modal.

```bash
modal run --detach modal_train_splatfacto.py::sweep_threshold \
  --base-exp-name "kitti_seq02_0034_sparse_every2_da2_lambda0.05_nomask_50000" \
  --thresholds "0.08 0.12 0.16 0.22" \
  --photo-mask-mode low
```

## Experiment Naming Convention

Experiment names are derived automatically:

```
{seq_slug}_sparse_every2_{depth_sup_type}_lambda{lambda}_{mask_label}_{iters}
```

Examples:
- `kitti_seq02_0034_sparse_every2_da2_lambda0.05_nomask_50000`
- `kitti_seq02_0034_sparse_every2_da2_lambda0.1_low012_50000`

## Evaluation

`run_eval` accepts the same parameters as `main` and derives the experiment name automatically.

```bash
# Eval the default run (lambda=0.05, nomask, 50k)
modal run modal_train_splatfacto.py::run_eval

# Eval a specific config
modal run modal_train_splatfacto.py::run_eval --lambda-depth 0.1
modal run modal_train_splatfacto.py::run_eval --lambda-depth 0.0
```

Results are printed to terminal and saved as `eval_output.json` in the experiment folder.

### Output Metrics

| Metric | Description | Better |
|--------|-------------|--------|
| `psnr` | Peak Signal-to-Noise Ratio (dB) | Higher |
| `ssim` | Structural Similarity (0–1) | Higher |
| `lpips` | Perceptual similarity (0–1) | Lower |
| `depth_mse` | Depth mean squared error | Lower |
| `depth_mae` | Depth mean absolute error | Lower |

All values are **averages over the validation set** (9 images: indices 4, 14, 24, ..., 84, every 10th frame). There is no separate test set in this nerfstudio setup.

## Downloading Outputs

```bash
# Download a specific experiment
modal volume get nerf-outputs <exp_name> ./local_outputs

# Example
modal volume get nerf-outputs \
  kitti_seq02_0034_sparse_every2_da2_lambda0.05_nomask_50000 \
  ./local_outputs
```

### View Tensorboard

```bash
tensorboard --logdir "./local_outputs/<exp_name>/splatfacto-da2/<timestamp>"
```

The `<timestamp>` folder is named after training start time (e.g. `2026-06-03_173124`).
Use the **latest** timestamp if multiple exist.

## Known Issues

### Tensorboard Logs Truncated (~3700 steps)

**Symptom:** tensorboard scalars stop at ~3700 steps even though training ran to 50k.

**Cause:** Modal's FUSE volume filesystem buffers writes. The tensorboard `SummaryWriter`
flushes periodically; only the data written to disk before the final `out_vol.commit()`
call is captured. The 1 GB event file size is mostly image data logged during early evals.

**Workaround:** Use `run_eval` to get accurate final metrics from the saved checkpoint.
The `step-000049999.ckpt` checkpoint is always correctly committed.

### Pillow Compatibility (pil_to_numpy)

The nerfstudio `pil_to_numpy` function used the internal `_getencoder` / `setimage` API
which changed in Pillow 12.x. This is fixed in the local nerfstudio source
(`nerfstudio/data/utils/data_utils.py`) to use `np.array(im)` instead.

### Image Build Time

The first build of the Modal image takes ~10–15 minutes due to CUDA compilation of
`tiny-cuda-nn` and `gsplat`. Subsequent builds are cached per layer:

- `_base` (PyTorch): rebuilds only if CUDA/PyTorch version changes
- `_with_tcnn`: rebuilds only if the tcnn commit hash changes
- `_with_gsplat`: rebuilds only if the gsplat version changes
- `image` (nerfstudio): rebuilds whenever `nerfstudio/` source files change

### `add_local_dir` with `run_commands`

Modal requires `copy=True` on any `add_local_dir` call that is followed by
`run_commands`. Without it, Modal raises `InvalidError` because local files are injected
at container startup rather than at build time.

## Cost Estimates (Modal A10G, $1.10/hr)

| Job | Duration | Approx. Cost |
|-----|----------|-------------|
| Single 50k run | ~20 min | ~$0.37 |
| Lambda sweep (4 runs parallel) | ~20 min | ~$1.47 |
| Threshold sweep (4 masks + 4 retrains) | ~50 min | ~$1.83 |
| `run_eval` | ~3 min | ~$0.06 |

Costs are estimates. Check `https://modal.com/usage` for actual billing.
