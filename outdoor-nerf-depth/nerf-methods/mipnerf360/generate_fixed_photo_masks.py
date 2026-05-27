import functools
import os
from os import path

from absl import app
from flax.training import checkpoints
import jax
from jax import random
import numpy as np
from PIL import Image

from internal import configs
from internal import datasets
from internal import models
from internal import train_utils
from internal import utils

configs.define_common_flags()
jax.config.parse_flags_with_absl()


def main(unused_argv):
  config = configs.load_config(save_config=False)

  if not config.fixed_photo_mask_dir:
    raise ValueError("Config.fixed_photo_mask_dir must be set.")

  out_dir = config.fixed_photo_mask_dir
  if not path.isabs(out_dir):
    out_dir = path.join(config.data_dir, out_dir)
  utils.makedirs(out_dir)

  config.fixed_photo_mask_dir = ''

  dataset = datasets.load_dataset('train', config.data_dir, config)

  key = random.PRNGKey(20200823)
  _, state, render_eval_pfn, _, _ = train_utils.setup_model(config, key)
  state = checkpoints.restore_checkpoint(
      config.checkpoint_dir, state, step=config.restore_step)
  step = int(state.step)
  print(f"Restored checkpoint step {step}.")
  print(f"Train images: {dataset.size}")
  print(f"Writing fixed masks to: {out_dir}")
  print(f"Mask mode: {config.photo_mask_mode}")
  print(f"Threshold: {config.photo_mask_threshold}")

  keep_ratios = []

  for idx in range(dataset.size):
    batch = dataset.generate_ray_batch(idx)
    train_frac = step / config.max_steps

    rendering = models.render_image(
        functools.partial(
            render_eval_pfn,
            state.params,
            train_frac,
        ),
        batch.rays,
        None,
        config,
    )

    rgb = np.array(rendering['rgb'])
    rgb_gt = np.array(batch.rgb)
    photo_error = np.abs(rgb - rgb_gt).mean(axis=-1)

    if config.photo_mask_mode == 'low':
      mask = photo_error < config.photo_mask_threshold
    elif config.photo_mask_mode == 'high':
      mask = photo_error > config.photo_mask_threshold
    else:
      raise ValueError(f"Unknown photo_mask_mode: {config.photo_mask_mode}")

    keep_ratio = float(mask.mean())
    keep_ratios.append(keep_ratio)

    name = dataset.image_names[idx]
    out_path = path.join(out_dir, name)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(out_path)

    print(
        f"{idx+1:04d}/{dataset.size:04d} {name} "
        f"keep_ratio={keep_ratio:.4f} "
        f"err_mean={float(photo_error.mean()):.6f} "
        f"err_p90={float(np.percentile(photo_error, 90)):.6f} "
        f"err_p95={float(np.percentile(photo_error, 95)):.6f}"
    )

  print("Done.")
  print(f"mean_keep_ratio={float(np.mean(keep_ratios)):.6f}")
  print(f"median_keep_ratio={float(np.median(keep_ratios)):.6f}")
  print(f"min_keep_ratio={float(np.min(keep_ratios)):.6f}")
  print(f"max_keep_ratio={float(np.max(keep_ratios)):.6f}")


if __name__ == "__main__":
  app.run(main)
