# Troubleshooting

Common failure modes and path notes.

- If `python -m train` cannot find `train.py`, make sure you are inside `outdoor-nerf-depth/nerf-methods/mipnerf360`.
- If CUDA/JAX fails to initialize, confirm that the installed JAX version matches the machine's CUDA version. The current `environment.yml` uses `jax[cuda12]==0.4.30`.
- If a depth folder is missing, check the selected sequence folder and choose a valid `depth_sup_type`.
- If training starts but checkpoints are not appearing, confirm that `checkpoint_dir` is writable.
- If using NeRF++ or Instant-NGP batch scripts, update the hard-coded data/output paths before launching.
- If `splatfacto` fails with `unsupported GNU version`, install/use GCC and G++ 11 through conda and export `CC`, `CXX`, and `CUDAHOSTCXX`. See [nerfstudio-splatfacto.md](nerfstudio-splatfacto.md) for the full AWS/CUDA compile walkthrough.
- If `splatfacto` fails with `fatal error: crypt.h: No such file or directory`, install `libxcrypt` and export `CPATH`, `C_INCLUDE_PATH`, and `CPLUS_INCLUDE_PATH` to include `$CONDA_PREFIX/include`.
- If the AWS machine freezes during `splatfacto` startup, set `MAX_JOBS=1` and `TORCHDYNAMO_DISABLE=1` before running `ns-train`.
