# Troubleshooting and Citation

Common failure modes, path notes, and citation information for the project.

## 9. Troubleshooting

- If `python -m train` cannot find `train.py`, make sure you are inside `outdoor-nerf-depth/nerf-methods/mipnerf360`.
- If CUDA/JAX fails to initialize, confirm that the installed JAX version matches the machine's CUDA version. This README follows the current `environment.yml`, which uses `jax[cuda12]==0.4.30`.
- If a depth folder is missing, check the selected sequence folder and choose a valid `depth_sup_type`.
- If training starts but checkpoints are not appearing, confirm that `checkpoint_dir` is writable.
- If using NeRF++ or Instant-NGP batch scripts, update the hard-coded data/output paths before launching.
- If `splatfacto` fails with `unsupported GNU version`, install/use GCC and G++ 11 through conda and export `CC`, `CXX`, and `CUDAHOSTCXX`.
- If `splatfacto` fails with `fatal error: crypt.h: No such file or directory`, install `libxcrypt` and export `CPATH`, `C_INCLUDE_PATH`, and `CPLUS_INCLUDE_PATH` to include `$CONDA_PREFIX/include`.
- If the AWS machine freezes during `splatfacto` startup, set `MAX_JOBS=1` and `TORCHDYNAMO_DISABLE=1` before running `ns-train`.

## Citation

This project uses code and data organization from:

```bibtex
@article{wang2023digging,
    title={Digging into Depth Priors for Outdoor Neural Radiance Fields},
    author={Chen Wang and Jiadai Sun and Lina Liu and Chenming Wu
            and Zhelun Shen and Dayan Wu and Yuchao Dai and Liangjun Zhang},
    journal={Proceedings of the 31th ACM International Conference on Multimedia},
    year={2023}
}
```
---
