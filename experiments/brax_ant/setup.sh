#!/usr/bin/env bash
# RunPod (CUDA 12 template). Installs brax + JAX 0.8.2 with CUDA 12, then checks the GPU.
set -e
pip install -q --ignore-installed blinker   # RunPod images ship a debian blinker that pip can't uninstall
pip install -q brax scikit-learn
pip install -q "jax[cuda12]==0.8.2"          # brax 0.14 breaks on JAX >= 0.9; ignore the flax version warning
python -c "import jax, brax, mujoco; print('jax', jax.__version__, 'brax', brax.__version__, 'mujoco', mujoco.__version__); print('devices:', jax.devices()); assert jax.devices()[0].platform == 'gpu', 'NO GPU VISIBLE'"
