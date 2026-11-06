#!/usr/bin/env bash
# RunPod (CUDA 12 template). Installs the pinned stack, then checks the GPU. Safe to re-run
# (also repairs a half-finished install). Packages live on the pod's temporary disk, so
# re-run this after every pod restart; /workspace (policy, outputs) survives.
set -e
cd "$(dirname "$0")"
pip install -q --ignore-installed blinker   # RunPod images ship a debian blinker that pip can't uninstall
pip uninstall -y -q jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt brax mujoco mujoco-mjx flax 2>/dev/null || true
pip install -q -r requirements-pod.txt
python -c "import jax, brax, mujoco, flax; print('jax', jax.__version__, 'brax', brax.__version__, 'mujoco', mujoco.__version__, 'flax', flax.__version__); print('devices:', jax.devices()); assert jax.devices()[0].platform == 'gpu', 'NO GPU VISIBLE'"
