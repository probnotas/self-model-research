#!/usr/bin/env bash
# ~5 min end-to-end check at tiny scale (small PPO, few trials, 2 epochs). Numbers are meaningless.
set -e
cd "$(dirname "$0")"
export SCALE=smoke OUT=${OUT:-/workspace/brax_ant_out_smoke}
python train_policy.py
python verify_legs.py || echo "(verify_legs at smoke scale uses an untrained policy; a FAIL here is a warning, check it at full scale)"
python rollouts.py train
python rollouts.py dev
python evaluate.py dev
python make_results_md.py
echo "smoke OK -> $OUT"
