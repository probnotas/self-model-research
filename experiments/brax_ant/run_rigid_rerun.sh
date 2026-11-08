#!/usr/bin/env bash
# Rigid run 2: run 1's policy was lost with its pod. Retrain, dev, freeze (same G1/G2 bars), test once.
# The user authorized the rigid test; pod.sh still needs ALLOW_TEST=1.
set -e
cd "$(dirname "$0")"
export SCALE=${SCALE:-full} OUT=${OUT:-/workspace/brax_ant_out} COMPLIANT=0
python train_policy.py
python verify_legs.py
python rollouts.py train
python rollouts.py dev
python evaluate.py dev
python gain.py dev
python make_results_md.py
python freeze_prereg.py
python rollouts.py test
python evaluate.py test
python gain.py test
echo "RIGID run 2 DEV+TEST done"
