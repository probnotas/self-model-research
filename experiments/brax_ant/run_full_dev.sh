#!/usr/bin/env bash
# Full run up to and including the DEV evaluation (self-model scores + command gain).
# Stops if the policy reward is below REWARD_TARGET or the leg map check fails. Reuses an existing policy.
set -e
cd "$(dirname "$0")"
export SCALE=full OUT=${OUT:-/workspace/brax_ant_out}
python train_policy.py          # skips if a policy exists; exits 2 if eval reward < REWARD_TARGET
python verify_legs.py           # exits non-zero if a body event doesn't mainly hit its own leg
python rollouts.py train
python rollouts.py dev
python evaluate.py dev
python gain.py dev
python make_results_md.py
echo "DEV done. Send me $OUT/{dev_results.json,gain_dev_results.json,results.md,policy_report.json,verify_legs.json}. Do NOT run the test split."
