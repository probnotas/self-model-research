#!/usr/bin/env bash
# Full run up to and including the DEV evaluation. Stops if the policy reward is below 5000 or the leg map check fails.
set -e
cd "$(dirname "$0")"
export SCALE=full OUT=${OUT:-/workspace/brax_ant_out}
python train_policy.py          # exits 2 if eval reward < 5000 -> stop and report
python verify_legs.py           # exits non-zero if a body event doesn't mainly hit its own leg
python rollouts.py train
python rollouts.py dev
python evaluate.py dev
python make_results_md.py
echo "DEV done. Send me $OUT/{dev_results.json,results.md,policy_report.json,verify_legs.json}. Do NOT run the test split."
