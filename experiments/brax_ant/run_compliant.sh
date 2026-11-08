#!/usr/bin/env bash
# Compliant (series-elastic) Ant, end to end, exactly as pre-registered in PREREG_COMPLIANT.md.
# The user authorized dev AND the held-out test for this run; pod.sh still needs ALLOW_TEST=1.
# 1) stiffness ladder: the softest SEA_DEFLECT whose PPO eval reward >= REWARD_TARGET (reward only, no attribution)
# 2) dev  3) freeze prereg.json  4) held-out test (seeds 888000+), scored once. No tuning anywhere.
set -e
cd "$(dirname "$0")"
export SCALE=${SCALE:-full} OUT=${OUT:-/workspace/brax_ant_out} COMPLIANT=1
CHOSEN=""
for D in 0.4 0.2 0.1; do
  echo "=== stiffness ladder: SEA_DEFLECT=$D"
  set +e; SEA_DEFLECT=$D python train_policy.py; rc=$?; set -e
  if [ $rc -eq 0 ]; then CHOSEN=$D; break; elif [ $rc -ne 2 ]; then exit $rc; fi
done
[ -n "$CHOSEN" ] || { echo "NO STIFFNESS REACHED THE REWARD GATE: stop and report (no attribution run)"; exit 3; }
echo "=== chosen SEA_DEFLECT=$CHOSEN"
export SEA_DEFLECT=$CHOSEN
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
echo "COMPLIANT DEV+TEST done (SEA_DEFLECT=$CHOSEN)"
