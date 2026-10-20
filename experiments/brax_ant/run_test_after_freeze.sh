#!/usr/bin/env bash
# Only after the dev review and "freeze". Both steps refuse to run without prereg.json.
set -e
cd "$(dirname "$0")"
export SCALE=full OUT=${OUT:-/workspace/brax_ant_out}
python freeze_prereg.py
python rollouts.py test
python evaluate.py test
