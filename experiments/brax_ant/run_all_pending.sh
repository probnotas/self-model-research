#!/usr/bin/env bash
# Everything still owed, in one pod session: rigid run 2 (dev + test), then the compliant run (dev + test).
set -e
cd "$(dirname "$0")"
bash run_rigid_rerun.sh
bash run_compliant.sh
echo "ALL done"
