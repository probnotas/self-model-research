#!/usr/bin/env bash
# Drive the RunPod GPU pod over SSH from your own computer (macOS / Linux / WSL).
#
#   ./pod.sh check            connection + GPU
#   ./pod.sh push             copy this folder's code to /workspace/brax_ant on the pod
#   ./pod.sh setup            install the pinned packages (after every pod start)
#   ./pod.sh run [script]     start a pipeline in the background (default run_full_dev.sh)
#   ./pod.sh status           is it still running? + last lines of the log
#   ./pod.sh log [n]          last n log lines (default 40)
#   ./pod.sh fetch            copy results (json, results.md) into ./results/
#   ./pod.sh sh 'command'     run any command on the pod
#
# Reads pod.env (see pod.env.example). The test split is refused unless ALLOW_TEST=1,
# which is only set after the user has reviewed dev and said "freeze".
set -euo pipefail
cd "$(dirname "$0")"
[ -f pod.env ] || { echo "missing pod.env: copy pod.env.example and fill it in"; exit 1; }
# shellcheck disable=SC1091
source pod.env
KEY="${SSH_KEY/#\~/$HOME}"
REMOTE=/workspace/brax_ant
OUT=/workspace/brax_ant_out
LOG=/workspace/run.log
PID=/workspace/run.pid
# alive = pid file exists and the process is in any state but Z (containers often don't reap zombies)
ALIVE="[ -f $PID ] && st=\$(awk '/^State:/{print \$2}' /proc/\$(cat $PID)/status 2>/dev/null) && [ -n \"\$st\" ] && [ \"\$st\" != Z ]"
SSH=(ssh -i "$KEY" -p "$POD_PORT" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "$POD_USER@$POD_HOST")

cmd="${1:-}"; shift || true
case "$cmd" in
  check)
    "${SSH[@]}" 'echo "connected to $(hostname)"; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader; df -h /workspace | tail -1; ls '"$OUT"'/rigid 2>/dev/null | head -20 || echo "(no outputs yet)"' ;;
  push)
    tar czf - --exclude=pod.env --exclude=results --exclude=out --exclude=__pycache__ --exclude='*.zip' . \
      | "${SSH[@]}" "mkdir -p $REMOTE && tar xzf - -C $REMOTE && echo pushed to $REMOTE" ;;
  setup)
    "${SSH[@]}" "cd $REMOTE && bash setup.sh" ;;
  run)
    script="${1:-run_full_dev.sh}"
    if [ "$script" = run_test_after_freeze.sh ] && [ "${ALLOW_TEST:-0}" != 1 ]; then
      echo "refusing: the test split runs only after the user reviews dev and says \"freeze\" (then ALLOW_TEST=1)"; exit 1
    fi
    "${SSH[@]}" "cd $REMOTE && if $ALIVE; then echo 'a pipeline is already running'; exit 1; fi; \
      echo \"=== $script \$(date -u)\" >> $LOG; OUT=$OUT nohup bash $script >> $LOG 2>&1 < /dev/null & echo \$! > $PID; echo started $script, log $LOG" ;;
  status)
    "${SSH[@]}" "if $ALIVE; then echo RUNNING; else echo 'NOT RUNNING (finished or crashed: read the log)'; fi; echo ---; tail -n 15 $LOG 2>/dev/null" ;;
  log)
    "${SSH[@]}" "tail -n ${1:-40} $LOG" ;;
  fetch)
    mkdir -p results
    "${SSH[@]}" "cd $OUT && tar czf - --ignore-failed-read */*.json */*.md 2>/dev/null" | tar xzf - -C results
    echo "fetched into $(pwd)/results:"; find results -type f | sort ;;
  sh)
    "${SSH[@]}" "$*" ;;
  *)
    sed -n '2,13p' "$0"; exit 1 ;;
esac
