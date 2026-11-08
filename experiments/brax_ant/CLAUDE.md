# Brax Ant experiment: briefing for a Claude session on the user's computer

You are driving a RunPod GPU pod over SSH with `./pod.sh` (read its header). The code in this
folder is the source of truth: edit here, then `./pod.sh push`. Never edit files on the pod.

## The research (short)
Armaan's self-model research: can a robot tell a **body change** (weak motor, 3x heavier leg,
damped joint, on one random leg) from a **world event** (short shove, steady push, torso
payload), and say **which leg**? It started on a simulated soft pendulum. There, plain
prediction error was a poor signal, and **command gain** plus a fell/body/world gate passed
a pre-registered test. This folder ports the question to the Brax Ant (MJX). Simulation
only, never hardware: say so in every write-up.

## Where things stand
- **Rigid run 1** (reward 3198, noise 0.05; `results/rigid_run1/`, `RESULTS_RIGID.md`): command
  gain AUC 0.737 vs 0.90, leg accuracy 0.546 (chance 0.25; bar 0.70), recorded as a NEGATIVE.
  Its policy was lost with the pod; the rigid test was never run.
- **Compliant (series-elastic) run** pre-registered in `PREREG_COMPLIANT.md` (G1 AUC >= 0.90,
  G2 leg acc >= 0.70, all 3 seeds; test seeds 888000+).
- **Next:** on a new pod, `ALLOW_TEST=1 ./pod.sh run run_all_pending.sh` (rigid run 2 dev+test,
  then compliant ladder+dev+test). The user authorized both test splits on 2026-11-08.

## Rules (from the user; do not bend them)
1. **Never run the test split** (`run_test_after_freeze.sh`, `rollouts.py test`,
   `evaluate.py test`, `gain.py test`) until the user has reviewed dev results and said
   **"freeze"**. `pod.sh` refuses it unless `ALLOW_TEST=1`; set that only after "freeze".
2. **Don't invent numbers.** Report only what the logs and JSON files say. Report negative
   results plainly.
3. **No spec changes without asking.** If something needs one (thresholds, noise, events,
   windows), propose it with the evidence and wait.
4. **Commit and push** to branch `claude/surprise-attribution-experiment-ivcvjp` of
   `probnotas/self-model-research`. Only commit if this folder is inside a clone of that
   repo. Do not open a PR unless asked.
5. **The pod bills while it runs.** When a run finishes and results are fetched, remind the
   user to Stop (or Terminate) the pod in the RunPod dashboard.

## Workflow
```bash
cp pod.env.example pod.env      # once; fill in from RunPod -> Connect -> "SSH over exposed TCP"
./pod.sh check                  # connection + GPU; lists existing outputs
./pod.sh push                   # code -> /workspace/brax_ant
./pod.sh setup                  # pinned install; needed after EVERY pod start (packages aren't on /workspace)
./pod.sh run                    # run_full_dev.sh in the background
./pod.sh status                 # RUNNING / NOT RUNNING + log tail; check every ~10-15 min, don't spam
./pod.sh fetch                  # results/rigid/{dev_results.json,gain_dev_results.json,results.md,...}
```
**Signs to watch for:**
- `setup` must end with `devices: [CudaDevice(id=0)]`. If you see `NO GPU VISIBLE`, the pod
  has no GPU: the user must start a new pod.
- The run log should show `action noise 0.05`. PPO prints `eval reward`, should reach about
  2500 or more, and exits with code 2 below that.
- **"DEV done"** at the end means it succeeded.
- **If it crashes:** read the traceback, fix the code here, push, and re-run. Finished
  steps that cache files (policy, models keyed by noise) are skipped automatically.

**Known pitfalls:**
- An unpinned `pip install brax` pulls a newer JAX that breaks everything; always use `setup`.
- RunPod's debian `blinker` needs `--ignore-installed`; `setup.sh` handles that.
- Rollouts made at a different noise are refused on purpose: regenerate them, don't bypass
  the check.

## Reading the dev results (for the user, in plain language)
- **Self-model scores** (`dev_results.json`): H1-H4 per seed (pass only if all 3 seeds pass),
  fall rates, and a per-event table.
- **Command gain** (`gain_dev_results.json`, `late` window, upright trials only):
  - AUC body vs nothing, and body vs each world event; 0.5 = coin flip, below 0.5 = backwards.
  - Leg accuracy (chance = 0.25).
  - `median_own_leg_z` per body event: below 0 means the changed leg's gain dropped, as
    predicted.
  - `flagged_body` gives counts at z thresholds 2, 2.5 and 3.
- **Then help the user decide:**
  - If command gain clears bars like AUC ≥ 0.90 and leg accuracy ≥ 0.50 on all 3 seeds,
    propose hypotheses and thresholds to add to `freeze_prereg.py`, then wait for "freeze".
  - If it doesn't, say so plainly and help write it up as a negative result (the method
    works on a soft pendulum, not on a legged robot).
