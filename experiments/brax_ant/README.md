# Brax Ant: body change vs world event

*Simulation only (Brax MJX), not hardware.*

This ports the soft-pendulum research to a walking robot, as suggested by AMI Labs staff.

**The question.** A self-model is trained only on a healthy walking Ant. After an event at t = 10 s:
1. Does it **notice**?
2. Can it tell a **body change** (weak motor, 3× heavier leg, damped joint, on one random leg) from a **world event** (short shove, steady push, torso payload)?
3. Can it say **which leg** changed?

## Files

| file | role |
|---|---|
| `common.py` | config (env vars `SCALE`, `COMPLIANT`, `OUT`), leg map derived from the model, events, trial loop, fall check |
| `train_policy.py` | PPO with the standard Ant hyperparameters; saves the policy; exits with code 2 if the eval reward is < 5000 at full scale |
| `verify_legs.py` | checks that each body event mainly changes its own leg's joint velocities (gotcha 1) |
| `rollouts.py` | frozen policy plus 0.1 action noise; 400-step trials; event at step 200; splits `train`, `dev`, `test` (test refused until frozen) |
| `selfmodels.py` | one-step and multi-step (10-step) self-models, MLP 3×512, GELU, AdamW, 30 epochs |
| `evaluate.py` | scores, raw AUCs, H1–H4 per seed, falls, per-event breakdowns; the test split needs `prereg.json` and stores its sha256 |
| `freeze_prereg.py` | writes `prereg.json` once, recording code hashes and the dev-results hash. Run only after the dev review |
| `make_results_md.py` | writes a plain-language `results.md` from `dev_results.json` |

**Leg map, checked against the model** (`verify_legs.py` prints it):

| leg | actuators | dofs | bodies | obs dims |
|---|---|---|---|---|
| 1 | 2, 3 | 6, 7 | 2, 3 | 5, 6, 19, 20 |
| 2 | 4, 5 | 8, 9 | 4, 5 | 7, 8, 21, 22 |
| 3 | 6, 7 | 10, 11 | 6, 7 | 9, 10, 23, 24 |
| 4 | 0, 1 | 12, 13 | 8, 9 | 11, 12, 25, 26 |

## RunPod commands (RTX 4090 or A100, CUDA 12 template, network volume at /workspace)

```bash
# 1. setup + GPU check
cd /workspace && git clone <this repo> self-model-research   # or git pull
cd self-model-research/experiments/brax_ant
bash setup.sh                       # must print a gpu device; ignore the flax version warning

# 2. ~5 min smoke run (tiny scale; numbers meaningless)
bash run_smoke.sh                   # writes to /workspace/brax_ant_out_smoke/rigid

# 3. full run up to and including DEV (PPO ~20-60 min on one GPU, rest ~30-60 min)
nohup bash run_full_dev.sh > /workspace/full_dev.log 2>&1 &
tail -f /workspace/full_dev.log
# then send back: /workspace/brax_ant_out/rigid/{dev_results.json,results.md,policy_report.json,verify_legs.json}

# 4. ONLY after the dev review and "freeze":
# bash run_test_after_freeze.sh

# Compliant version (after the rigid one is finished): same commands with COMPLIANT=1, e.g.
# COMPLIANT=1 nohup bash run_full_dev.sh > /workspace/full_dev_compliant.log 2>&1 &
```

**Stop conditions built in:**
- `train_policy.py` stops the run if the eval reward is < 5000.
- `verify_legs.py` stops the run if any body event doesn't mainly hit its own leg.
- `evaluate.py` flags `recommend_lower_push` if short pushes knock the Ant over more than 20% of the time.
