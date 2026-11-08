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
| `common.py` | config (env vars `SCALE`, `COMPLIANT`, `SEA_DEFLECT`, `OUT`), leg map derived from the model, series-elastic joints (`COMPLIANT=1`, see `PREREG_COMPLIANT.md`), events, trial loop, fall check |
| `train_policy.py` | PPO with the standard Ant hyperparameters; saves the policy; exits with code 2 if the eval reward is < `REWARD_TARGET` at full scale |
| `verify_legs.py` | checks that each body event mainly changes its own leg's joint velocities (gotcha 1) |
| `rollouts.py` | frozen policy plus action noise (0.05; see below); 400-step trials; event at step 200; saves what the policy asked for (`base`) and the noisy action; splits `train`, `dev`, `test` (test refused until frozen) |
| `selfmodels.py` | one-step and multi-step (10-step) self-models, MLP 3×512, GELU, AdamW, 30 epochs |
| `evaluate.py` | scores, raw AUCs, H1–H4 per seed, falls, per-event breakdowns; the test split needs `prereg.json` and stores its sha256 |
| `gain.py` | **command gain**, the pendulum's winning signal: an inverse model learns how much each motor's noise nudge moves its leg; a leg whose gain drops is the changed leg. Upright trials only; z-scored against held-out healthy train trials |
| `freeze_prereg.py` | writes `prereg.json` once, recording code hashes and the dev-results hash. Run only after the dev review |
| `make_results_md.py` | writes a plain-language `results.md` from `dev_results.json` |
| `run_rigid_rerun.sh` | rigid run 2 (run 1's policy was lost with its pod): retrain, dev, freeze, test once |
| `run_compliant.sh` | compliant run as pre-registered: stiffness ladder (reward only), dev, freeze, test once on seeds 888000+ |
| `run_all_pending.sh` | both of the above, in one pod session. `pod.sh` refuses all three without `ALLOW_TEST=1` |
| `RESULTS_RIGID.md` | rigid result, recorded as a negative |
| `PREREG_COMPLIANT.md` | compliant pre-registration, committed before any compliant run |

**Leg map, checked against the model** (`verify_legs.py` prints it):

| leg | actuators | dofs | bodies | obs dims |
|---|---|---|---|---|
| 1 | 2, 3 | 6, 7 | 2, 3 | 5, 6, 19, 20 |
| 2 | 4, 5 | 8, 9 | 4, 5 | 7, 8, 21, 22 |
| 3 | 6, 7 | 10, 11 | 6, 7 | 9, 10, 23, 24 |
| 4 | 0, 1 | 12, 13 | 8, 9 | 11, 12, 25, 26 |

## Running from your own computer over SSH (recommended)

Open this folder in a Claude Code session on your computer (it reads `CLAUDE.md`), or run the commands yourself:

```bash
cp pod.env.example pod.env      # fill in from RunPod -> Connect -> "SSH over exposed TCP"
./pod.sh check && ./pod.sh push && ./pod.sh setup && ./pod.sh run
./pod.sh status                 # later; then ./pod.sh fetch
```
Needs macOS, Linux or WSL with `ssh` and `tar`. Your SSH public key must be in RunPod (Settings -> SSH Public Keys) before the pod starts. Package versions are pinned in `requirements-pod.txt`.

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

# Rigid run 2 + compliant (series-elastic) run, dev and test, as pre-registered:
# ALLOW_TEST=1 ./pod.sh run run_all_pending.sh
```

**Stop conditions built in:**
- `train_policy.py` stops the run if the eval reward is < `REWARD_TARGET` (2500; see the deviations).
- `verify_legs.py` stops the run if any body event doesn't mainly hit its own leg.
- `evaluate.py` flags `recommend_lower_push` if short pushes knock the Ant over more than 20% of the time.

## Deviations from the original spec (decided after the first dev run)

- **Policy reward.** MJX PPO with the standard settings reached 2789 (peak 3098), not about 5000. Accepted; the gate is now 2500.
- **Action noise 0.1 → 0.05.** Healthy 20 s walks flipped 23/128 at 0.1, 7/128 at 0.05 and 11/128 at 0 (seeds 90000+). Override with `ACTION_NOISE=...`. Rollouts and cached models record the noise, and scripts refuse to mix noise levels.
- **First dev run** (noise not recorded, so possibly still 0.1): H1 0.70, H2 0.78, H3 about 0.5, all failing; H4 passed by 0.02. Healthy fall rate 0.19. Plain prediction error is a weak body-change signal here, as on the soft pendulum.
- **Added `gain.py`** (command gain), the signal that passed the pre-registered pendulum test. It is exploratory on dev; whatever is chosen gets written into `prereg.json` before the test split is touched.
