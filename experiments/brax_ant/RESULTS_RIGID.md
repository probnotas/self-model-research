# Rigid Brax Ant: command gain is a NEGATIVE result

*Simulation only (Brax MJX). Nothing here was run on hardware.*

**Verdict: FAIL.** On the rigid Ant, command gain does not reliably tell body changes from
nothing, and it does not reliably say which leg changed. Only one of the three body changes
(weak motor) is detected cleanly.

## Setup
- **Controller:** PPO policy, final eval reward 3198.3, action noise 0.05.
- **Trials:** 512 dev trials; one event at t = 10 s of a 20 s walk.
- **Body changes (one random leg):** weak motor, 3× heavier leg, damped joint.
- **World events:** short shove, steady push, torso payload.
- **Signal:** command gain, the signal that passed the pre-registered soft-pendulum test.
  The action noise is the probe, and an inverse model is trained on healthy walking only.
  Scored in the late window on the 455 of 512 trials where the Ant stayed upright, over 3
  inverse-model seeds.
- **Bars** (same as for the compliant run, frozen in `prereg.json`):
  - G1: AUC body vs nothing ≥ 0.90.
  - G2: leg accuracy ≥ 0.70.
  - Each must hold on all 3 seeds.

## Dev results (`results/rigid_run1/gain_dev_results.json`)
| | mean (min–max, 3 seeds) | bar | result |
|---|---|---|---|
| G1 detection AUC, body vs nothing | 0.737 (0.716–0.762) | ≥ 0.90 | **FAIL** |
| G2 leg accuracy | 0.546 (0.540–0.555) | ≥ 0.70 | **FAIL** |

**No reliable leg attribution.** With 4 legs, chance is 0.25, so 0.546 is above chance. But
almost all of it comes from one body change: weak-motor trials name the right leg 94% of the
time, while heavy-leg trials manage 0.28 (chance level) and damped-joint trials 0.41. The
dev-review guide used a 0.50 bar, and 0.546 cleared it, but we do not count that as leg
detection working. It is not reliable across body changes.

**By body change (seed 0):**
| body change | AUC vs nothing | right leg | own-leg gain z (median) | flagged at z ≥ 2.5 |
|---|---|---|---|---|
| weak motor | 0.955 | 0.94 | −6.31 | 97 / 111 |
| heavy leg | 0.637 | 0.28 | −0.44 | 20 / 108 |
| damped joint | 0.614 | 0.41 | −0.70 | 21 / 118 |
| (nothing) | — | — | — | 1 / 31 |

Only the weak motor is detected cleanly. The heavy leg and the damped joint are mostly
missed. In all three, the changed leg's gain moves in the predicted direction (z < 0), but
for the heavy leg and the damped joint it moves too little to separate from healthy noise.
The leg-map check (`results/rigid_run1/verify_legs.json`) also found that heavy-leg changes on legs 2 and 4 show
up across neighbouring legs, not just their own.

**Prediction-error self-models (same dev run, `dev_results.json`):**
- H1 (detection AUC 0.742, bar 0.90): fails.
- H2 (0.807, bar 0.80), H3 (0.512, bar 0.50) and H4: pass, by about 0.01 at most.

## Held-out test split
**Not run yet.** The pod was terminated (no GPUs available) before the test split could run,
and the trained rigid policy (run 1, eval reward 3198.3) existed only on that pod's disk. The
test split has never been touched.

Plan, so the test is scored on the same policy it is frozen for:
1. On a new pod, retrain the rigid policy (run 2). PPO uses the same seed, but GPU
   non-determinism means it won't be bit-identical.
2. Re-run rigid dev with that policy and report run 2's dev numbers next to run 1's above.
3. Freeze `prereg.json` with the same G1 and G2 bars, then run the test split once. No tuning.

## Caveats
- The rigid gain bars were frozen **after** the rigid dev run had already failed them. The
  rigid test confirms a dev negative; it is not a blind test. The compliant run's bars are
  frozen before its dev run.
- One controller, one robot design, simulation only.
