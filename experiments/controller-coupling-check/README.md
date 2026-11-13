# Controller coupling check: does the controller cause the motion drop?

*Simulation only, not hardware.* Run with `python experiments/controller-coupling-check/experiment.py` (a few minutes on 4 CPU cores; needs gymnasium, numpy, torch, matplotlib). Results are in [`RESULTS.md`](RESULTS.md).

## Purpose

To find out whether the **×0.30 motion drop** after a body change on the soft compliant pendulum is caused by the **controller reacting to the body change**, rather than by passive dynamics. That drop is behind the inverted fault-detection AUCs (`surprise_attribution/findings_arc.md`, Exp 6–7).

## Context

`experiments/resonance-check-in-env/` drove the same soft body with an open-loop sine. The smallest drop it could produce for mass ×1.5 was **×0.65**, at any drive frequency from 0.1 to 10 rad/s. Passive physics doesn't reach ×0.30, so something else is at play. The controller is the prime suspect.

## What's reused, unchanged

- **Environment:** `surprise_attribution/compliant_env.CompliantPendulum`, with the "soft" stiffness (k = 4) and "none" noise config from `compliant_config` (`soft_none`).
- **Learned forward model:** `surprise_attribution/compliant_models/soft_none.pt`, loaded through `compliant_trials.get_model`. It's cached, never retrained.
- **Controller:** the main experiment's CEM model-predictive controller, `planner.choose_action`, planning with that same forward model. It's a planner on a learned model, not a separately trained policy.
- **Episode loop:** `run_history.record_episode`. It's the same loop that recorded the main experiment's episodes: upright start within ±0.2 rad, 325 steps, body change via `env.change_body` at the onset step.
- **Prediction error:** ‖s<sub>t+1</sub> − ŝ<sub>t+1</sub>‖ from that loop.
- **Motion metric** (as in `run_twosided.py`): mean ‖s<sub>t+1</sub> − s<sub>t</sub>‖ of the observed state over steps onset+20 to onset+100 (1–5 s after onset).

## Design

- **Seeds:** 30, from 640000 upward. They're fresh, and not used by any other experiment or pre-registration.
- **Pairing:** each seed runs twice, once healthy and once with mass ×1.5 at the onset step (drawn from 60–80, the main experiment's range). The planner's random numbers are seeded per episode, so both runs are identical until the onset; the script checks this.
- **Per-timestep log:** arm angle and velocity, controller torque, prediction error. Saved to `episodes.npz`.
- **Per condition:** the motion metric, controller torque magnitude |u| and prediction error, as means and standard deviations across seeds over the same window.
- **Changed vs healthy:** ratios, both as a ratio of means and as the geometric mean of per-seed paired ratios with a 95% bootstrap CI.
- **Plots:** motion, torque and prediction error vs time, healthy and changed overlaid (mean ± SD across seeds, aligned at the onset).

## Decision rule (fixed before running)

The rule uses the paired torque-magnitude ratio, changed / healthy, over the window:

- **A) torque drops:** 95% CI entirely below 1 and point estimate ≤ 0.90. → **Coupling effect confirmed.**
- **C) torque rises:** 95% CI entirely above 1 and point estimate ≥ 1.10. → **The controller is compensating but failing**: a different mechanism.
- **B) otherwise (same):** → **Something else.** Investigate further.

A, B or C only applies if the motion drops (paired motion-ratio CI entirely below 1). If motion doesn't drop under these conditions, the results say so instead.
