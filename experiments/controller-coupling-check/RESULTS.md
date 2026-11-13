# Results: controller coupling check

*Simulation only. Soft compliant pendulum (soft_none), the main experiment's CEM planner on the learned model `soft_none.pt`, 30 paired seeds (640000–640029), mass ×1.5 at the onset. Window: onset+20 to onset+100 steps (1–5 s after). All numbers come from `results.json`.*

## Outcome: **A**: controller output DROPS on the heavier body → coupling effect confirmed

| metric (window mean per seed) | healthy (mean ± SD) | mass ×1.5 (mean ± SD) | ratio of means | paired ratio |
|---|---|---|---|---|
| motion ‖Δs‖ | 0.0542 ± 0.0218 | 0.0165 ± 0.0051 | ×0.30 | ×0.32 (95% CI 0.27–0.40) |
| controller torque magnitude (N·m) | 0.7742 ± 0.2829 | 0.3887 ± 0.0812 | ×0.50 | ×0.53 (95% CI 0.46–0.63) |
| prediction error ‖s − ŝ‖ | 0.0922 ± 0.0074 | 0.0839 ± 0.0017 | ×0.91 | ×0.91 (95% CI 0.89–0.94) |

- **Main experiment's motion drop:** ×0.30 (median 0.0501 → 0.0150).
- **Motion medians here:** 0.0589 → 0.0172.
- **Torque at the ±2 N·m limit in the window:** healthy 0.1%, mass ×1.5 0.0% of steps.
- **Still upright (|θ| < 90°) throughout the window:** healthy 100%, mass ×1.5 100% of seeds.
- **Check:** healthy and changed runs are identical before the onset (max difference 0.0e+00).

Decision rule (fixed in README.md before running): torque paired-ratio CI entirely below 1 and ≤ 0.90 → A; entirely above 1 and ≥ 1.10 → C; otherwise B. This applies only if the motion paired-ratio CI is entirely below 1.

Plots: `motion_vs_time.png`, `torque_vs_time.png`, `prediction_error_vs_time.png` (mean ± SD across seeds, aligned at the onset).
## Interpretation

**The main experiment's drop is reproduced.** On 30 fresh seeds, the same body, model and controller give a motion ratio of ×0.30 (ratio of means; paired ×0.32, 95% CI 0.27–0.40). The main experiment's ×0.30 lies inside that interval. The drop starts within about 0.5 s of the body change and holds for the whole window (`motion_vs_time.png`). Every run stays upright.

**The controller's torque halves** (×0.50 of means, paired ×0.53, CI 0.46–0.63), in step with the motion (`torque_vs_time.png`). By the pre-set rule, that's **outcome A**. It isn't torque saturation: torque hits the ±2 N·m limit on 0.1% of healthy steps and 0% of changed ones.

**The drop splits into two parts** (arithmetic on the table above, ratio of means):

| part | ratio |
|---|---|
| controller torque, changed / healthy | ×0.50 |
| motion produced per unit of torque, changed / healthy (0.0425 vs 0.0701) | ×0.61 |
| product | **×0.30** |

The second factor is the passive body: a 1.5× heavier link moves less for the same torque. It matches the open-loop result in `experiments/resonance-check-in-env/` (×0.65–0.67 away from resonance; 1/1.5 = 0.67). The first factor is the controller, which commands about half as much torque on the heavier body. **Passive physics alone gives about ×0.6–0.65; the controller's reaction takes it the rest of the way to ×0.30.**

**Prediction error falls too** (×0.91, CI 0.89–0.94). This is the inverted signal from the main experiment, reproduced: on the soft body, the healthy model's one-step error mostly tracks how much the pendulum moves, so a quieter pendulum looks *less* surprising.

## What this does and doesn't establish

- **Established:** the motion drop happens in closed loop with this controller. About half of it (in log terms) is the controller commanding less torque, and the rest is the passive inertia effect.
- **Not established: which way the causation runs inside the loop.** The planner reacts to the state, so "less torque → less motion" and "less motion → the planner sees smaller errors → less torque" are both possible, and probably reinforce each other. Torque falling together with motion is consistent with a feedback loop that amplifies the passive ×0.6 into ×0.30. It doesn't prove the controller started it.
- **A clean next test (not run):** replay each seed's *healthy* torque sequence open-loop on the heavier body after the onset. That breaks the loop. If motion drops only about ×0.6, the extra drop comes from the controller's reaction. If it drops about ×0.30, the body alone does it, even with healthy torques.
- **Scope:** the soft body with no sensor noise (`soft_none`), mass ×1.5 only. The main experiment mixed mass ×1.3–1.7 and length ×1.2–1.5, and also had a low-noise configuration.
