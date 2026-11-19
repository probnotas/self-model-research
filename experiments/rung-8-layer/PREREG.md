# Rung 8 pre-registration

*Written and committed BEFORE the main run.* At the time of this commit, only the healthy pilots and the healthy calibration had been run (see `PILOT.md`, `pilot_balance_results.json` and `calibration.json`). No perturbed episode had been run. Simulation only.

## Fixed design
- **Task:** balance. Start within ±0.2 rad of upright; perturbation at step 50; 500 steps. Success = the arm never passes 90°, and |θ| ≤ 10° on each of the final 50 steps.
- **Conditions:** A policy alone, B filter, C adaptation, D oracle, exactly as in `README.md` and `experiment.py` at this commit.
- **Fixed parameters:** threshold_B = 0.020092913221325028; W = 20; adaptation range 1/g ∈ [0.5, 2]; 3 retries; exclusion radius 0.25 N·m.
- **Perturbations:** healthy, payload_add (mass ×1.5), spring_soft, spring_stiff, damping_high, motor_weak, gravity_tilt, sensor_delay.
- **Seeds:** 882000–882029, 30 per cell, paired across all cells.
- **Success rate** is reported per cell, in percentage points (pp), with Wilson 95% intervals. **Claims are judged on the point estimates, exactly as worded below.**

## Claims
1. **The policy alone degrades.** A's success rate under mass ×1.5 is at least **30 pp** lower than A's healthy success rate.
2. **The oracle barely degrades.** D's success rate under mass ×1.5 is at most **10 pp** lower than D's healthy success rate.
3. **The self-model helps.** Under mass ×1.5, max(B, C) − A ≥ **15 pp**.
4. **Adaptation beats filtering somewhere.** C's success rate is strictly greater than B's on at least one of the 7 perturbation conditions.

Each claim is reported as PASS or FAIL. A FAIL is reported plainly in `RESULTS.md`.

## Disclosed before running (from earlier work and physics; not used to change the claims)
- **Claim 1 may fail.** In `experiments/controller-coupling-check/`, the same controller kept the same soft body upright under mass ×1.5 in 30/30 seeds over 1–5 s after the change. The body got quieter rather than falling. If A doesn't degrade, claim 3 can't pass either.
- **gravity_tilt is expected to be impossible for every condition:** about 3.5 N·m of static torque is needed at θ = 0, against a 2 N·m limit.
- **Claim 4 is a weak test:** a strict inequality on any one of 7 perturbations with 30 seeds can pass by chance. Its result is reported with that caveat.
