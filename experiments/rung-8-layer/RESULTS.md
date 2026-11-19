# Rung 8 results

*Simulation only. Pre-registration: `PREREG.md` (commit `452f32a`, before the main run). This page is scored by `analyze.py`. The only change to `analyze.py` after the pre-registration commit converts NumPy booleans so the JSON can be saved; the scoring logic is untouched. The numbers come from `results.json` and `RESULTS_numbers.md`; per-episode data is in `main_runs.json`.*

## The pre-registered claims

| claim | measured | verdict |
|---|---|---|
| 1. Under mass ×1.5, policy alone degrades ≥ 30 pp | A: 100% healthy → 100% mass ×1.5 (0 pp) | **FAIL** |
| 2. Under mass ×1.5, the oracle degrades ≤ 10 pp | D: 100% → 100% (0 pp) | PASS (trivially: nothing degraded) |
| 3. Under mass ×1.5, max(filter, adaptation) beats policy alone by ≥ 15 pp | max(0%, 100%) − 100% = 0 pp | **FAIL** |
| 4. Adaptation beats filter on ≥ 1 perturbation | C > B on 5 of 7 | PASS, but **degenerate**: the filter failed everywhere, including on the healthy body (see below) |

**Bottom line: the pre-registered hypothesis isn't supported.** On this task, in simulation, adding the self-model as a filter or as adaptation didn't improve success under any body change.
- The main reason: the policy alone never failed under a body change, so there was nothing to fix.
- The filter as designed broke the controller outright.

## Success rates (30 paired seeds per cell; Wilson 95% intervals)

| perturbation | A policy | B filter | C adapt | D oracle |
|---|---|---|---|---|
| healthy | 100% (89–100) | 0% (0–11) | 100% (89–100) | 100% (89–100) |
| payload_add (mass ×1.5) | 100% (89–100) | 0% (0–11) | 100% (89–100) | 100% (89–100) |
| spring_soft (k ×0.5) | 100% (89–100) | 0% (0–11) | 100% (89–100) | 100% (89–100) |
| spring_stiff (k ×2) | 100% (89–100) | 0% (0–11) | 100% (89–100) | 100% (89–100) |
| damping_high (×3) | 100% (89–100) | 0% (0–11) | 100% (89–100) | 100% (89–100) |
| motor_weak (limit ×0.6) | 100% (89–100) | 0% (0–11) | 100% (89–100) | 100% (89–100) |
| gravity_tilt (+45°) | 0% (0–11) | 0% (0–11) | 0% (0–11) | 0% (0–11) |
| sensor_delay (2 steps) | 0% (0–11) | 0% (0–11) | 0% (0–11) | 100% (89–100) |

| perturbation | falls past 90°, A / B / C / D | B steps rejected | C steps adapted | C mean scale (1/g) |
|---|---|---|---|---|
| healthy | 0 / 30 / 0 / 0 | 35.9% | 80.0% | 1.45 |
| payload_add | 0 / 30 / 0 / 0 | 29.2% | 34.6% | 1.62 |
| spring_soft | 0 / 30 / 0 / 0 | 32.6% | 67.4% | 1.56 |
| spring_stiff | 0 / 30 / 0 / 0 | 44.1% | 78.0% | 1.28 |
| damping_high | 0 / 30 / 0 / 0 | 37.6% | 70.2% | 1.48 |
| motor_weak | 0 / 30 / 0 / 0 | 31.9% | 47.8% | 1.61 |
| gravity_tilt | 30 / 30 / 30 / 30 | 89.9% | 5.8% | 1.21 |
| sensor_delay | 11 / 30 / 28 / 0 | 41.0% | 44.2% | 1.14 |

Figure: `success_rates.png`.

## What happened

**1. The body changes didn't hurt the policy.** The MPC policy, planning through the frozen healthy self-model, balanced 30/30 under every body-parameter change: mass ×1.5, spring ×0.5 and ×2, damping ×3, and motor limit ×0.6. Median final error under mass ×1.5 was 2.5° (healthy 2.8°). This matches `experiments/controller-coupling-check/`: on this soft body a heavier arm gets quieter, not unstable. The disclosure in `PREREG.md` flagged this risk. With no degradation, claims 1 and 3 couldn't pass.

**2. The filter (B) broke the controller, even with no body change.** It rejected 36% of commands on the healthy body. After 3 failed replans it applied zero torque, and the pendulum fell in all 30 healthy episodes, ending near hanging (median final error 180°).
- **The cause is the threshold rule:** 2 × the median mismatch was calibrated on the *unfiltered* policy, so far more than a few percent of ordinary commands exceed it. Each rejection also changes the trajectory, which raises later mismatches.
- **Calibrating on the median is too tight for a gate whose fallback is "do nothing".** That's a design error in the filter, not evidence about self-models.
- **It also makes claim 4 meaningless:** adaptation beats a filter that never works.

**3. Adaptation (C) was harmless but useless here.**
- **It succeeded wherever the policy did,** and nowhere else.
- **It adapted most of the time even on the healthy body** (80% of steps, mean scale 1.45). So its online gain g sits around 0.7 on the healthy body: the frozen soft model over-predicts how much a command moves the arm even when nothing has changed. That's consistent with the soft model's known error floor (`findings_arc.md`).
- **On sensor delay it did worse than the policy alone:** 28 falls vs 11.

**4. The two non-body perturbations:**
- **gravity_tilt defeated every condition, as predicted** in `PREREG.md`: about 3.5 N·m of static torque is needed against a 2 N·m motor.
- **sensor_delay is the only perturbation that hurt the policy** (0% success, 11/30 falls), and neither self-model layer helped. The oracle's 100% there isn't comparable: by design it sees the true, undelayed state.

## Caveats
- **Simulation only.** One soft compliant pendulum, one controller (CEM-MPC), one self-model file, no sensor noise.
- **The task was changed from swing-up to balance before pre-registration,** because swing-up is impossible on this body (`PILOT.md`).
- **Binary success hides smaller effects.** A ceiling of 100% for A means this design can't detect a self-model benefit on these body changes, even if one exists in other measures (final error, energy, recovery time).
- **B and C are one specific way each to use a self-model at runtime.** The results speak to these implementations, not to filtering or adaptation in general.

## What a next version would need (not pre-registered here)
- **Perturbations that actually hurt the policy alone,** e.g. larger factors, or changes during more demanding motion, so there's room to improve.
- **A filter calibrated in closed loop with the filter switched on,** using a high quantile instead of 2 × median, and a fallback that isn't zero torque (e.g. hold the last accepted command).
- **A self-model whose healthy gain is close to 1,** so adaptation only acts when something has actually changed.
