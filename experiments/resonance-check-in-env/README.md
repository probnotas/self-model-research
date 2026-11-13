# Resonance check inside the actual compliant sim

*Simulation only, not hardware.* Run with `python experiments/resonance-check-in-env/experiment.py` (about 30 s on a CPU; needs gymnasium, numpy, matplotlib). Results are in [`RESULTS.md`](RESULTS.md).

## Purpose

On the soft compliant pendulum, a body change made the robot **quieter**: median motion dropped from 0.0501 to 0.0150 (×0.30, `surprise_attribution/findings_arc.md`, Exp 10). That's why "more surprise = body change" scored inverted (AUC 0.11 as I reported it; the soft-body AUCs in the repo range from 0.05 to 0.48 across analyses).

On Nov 11, 2026, Prof. Shai Revzen (UMich BIRDS Lab) suggested this could be **pure resonance detuning**: the body change moves the system's natural frequency, and the same motor effort then produces less motion.

## Why this experiment

A simplified torsional-spring playground (`experiments/resonance-check/`) showed resonance alone gives at best about a ×0.45 drop at my body stiffness (k = 4), not ×0.30. But that playground is a different system from my sim:
- **No motor inertia or friction.**
- **No hysteresis.**
- **Different gravity and inertia terms.**

This experiment tests resonance directly **in my actual compliant sim environment**, so the comparison is apples-to-apples.

## Method

- **The environment is reused unchanged:** `surprise_attribution/compliant_env.CompliantPendulum` with the "soft" stiffness from `compliant_config` (k = 4 N·m/rad). That keeps the motor inertia and friction, the ±30% hysteresis, the spring damper, the joint damping, and Pendulum-v1's m = 1, l = 1, g = 10, dt = 0.05, torque limit 2 and speed limit 8.
  - Sensor noise only affects what the robot observes, never the physics, so amplitudes are measured on the true link angle.
- **The body change uses the same hook as the main experiment:** `env.change_body({"m": 1.5})`, applied right after reset.
  - Extra condition (not in the original spec): length ×1.35. Half of the main experiment's body changes were length changes (×1.2–1.5).
- **Drive:** the planner is replaced by an open-loop torque `u(t) = A sin(ω t)`, with A = 2 N·m (the env's maximum).
  - ω ∈ {0.1, 0.3, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 8, 10} rad/s, plus a fine grid of 2–6 rad/s in steps of 0.05 around the peak.
- **Timing:** start hanging at rest, run 30 s for the transient to die out, then measure over the last 10 s (stretched to one full period at slow drives).
- **Amplitude:** peak-to-peak / 2 of the arm angle. Robustness check: 5 random Pendulum-v1 start states per point.
- **Outputs:**
  - `amplitude_vs_freq.png`: healthy vs changed curves.
  - `ratio_vs_freq.png`: changed / healthy, with the main experiment's ×0.30 marked.
  - `results.json`: every run.
  - `RESULTS.md`: the numbers and interpretation.

**Decision rule (set before running):** if the minimum ratio is around 0.30, resonance can explain the main experiment's motion drop; if it's above 0.35, something beyond resonance is contributing.

## The main limitation, stated up front

The main experiment's pendulum was **balanced upright by a controller**. With only 2 N·m of torque against gravity (up to 5 N·m here), an open-loop sine can't hold it upright: it falls and swings around the **bottom**, where gravity acts as a restoring spring. Upright, the soft joint has no natural frequency of its own (gravity pushes away from the top, and k = 4 is too weak to hold against it). The oscillation there is created by the controller.

So this experiment measures the resonance contribution **of this exact body**, but not the closed-loop dynamics that produced ×0.30.
