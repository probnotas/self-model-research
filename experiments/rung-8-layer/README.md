# Rung 8: does a self-model layer help a controller survive body changes?

*Simulation only, not hardware.* The pre-registered claims are in [`PREREG.md`](PREREG.md), committed before the main run. Results are in [`RESULTS.md`](RESULTS.md).

## Purpose

This is a controlled 4-way comparison. It tests whether a learned self-model, added as a runtime layer to a model-predictive controller (MPC) on a compliant pendulum, improves task success when the body changes. It also asks which runtime use of the self-model works better: **filtering** commands, or **adapting** them.

## What's reused, unchanged

- **Body:** `surprise_attribution/compliant_env.CompliantPendulum`, "soft" (k = 4 N·m/rad), with hysteresis, spring damper, motor inertia and friction, joint damping, and no sensor noise.
  - `rung8.PerturbedCompliant` restates its `step` with the constants exposed so perturbations can be switched on. With no perturbation it matches the original step for step (max |difference| 0.0; see `PILOT.md`).
- **Self-model:** the existing learned forward model `surprise_attribution/compliant_models/soft_none.pt`, not retrained.
- **Policy:** the existing CEM-MPC, `planner.choose_action`, planning through that same model (horizon 20, 100 candidates, 5 iterations, 10 elites).

## Task: balance (changed from swing-up after the pilot)

- **Requested originally:** swing-up from hanging. The pilot (`PILOT.md`) showed that's impossible on this body with its 2 N·m motor, even for an oracle with perfect physics.
- **The task used instead:** balancing, as in the main experiments.
  - **Start:** within ±0.2 rad (and ±0.2 rad/s) of upright.
  - **Length:** 500 steps (25 s), with the perturbation switched on at step 50.
  - **Success:** the arm never passes 90° from upright, and |θ| ≤ 10° on each of the final 50 steps.
- **Healthy pilot** (seeds 880100–880109): A 10/10, oracle 10/10.

## Conditions

The self-model learned the healthy body and never sees the change directly. B and C give it new information by comparing its predictions with what actually happened over the last W = 20 steps:

- **Command gain g:** a least-squares fit of the command's observed effect to its predicted effect.
  - The effect is the next state minus the self-model's prediction for zero torque.
  - g ≈ 1 means commands move the body as expected; g < 1 means they move it less.

| | condition | what it does |
|---|---|---|
| **A** | policy alone | Executes the MPC's command as is. |
| **B** | + self-model filter | For a command u, the policy's own forecast is f(s, u). The self-model's forecast, corrected by g, is f(s, 0) + g·(f(s, u) − f(s, 0)). Their mismatch is \|1 − g\|·‖f(s, u) − f(s, 0)‖. If it's ≥ threshold_B, the command is rejected and the MPC replans with it excluded (any candidate whose first torque is within 0.25 N·m), up to 3 retries. If every retry fails, it applies zero torque. threshold_B = 2 × the median mismatch on healthy calibration episodes (seeds 881000–881019) = **0.0201**, fixed before any perturbation run (`calibration.json`). Logged: rejection rate per episode. |
| **C** | + self-model adaptation | Executes u / g (clipped to ±2 N·m) when 0.5 ≤ 1/g ≤ 2, and u otherwise. Logged: mean scale and fraction of steps adapted. |
| **D** | oracle | The same CEM, but rolling out the true physics with the true post-change parameters from the true full state (including the hidden motor state and, under sensor delay, the undelayed state). An upper bound. |

## Perturbations (switched on at step 50)

| name | change | note |
|---|---|---|
| healthy | none | |
| payload_add | arm mass ×1.5 (1.0 → 1.5 kg) | the main experiment's `change_body` hook; this is "mass ×1.5" in the claims |
| spring_soft | k ×0.5 | |
| spring_stiff | k ×2 | |
| damping_high | joint damping ×3 | |
| motor_weak | torque limit ×0.6 (2 → 1.2 N·m) | the policy still plans up to ±2 |
| gravity_tilt | gravity rotated +45° | holding θ = 0 then needs about 3.5 N·m of static torque (1 × 10 × 0.5 × sin 45°) against a 2 N·m limit, so no condition is expected to succeed |
| sensor_delay | observations 2 steps old | not a body parameter; the oracle gets the true current state |

## Design

- **Seeds:** 30 per cell (882000–882029), shared across conditions and perturbations (paired). That's 4 × 8 × 30 = 960 episodes.
- **Run:**
  ```
  python experiments/rung-8-layer/experiment.py main
  python experiments/rung-8-layer/analyze.py
  ```
