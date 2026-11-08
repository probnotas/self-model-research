# Pre-registration: compliant (series-elastic) Brax Ant, command gain

*Simulation only (Brax MJX). The compliant joint is a numerical model, not hardware.*

This file was committed before any compliant run: no compliant policy, rollout or score
existed when it was written. Check the git history for the commit time.

## Why
On the soft pendulum, command gain passed a pre-registered test on a **compliant** body. The
rigid Ant failed: dev AUC 0.737, and no reliable leg attribution (`RESULTS_RIGID.md`). This run
asks whether the method works on a legged robot whose joints are compliant like the pendulum's.
The writeup reports rigid and compliant side by side.

## The body
`common.py`, `COMPLIANT=1`. Each of the 8 Ant joints becomes a series-elastic joint, built the
same way as `surprise_attribution/compliant_env.py`:
- **Motor:** `phi_dd = (gear*action - tau_s - B_m*phi_d) / J_m`.
  - J_m = 0.15 × the joint's inertia at rest (pendulum: 0.05 / (1/3)).
  - B_m = J_m / 0.1 s (pendulum: 0.05 / 0.5).
  - The action is motor torque, as in the rigid Ant.
- **Spring:** `tau_s = k*d*(1 + 0.3*tanh(d*d_dot/1e-3)) + c*d_dot`, where d = motor angle − link
  angle.
  - ±30% hysteresis, as in the pendulum.
  - c = 0.3 × critical damping 2√(k·J_m), as in the pendulum.
- **Link:** the unchanged MuJoCo Ant joint, driven only by the spring torque.
- **Sensors:** Gaussian noise on the joint angles and joint velocities the policy and models
  read: 0.002 rad and 0.02 rad/s, the pendulum's "low" level.
- **Integration:** physics step 0.005 s (the pendulum's substep), 10 frames per 0.05 s control
  step.
- **Events:** the same as rigid.
  - Weak motor scales the motor's gear.
  - Heavy leg and damped joint act on the link.
  - Short shove, steady push and torso payload act on the torso.
  - Same windows; action noise 0.05.

## Stiffness: one level, chosen by walking ability only
- **Ladder:** spring deflection at full motor torque of 0.4, 0.2, then 0.1 rad (k = gear / deflection).
- **Rule:** use the **softest** rung whose PPO eval reward is ≥ 2500 (the rigid gate).
  - The choice uses only the PPO reward, never any attribution number.
  - This is the pendulum's rule: stiffness was "chosen only on whether the joint can balance".
- **If no rung reaches 2500:** stop and report, with no attribution run.

## Hypotheses and bars (frozen into `prereg.json` by `freeze_prereg.py`)
Command gain (`gain.py`), late window (15–19.5 s vs 1–9.5 s), trials where the Ant stayed
upright:
- **G1 detection:** AUC body change vs nothing ≥ **0.90**.
- **G2 leg attribution:** leg accuracy on body changes ≥ **0.70**. Chance is 0.25 with 4 legs.
  0.70 is a real bar, not "above chance".
- **Seeds:** each passes only if it holds on **all 3** inverse-model seeds (0, 1, 2).

Also reported, with no pass/fail: per-event AUC and leg accuracy, AUC body vs each world
event, own-leg z, flag counts, fall rates, and the prediction-error self-models (H1–H4,
unchanged from the rigid run).

## Splits and seeds
- **Train:** seeds 1000+ (healthy only).
- **Dev:** seeds 2000+.
- **Held-out test:** fresh seeds **888000+**, never used by any run, rigid or compliant. The
  rigid test uses 777000+.
- **Order** (`run_compliant.sh`, no human step in between): ladder → leg-map check (as for rigid, except that a weak motor is checked on the motor's own velocity, where it acts) → dev →
  freeze `prereg.json` (code hashes plus dev-results hash) → test, scored once.

## Honesty rules
- No tuning to pass.
- Negatives are reported as negatives.
- A chance-level number is never called a success.
- Dev numbers are reported but decide nothing. The verdict is G1 and G2 on the **test** split.
