# Rung 8 pilot (run BEFORE any pre-registration)

*Simulation only.* This is the healthy-body feasibility check for the swing-up task: start hanging at θ = π, then hold |θ| ≤ 10° for 50 consecutive steps within 500 steps. It was run before any claims were written down.

| driver (healthy soft body, k = 4, from hanging) | seeds | success | best lift above hanging |
|---|---|---|---|
| Condition A: CEM-MPC through `soft_none.pt`, horizon 20 (`pilot.py`) | 10 | 0/10 | 6.2–8.1° |
| Oracle: same CEM on the true physics, horizon 20 (`pilot_feasibility.py`) | 3 | 0/3 | 13.5–14.1° |
| Oracle, horizon 60 | 3 | 0/3 | 2.5–4.4° |
| Condition A, horizon 60 / 100 | 3 / 3 | 0/3, 0/3 | 3.9–4.8° |
| Energy pumping, 2 N·m in the direction of motion, then a PD catch | 1 | 0/1 | 62° (118° short of upright) |
| Steady sine at resonance (`experiments/resonance-check-in-env/`) | n/a | n/a | ±49° amplitude |

`PerturbedCompliant` (in `rung8.py`) matches `CompliantPendulum` step for step with no perturbation: max |difference| 0.0 over 400 random-torque steps.

**Conclusion:** with the 2 N·m torque limit, the soft series-elastic joint (with its spring damper, motor friction and joint damping) loses energy faster than any driver tried could add it. Swing-up isn't achievable here, even with perfect physics knowledge. Every condition would score 0% under every perturbation, so the pre-registered claims couldn't be tested. The task has to change before anything is pre-registered.
