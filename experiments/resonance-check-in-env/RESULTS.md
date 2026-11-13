# Results: resonance check inside the actual compliant sim

*Simulation only. Soft compliant pendulum (k = 4 N·m/rad, `compliant_env.py` unchanged), open-loop drive u = 2 sin(ωt). All numbers come from `results.json`.*

## The rule (set before running)

**If the minimum ratio is around 0.30, resonance can explain the main experiment's motion drop; if it's larger than 0.35, something beyond resonance is contributing.**

## Numbers

| | healthy | mass ×1.5 | length ×1.35 (extra) |
|---|---|---|---|
| resonance peak (coarse grid) | 3.5 rad/s | 3.5 rad/s | 3.0 rad/s |
| resonance peak (fine grid, 0.05 steps) | 3.50 rad/s | 3.65 rad/s | not run |
| amplitude at the peak | 0.855 rad | 0.747 rad | 0.862 rad |

- **Peak shift, mass ×1.5:** ×1.00 on the coarse grid (×1.04 on the fine grid, slightly *up*). Pure spring-mass resonance would predict √(1/1.5) = ×0.82, down.
- **Amplitude ratio at the healthy peak** (ω = 3.5): **×0.87**. The main experiment wasn't sine-driven, so it has no "original drive frequency"; per the spec, the healthy peak stands in.
- **Minimum ratio, mass ×1.5, over all frequencies:** **×0.65** (ω = 0.5). The fine grid (2–6 rad/s) ranges from ×0.67 to ×0.92.
- **Minimum ratio, length ×1.35 (extra):** ×0.48 (ω = 5).
- **Main experiment:** ×0.30 (median motion 0.0501 → 0.0150).
- **Robustness:**
  - Five random start states give the same amplitudes to 3–4 decimals, so no multiple steady states were found.
  - The arm never spun over the top; it swung around the bottom (mean angle 3.09–3.20 rad ≈ π) in every run.

## Interpretation

**The minimum ratio for the mass change is ×0.65, well above 0.35. By the rule set beforehand, something beyond resonance is contributing to the main experiment's ×0.30 drop.** Even the length change, which detunes more, bottoms out at ×0.48.

**Why mass barely detunes this body.** Swinging around the bottom, gravity's restoring torque grows with mass exactly as fast as the arm's inertia does, so the gravity part of the natural frequency doesn't depend on mass at all. Only the soft spring's share shifts, and it's small here: the peak doesn't move down at all. What a heavier arm does do is respond less to the same torque, about 1/1.5 ≈ ×0.67 away from resonance. That's the floor seen at slow and fast drives. Neither effect gets anywhere near ×0.30.

**This agrees with the simplified playground** (`experiments/resonance-check/`, best case about ×0.45 at k = 4), and is even further from ×0.30 in the actual env.

## What this does and doesn't show

- **Shows:** in my exact compliant body (same spring, hysteresis, motor, damping), passive resonance detuning from a 1.5× mass change can't shrink the motion to ×0.30 at any drive frequency from 0.1 to 10 rad/s.
- **Doesn't show:** what produced ×0.30. The main experiment balanced the arm **upright with a controller**, and this test can't reproduce that regime: 2 N·m can't hold it up open-loop, and upright the soft joint has no passive resonance. The likely remaining explanation is the **closed loop**: the controller's response to a heavier body, combined with the spring. That's a property of controller plus body, not resonance of the body alone.
- **Suggested next test:** run the main experiment's own planner on the soft body, healthy vs mass ×1.5, and measure the frequency content and size of the motion. That pins down whether the controller (for example, a lower effective gain or a different oscillation frequency) produces the quieter motion.
