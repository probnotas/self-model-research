# Fixes 1 and 2: stricter gain threshold, plus active self-test after a fall. Pre-registration

*Written and committed before the fresh trials exist. In simulation only, not hardware. Fix 3 (hardware) can't run here; see `HARDWARE_PLAN.md`.*

## What changed from `experiments/gated_gain/` (which failed its safety criterion by one trial)

**Fix 1, false alarms.**
- **Threshold:** the gain threshold rises from |z| ≥ 2 to **|z| ≥ 2.5**.
- **Calibration:** the healthy calibration set grows from 50 to **150** episodes.
- **How it was chosen:** on DEV data only (the already-used seed-777000 trials, `dev_threshold.py`). There, |z| ≥ 3 would have cut soft/low detection to 78/120, while 2.5 kept 102/120 and cut medium false alarms from 7 to 2.

**Fix 2, a fall no longer means "can't tell".**
- **Trigger:** if the robot has fallen, it runs an active self-test (`sim_probe.py`).
- **Sequence:**
  1. The controller switches off at onset+200.
  2. 2 s of zero command (settle).
  3. 3 s of a known command, u = 1.0·sin(2π·1.5 Hz·t) N·m.
- **Measurement:** the response is the lock-in amplitude of θ̇ at 1.5 Hz over the last 2 s of the probe, compared (as a z of the log) with **50 healthy hanging-calibration episodes**.
- **How it was chosen:** on a DEV batch (seed 555000, `dev_probe.py`). Fallen pushes gave z ≈ 0, fallen body changes gave z −1.6 to −196, and healthy hanging gave −2.0 to +0.

## Fresh data

- **Master seed:** **999000**, never used before. Trial seeds run from 999000 to 999629 (480 factorial trials plus **150** calibration episodes); the hanging-calibration seeds are 1999000 .. 1999049.
- **Design:** the same factorial (4 × 120) and the same draw ranges as every earlier run.
- **Configurations:** `soft_none`, `soft_low`, `rigid_none` and `medium_none`.
- **Held fixed:** physics, original single-state models, controller and episode loop up to onset+200.
- **Inverse model:** unchanged from `experiments/inverse/`, trained on the healthy seed-90000 recordings, with model seeds 0, 1 and 2.

## The rule (per trial)

1. **FELL?** max |θ| over onset+180 .. onset+199 ≥ π/2.
2. **If upright:**
   - **Gain:** g = Σ(â·a) / Σ(a²) over windows ending at t in onset+100 .. onset+189. Windows end by onset+199, so none reaches the controller-off phase.
   - **Score:** z_gain against the 150 fresh calibration episodes.
   - **Verdict:** BODY if |z_gain| ≥ 2.5, otherwise WORLD.
3. **If fell:**
   - **Score:** z_probe of the log probe amplitude, against the 50 hanging-calibration episodes.
   - **Verdict:** BODY if |z_probe| ≥ 3, otherwise WORLD.
4. **Continuous score for AUC:** |z_gain| for upright trials, |z_probe| for fallen ones.

## Success criteria (every one must hold for EACH of the 3 model seeds, on ALL FOUR configurations)

- **S (safety):** push-only called BODY ≤ 10/120, and undisturbed called BODY ≤ 10/120.
- **D (detection):** body-only called BODY ≥ 80/120.
- **A (attribution):** AUC of the continuous score, body-only vs push-only, ≥ 0.90.

**Outcome:** PASS only if S, D and A all hold. Otherwise, report exactly what failed, where, and on which seed.

**Known risk, stated in advance:** on the DEV data, rigid pushes were flagged 12–14/120 at every threshold, against 2–6 in `experiments/gated_gain`. Rigid S may fail.

## Also reported

- **Verdict counts** for each cell, configuration and seed, including body + push (masking).
- **The FELL count** per cell.
- **The z_probe distribution** for fallen push-only and fallen body trials.
- **Comparison:** the same numbers under the old rule (|z| ≥ 2, no probe, fall = no judgment).
