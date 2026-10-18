# Fell / body / world: a gated command-gain detector, pre-registration

*Written and committed before the fresh trials exist. In simulation only, not hardware.*

## Why

- **Rung 6** ("quieter = body") replicated on the soft and rigid joints, but scored 0.017 on the medium joint.
- **Inverse-dynamics command gain** (`experiments/inverse/`) detects body changes on soft very well (AUC 0.987–0.995). It can't separate them from pushes, because about half of soft pushes knock the pendulum over for good.
- **A design check on the old exploratory data (seed 90000) changed the picture:**
  - on the medium joint, every body change makes the pendulum fall (0/120 upright at 9–10 s);
  - so medium's "louder" was a fall, not a body-specific direction.
- **Hence a three-way verdict:** first ask whether the robot is still upright, and only then judge body vs world, using a physically motivated signal: how strongly the body responds to my commands.

## Fresh data

- **Design:** the same factorial as every earlier run: 4 cells × 120 trials plus 50 calibration episodes.
- **Master seed:** **888000**, never used before. Trial seeds run from 888000 to 888529.
- **Configurations:** `soft_none`, `soft_low`, `rigid_none` and `medium_none`.
- **Held fixed:** the physics, the original single-state models and the controller, all unchanged. Episodes are recorded as in `surprise_attribution/run_history.record_episode`.

## Model (unchanged from `experiments/inverse/`)

- **Inverse model:** endpoint inverse dynamics. The input is states s_{t−9} .. s_{t+10} ([cos θ, sin θ, θ̇], 60 numbers); the output is actions a_t .. a_{t+9}.
- **Network and training:** MLP 60→64→64→10, MSE loss, Adam lr 1e-3, batch 512, 40 epochs.
- **Training data:** the healthy seed-90000 recordings per configuration (pre-onset steps of all trials plus the calibration episodes).
- **Seeds:** model seeds 0, 1 and 2.

## The rule (per trial)

**Window:** late, t in onset+100 .. onset+199 (5–10 s after onset).

**Step 1, FELL:**
- **Test:** max |θ| over the last second of the window (onset+180 .. onset+199) ≥ π/2. In plain terms, the pendulum is below horizontal at some point in that second.
- **Verdict:** FELL. No body/world judgment is made.

**Step 2, otherwise:**
- **Gain:** g = Σ(â·a) / Σ(a²) over the window.
- **Score:** z = (log g − mean) / sd, against this configuration's 50 fresh calibration episodes.
- **Verdict:** BODY if |z| ≥ 2; otherwise WORLD (no persistent change).

## Success criteria (every one must hold for EACH of the 3 model seeds)

**S (safety), on all four configurations:**
- push-only trials called BODY ≤ 10/120, and
- undisturbed trials called BODY ≤ 10/120.

**D (detection), on `soft_none`, `soft_low` and `rigid_none`:** body-only trials called BODY ≥ 80/120.

**A (attribution among upright trials), on `soft_none`, `soft_low` and `rigid_none`:**
- **Score:** |z|.
- **Comparison:** body-only vs push-only, restricted to trials not called FELL.
- **Threshold:** AUC ≥ 0.90.

**Medium joint, stated in advance:** body changes are expected to be FELL, as in the exploratory data. D and A are not applied there, and the rule is not claimed to attribute body changes that make the robot fall.

**Outcome:**
- **Pass:** S, D and A all hold.
- **Otherwise:** report exactly which criteria failed, on which body and seed.

## Also reported

- **The three-way verdict counts** for every cell (neither, body only, push only, body + push) and configuration.
- **AUC |z| body-only vs undisturbed.**
- **Median gain per cell.**
- **Motion correlation** within undisturbed trials.
- **Context:** the Rung 6 rule B evaluated the same way on these fresh trials.

Every number is reported as the mean and the min–max over the 3 model seeds.
