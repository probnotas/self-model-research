# JEPA quiet-body follow-up: pre-registration

*Written and committed before any of the runs below. In simulation only, not hardware.*

## Background

A first exploratory JEPA run (`exploratory_run1.py`, `exploratory_run1_results.json`, one seed) found, on the soft joint:
- **Body changes:** JEPA summary-prediction error *rises* (AUC body-only vs undisturbed 0.992 and 1.000).
- **Pushes:** the error rises far more, so body vs push separation fails (AUC 0.48).

This follow-up tests whether persistence (a late window) separates them, and whether the effect comes from the latent space or just from the 10-step window.

## Data (all fixed)

- **Training:** healthy data only, from the seed-90000 recordings (`surprise_attribution/results_history/episodes/`): the pre-onset steps of all 530 trials, plus the 50 calibration episodes in full.
- **Testing:** the seed-777000 trials (`surprise_attribution/results_replication/episodes/`).
- **Note:** these test trials are *not* untouched. The exploratory run already scored their early window.
- **Configurations:**
  - **Targets:** `soft_none` and `soft_low`.
  - **Control:** `rigid_none`, reported but not part of any criterion.

## Models (unchanged from the exploratory run, except the seed)

**JEPA.**
- **Encoder:** MLP 40→64→64→8, taking the last 10 steps of [cos θ, sin θ, θ̇, u].
- **Target encoder:** an EMA copy (τ 0.99), with stop-gradient.
- **Predictor:** MLP (8 + 10 commands)→64→64→8. It predicts the target-encoder summary of the next 10-step window.
- **Training:** MSE loss, Adam lr 1e-3, batch 512, 40 epochs.

**Raw-space ablation.**
- **Network:** the same input (the last 10 steps, 40 numbers) plus the next 10 commands, through an MLP →64→64→30.
- **Output:** the next 10 raw states [cos θ, sin θ, θ̇] directly.
- **Training:** the same optimiser, batch size and epochs.
- **Error:** the mean squared error over the 30 outputs.

**Seeds:** 0, 1 and 2, for torch initialisation and batch order. Each seed trains both models for each configuration.

**Collapse check (JEPA).** Measure the mean per-dimension standard deviation of target summaries on 2,000 held-out healthy windows. A run is INVALID if, after training, that spread is below 0.05× its value at initialisation. Invalid runs are reported and count as failures; they are not rerun.

## Scores

**Per step:** for a context window ending at step t, the error is computed on the next window (t+1 .. t+10).

**Per trial:**
- **Early:** the mean over t in onset+20 .. onset+99 (1–5 s after onset).
- **Late:** the mean over t in onset+100 .. onset+199 (5–10 s after onset).
- **Ratio:** late ÷ early (descriptive only).

**Threshold for verdicts:** body if score ≥ the mean + 2 sd of the same score on the 50 fresh calibration episodes.

## Success criteria (every one must hold for EACH of the 3 seeds)

**P (persistence works).**
- **Score:** the JEPA *late-window* score.
- **Must hold on both soft configurations:**
  - AUC body-only vs push-only ≥ 0.90, and
  - push false positives ≤ 10/120.

**L (it's the latent space, not the window).**
- **Must hold on both soft configurations:** the ablation's early-window error does *not* rise after body changes, or rises far less.
- **"Does not rise":** ablation AUC (body-only > undisturbed) < 0.60.
- **"Rises far less":** (ablation median ratio body-only ÷ undisturbed − 1) ≤ 0.25 × (JEPA median ratio body-only ÷ undisturbed − 1).
- Either condition is enough.

## Also reported (no pass/fail)

- **For both models, every seed, both windows:**
  - AUC body vs undisturbed and body vs push;
  - body-only caught and push false positives;
  - the correlation with motion (over undisturbed + body-only, and within undisturbed only).
- **The late÷early ratio:** body vs push AUC.
- **Push recovery:** for each push-only trial, whether the pendulum returns to balancing and when. Balancing means calm, |θ| < 0.3 and |θ̇| < 0.5, for more than 10 consecutive steps after the push ends, as in the repo's settle definition.
- **Breakdowns:** JEPA late-window false positives split by push size (1, 2, 4 N·m), by duration (0.2 s, 0.5 s), and by recovered vs not.
- **Context:** the raw one-step error and the Rung 6 rule-B numbers for the same trials.

Every number is reported as the mean and the min–max across the 3 seeds.
