# Inverse-dynamics mismatch on the soft pendulum: pre-registration

*Written and committed before any run. In simulation only, not hardware.*

## Idea

The idea is "if the error follows my commands, it's me". An inverse-dynamics model trained on the healthy body guesses which commands produced the observed motion. Two scores come from it:
- **Mismatch:** guessed vs commanded actions. This is the change detector that comes free with inverse-dynamics methods.
- **Gain:** how strongly the guessed actions scale with the commanded ones. A body change should change the response *to my commands*. A push adds motion that isn't tied to my commands.

The model mirrors the endpoint inverse-dynamics loss (EP-IDM) of Toso, LeCun, Anderson and Bounou, *"Preserving Unstable Modes Through Inverse Dynamics in JEPA World Models"* (jepa-control.github.io): reconstruct the action sequence over a horizon from the states at its two ends. Here it works on raw states, not pixels, because this is our pendulum, not their CartPole.

## Data (fixed; the same as the JEPA follow-up)

- **Training:** healthy data only, from the seed-90000 recordings (`surprise_attribution/results_history/episodes/`): the pre-onset steps of all 530 trials, plus the 50 calibration episodes in full.
- **Testing:** the seed-777000 trials (`surprise_attribution/results_replication/episodes/`). These trials have already been scored by earlier analyses (Rung 6 and the JEPA runs), so they aren't untouched.
- **Configurations:**
  - **Targets:** `soft_none` and `soft_low`.
  - **Control:** `rigid_none`, reported but not part of any criterion.

## Model

**Inverse model.**
- **Input:** the 10 observed states up to and including s_t (context, so hidden spring state can be inferred), plus the 10 observed states s_{t+1} .. s_{t+10}. That's 20 states × [cos θ, sin θ, θ̇] = 60 numbers.
- **Output:** the 10 actions a_t .. a_{t+9}.
- **No past actions as input.** Commands are autocorrelated, so past actions would let the model copy them instead of reading the dynamics.
- **Training:** MLP 60→64→64→10, MSE loss, Adam lr 1e-3, batch 512, 40 epochs.

**Seeds:** 0, 1 and 2, for initialisation and batch order.

## Scores (per trial, from windows ending at t)

- **Windows:** early = t in onset+20 .. onset+99 (1–5 s); late = t in onset+100 .. onset+199 (5–10 s).
- **M (mismatch):** the mean |â − a| over the 10 actions and over the window.
- **G (gain):** g = Σ(â·a) / Σ(a²) over the window; the score is |z| of log g against the 50 fresh calibration episodes (two-sided, since the direction isn't known in advance).
- **Threshold for verdicts:**
  - M: body if score ≥ the calibration mean + 2 sd of the same score;
  - G: body if |z| ≥ 2.

## Success criteria (EARLY window; each must hold for EACH of the 3 seeds, on BOTH soft configurations)

- **M1 (mismatch detects):** M's AUC body-only vs undisturbed ≥ 0.75.
- **M2 (mismatch attributes):** M's AUC body-only vs push-only ≥ 0.75, and push false positives ≤ 15/120.
- **G (gain attributes):**
  - G's AUC body-only vs undisturbed ≥ 0.75;
  - AUC body-only vs push-only ≥ 0.75; and
  - push false positives ≤ 15/120.

## Also reported (no pass/fail)

- **Late-window versions** of M and G.
- **The signed direction of log g** after body changes and after pushes.
- **Push false positives split by recovered vs not recovered,** using the same recovery definition as `experiments/jepa`.
- **Correlation with motion** (over undisturbed + body-only, and within undisturbed only).
- **Context, on the same trials:** raw one-step error, the 10-step raw-window model (`experiments/jepa`), and Rung 6 rule B.

Every number is reported as the mean and the min–max over the 3 seeds.
