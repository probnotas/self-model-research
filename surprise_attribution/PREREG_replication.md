# Pre-registration: replication of the behaviour-change score on fresh trials

*Written and committed before any fresh trial exists. Nothing below may change after the fresh run starts. Any analysis not listed here is reported as exploratory.*

*In simulation only. The compliant body is simulated, not hardware.*

## Why this test exists

`results_twosided/` (exploratory, master seed 90000) found that on the soft body a body change makes the pendulum move *less* while balancing. A two-sided score on that motion separated body changes from undisturbed runs: AUC 0.912 (soft/none) and 0.851 (soft/low). By then the same 6,360 trials had been analysed about seven ways, so that result is a hypothesis. This test checks it on trials nobody has looked at.

## Fresh data

**Design:** the same factorial as `run_factorial.py` — 4 cells × 120 trials plus 50 calibration episodes, with the same draw ranges, push sizes and durations.

**Master seed:** **777000**. It has never been used in this repo, and the trial seeds run from 777000 to 777529, with no overlap with earlier experiments.

**Configurations:**
- **Targets:** `soft_none` (k = 4, no noise) and `soft_low` (k = 4, low noise).
- **Controls, reported but not part of the success test:** `rigid_none` and `medium_none`.

**Held fixed, unchanged from before:**
- **Physics:** `compliant_env.py` and `compliant_config.py` parameters.
- **Forward models and controller:** `forward_model.pt` for rigid; the cached `compliant_models/<config>.pt` otherwise. No retraining.
- **Episodes:** the same loop as `run_history.record_episode`; observations and actions are recorded.

## The signal and the score (identical to `run_twosided.py`)

1. **Motion:** m_t = ‖s_{t+1} − s_t‖ on the observed state.
2. **Window:** steps onset+20 to onset+100 (1–5 s after onset). Every push is over by then.
3. **Per trial:** x = log(mean motion in the window).
4. **Standardise:** z = (x − mean) / sd, where mean and sd come from the 50 **fresh** calibration episodes of the same configuration.

## The rules

**Rule A, two-sided motion.**
- Verdict: body if |z| ≥ 2, otherwise world.
- AUC score: |z|.

**Rule B, signed motion (quieter = body).**
- Verdict: body if z ≤ −2, otherwise world.
- AUC score: −z.
- Evaluated on soft and rigid only. On medium stiffness the exploratory data showed body changes making the pendulum *louder*, so rule B is not expected to apply there. Medium is reported for rule A only, and rule B's medium numbers are labelled "not applicable, reported for completeness".

**Rule C, self-model error, two-sided.** The same as rule A, but on the original model's one-step error instead of motion. It's reported so we know whether the self-model contributes. It's not part of the success test.

## Metrics, per configuration and rule

- AUC body-only vs undisturbed.
- AUC body-only vs push-only.
- Body changes detected (body-only and body+push).
- False positives (push-only called body).
- Masking (body+push called world).
- Undisturbed runs flagged.

Each rate gets a 95% Wilson interval.

## Success criteria (fixed now)

The criteria must hold on **both** `soft_none` and `soft_low`:

- **S1 (rule A):** AUC body-only vs undisturbed ≥ 0.75.
- **S2 (rule B):**
  - AUC body-only vs push-only ≥ 0.75, and
  - false positives ≤ 15 of 120.

**Outcome:**
- **Replicated:** S1 and S2 hold on both soft configurations.
- **Partially replicated:** S1 and S2 both hold on only one of them, or only S1 holds on both.
- **Failed:** anything else.

Whatever the outcome, every number is reported. No change to the window, threshold, signal, configurations or criteria after the fresh run starts. The fresh trials are scored once.

## Known limitations, stated up front

- **One controller and one model per configuration.**
- **The signal doesn't come from the self-model.** Rules A and B use raw motion, so a pass would show that *behaviour change* is detectable on the soft body, not that the self-model's surprise detects it. Rule C measures the self-model's own contribution.
- **Rule B's direction was suggested by the exploratory data.** That's exactly why it is tested here, on fresh trials, rather than claimed.
