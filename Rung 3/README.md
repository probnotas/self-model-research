# Rung 3 — ensembles and meta-learning

Published write-up: `rung3.html`

    python3 run_rung3.py      # the full driver (conditions a-g, sweeps, generalisation)
    python3 stats_rung3.py    # the significance arithmetic the original tables lacked

## What was tried

**Stage 1.** An ensemble of 5 forward models (different inits + bootstrap resampling)
with uncertainty-aware CEM: the task cost is scored against the ensemble mean and
`lambda x disagreement` is added as a penalty, following the PETS intuition.
Lambda swept over [0, 0.1, 1.0, 10.0], with lambda=0 as the control isolating
"ensemble alone".

**Stage 2.** GrBAL-style meta-learning (Nagabandi et al. 2018): mass, length,
damping and torque scaling randomised per episode, a MAML inner/outer loop, K=10
and K=25, online sliding-window adaptation at control time.

## The outcome

All conditions, 15 runs on the damaged body (mass +50%), lower is better:

| | condition | cost ± sem |
|---|---|---|
| a | single model, stale, no adaptation | 6.48 ± 0.51 |
| b | single + naive fine-tune (100 samples) | **5.56 ± 0.63** |
| c | ensemble, λ=0 | 6.42 ± 0.76 |
| d | ensemble + uncertainty, λ=0.1 | 6.05 ± 0.66 |
| e | meta + online adapt, K=25 | 6.41 ± 0.50 |
| f | meta + online adapt + uncertainty | 6.15 ± 0.74 |
| g | model-free policy, no adaptation (Rung 2) | 3.00 |

**0 of 15 pairwise comparisons separate** at |z| > 1.96. The best model-based
condition is plain fine-tuning, and it sits 1.14 standard errors from doing
nothing at all.

The uncertainty penalty buys about 0.05 at λ=0.1 and costs +2.5 to +2.7 at λ=10.

## The one significant effect

Meta-learning paid off only on damage **never seen in meta-training**:

| length +50% | cost ± sem |
|---|---|
| b) naive fine-tune | 6.57 ± 0.53 |
| f) meta + uncertainty | **4.93 ± 0.41** |
| a) no adaptation | 5.44 ± 0.25 |

f vs b: **z = −2.44, separates**. The only comparison in Rung 3 that clears the
noise floor. On added damping, nothing separates (z = −0.63).

There is a hint that naive fine-tuning is *worse* than not adapting at all on
unfamiliar damage (b vs a, z = +1.93), consistent with overfitting to a body it
has barely met — but at 15 runs that is suggestive, not established.

## Why it stalled

Measurement, not method. Identical configurations moved by 1.3 between the
validation and test seed sets — larger than every effect being measured. At 15
evaluation episodes the noise floor sits above the signal, so most of these
conditions are untestable rather than equal.

The fix needs no new code: ~60 evaluation episodes, about 4x the compute. Rung 4
was designed at 20 runs per condition with effects an order of magnitude larger
than their spread, and its numbers resolve cleanly.
