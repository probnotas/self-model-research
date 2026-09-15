# Rung 4 — body change vs world event

Both look identical at the instant they happen: the forward model's prediction
stops matching reality. This rung asks whether the *shape* of that error tells
you which one it was.

    python3 world.py            # the disturbance-capable pendulum (verifies equivalence)
    python3 identifiability.py  # which body changes are even separable in principle
    python3 rung4.py calibrate  # size the push against the body change
    python3 rung4.py run        # 7 conditions x 20 seeds
    python3 analyse_rung4.py    # detect-then-classify pipeline
    python3 summarise.py        # freeze the operating point -> summary_rung4.json

## Setup

A forward model trained on 2,000 random-action transitions from the healthy body,
frozen, driving CEM balance-hold. Prediction error is logged every step. At step
80 one of two things happens:

- **body change** — a physical parameter is permanently altered
- **world event** — an external torque is added for 5 or 10 steps, then removed

The external torque is applied after the actuator gain and is never part of the
commanded action, so the model cannot predict it.

## The control that makes this non-trivial

The world push is calibrated so its error spike **matches or exceeds** the body
change's. Without that, "body vs world" degenerates into "big vs small".

| condition | peak error | error at +60 steps |
|---|---|---|
| baseline | 5.48e-05 | 4.44e-05 |
| body: mass doubled | 2.11e-03 | **3.65e-03** |
| body: length +50% | 8.01e-04 | **1.19e-03** |
| body: damping added | 5.28e-05 | 4.49e-05 |
| world: push (peak-matched) | 2.00e-03 | 4.72e-05 |
| world: push (bigger spike) | **3.52e-03** | 4.93e-05 |
| world: long push (10 steps) | 3.70e-03 | 4.61e-05 |

The bigger-spike push peaks **higher than any body change** and still returns to
baseline. Peak size carries no information; persistence carries all of it.

## The classifier

Deliberately dumb, and no learned component:

1. **detect** — 10-step trailing mean of prediction error crosses a threshold
   calibrated on undisturbed runs only (leave-one-out, mean + 4 sigma)
2. **classify** — 30 steps (1.5 s) after the alarm, is the error *still* above
   the threshold? Yes: body. No: world.

| | result |
|---|---|
| false alarms on undisturbed runs | **0 / 20** |
| world events called "world" | **60 / 60** |
| detected body changes called "body" | **40 / 42** |
| accuracy on detected events | **98%** |
| time to a verdict | 1.5 s after the alarm |

Lag matters: at 0.5 s accuracy is 34% (too early — the world event has not
subsided yet), at 1.0 s it is 82%, and it saturates at 1.5 s.

## What did not work

**Directionality is not a discriminator.** The hypothesis was that body-change
error would be one-directional. Residual coherence (norm of the mean residual /
mean residual norm) says otherwise:

| condition | during | later |
|---|---|---|
| body: mass doubled | 0.94 | 0.13 |
| body: length +50% | 0.64 | 0.91 |
| world: push | 0.82 | 0.56 |

Body changes do not share a coherence signature — mass drops to 0.13 while
length rises to 0.91. Persistence works; direction does not.

**A mild body change is invisible.** Added damping (0.5) produces 5.28e-05 peak
error against a 5.48e-05 baseline — below the detection floor. It is detected in
10% of runs and misclassified in both. This is a detection limit, not a
classification one, and it sets the sensitivity floor of the whole approach.

Detector sensitivity vs false alarms:

| threshold | false alarms | mass | length | damping |
|---|---|---|---|---|
| mean + 3σ | 20% | 100% | 100% | 15% |
| **mean + 4σ** | **0%** | **100%** | **100%** | 10% |
| mean + 5σ | 0% | 100% | 100% | 0% |

## Identifiability — a harder limit than measurement

`identifiability.py`. The pendulum's angular acceleration is

    thdd = 3g/(2l) sin(th) + 3/(m l^2) (u * torque_scale)

so `m` and `torque_scale` enter **only** through the ratio `torque_scale / m`.
Doubling the mass and halving the actuator gain are bit-identical — max state
deviation 0.000e+00 over 500 shared commands.

No forward model, no amount of data, and no better algorithm can separate "I got
heavier" from "my motor got weaker" on this body. They are the same event.

So the capability ladder has three rungs, not two:

1. **that** something changed — Rung 2, 0.13 s
2. **whether** it was body or world — Rung 4, 98% in 1.5 s
3. **which parameter** changed — partly impossible in principle, not just hard
