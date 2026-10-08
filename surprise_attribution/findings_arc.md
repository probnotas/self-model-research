# An arc of elimination: why surprise attribution fails on a compliant body

*In simulation only. The compliant body is a simulated series-elastic joint, not hardware.*

This document tells the story of five experiment runs as one argument. Each run tested one explanation for a failure, ruled it out or narrowed it, and handed a sharper question to the next. Nothing here is a working detector for a compliant body. What the arc does produce is a narrower and better-supported statement of *why* the detector fails.

Every number below is copied from a results file in `surprise_attribution/`. A few numbers were computed directly from the per-trial CSVs rather than taken from a summary; those are marked *(from the per-trial CSV)*. Anything not logged is marked as such.

| step | experiment | results folder |
|---|---|---|
| 1 | Factorial, rigid body, simultaneous disturbances | `results_factorial/` |
| 2 | The same factorial on a compliant, noisy grid | `results_compliant/` |
| 3 | Per-body calibration (z-scoring) | `results_calibrated/` |
| 4a | History-input forward model | `results_history/` |
| 4b | Shape feature, no settle gate | `results_shape/` |

---

## 1. The question

A robot with a learned self-model (a forward model that predicts its own next state) gets surprised whenever a prediction fails. Two very different causes produce that surprise:

- **A body change:** mass or length changed, permanently. The self-model is now wrong and should adapt.
- **A world event:** something pushed it, briefly. The self-model is still right and should not adapt.

Adapting to a push corrupts a correct model. Failing to adapt to a body change leaves a stale one. So the question is:

> Can the self-model's own one-step prediction error tell a body change from a world event, and does that still work on a compliant, noisy body?

**Setup shared by every step:**

- **Body and task.** Gymnasium `Pendulum-v1`, balancing upright, driven by a CEM model-predictive controller that plans through a learned forward model.
- **Forward model.** An MLP trained on 2,000 random-action transitions. It predicts the next observation `[cos θ, sin θ, θ̇]` from the current one plus the torque.
- **Surprise.** The one-step prediction error ‖s′ − f(s, a)‖.
- **The original rule** (`alarm_classifier.decide`). Wait for the pendulum to *settle* (calm and near upright), then average the error over 2 s.
  - Above a body/world threshold → **body**; otherwise → **world**.
  - If it never settles within 10 s, check whether the error is *persistently* above an alarm threshold → **alarm**; otherwise → **world**.
- **The trials.** A 2×2 factorial with 120 trials per cell plus 50 undisturbed calibration episodes, all derived from master seed 90000. The four cells are: neither, body change only, push only, body change + push at the same moment.
  - **Body changes:** mass ×1.3–1.7 or length ×1.2–1.5.
  - **Pushes:** 1, 2 or 4 N·m, lasting 0.2 s or 0.5 s.

**Two failure rates matter most:**

- **False positive:** a push alone called "body". This is the safety-critical error, because the robot rewrites a correct model.
- **Masking:** a body change hidden by a simultaneous push and called "world".

---

## 2. Experiment 1: rigid body, simultaneous disturbances

**Tested:** whether the rule still works when a body change and a push land at the same moment, unannounced, on the rigid pendulum.

**Found** (`results_factorial/summary.txt`):

| ground truth | → body | → world | → alarm |
|---|---|---|---|
| neither | 0 | 120 | 0 |
| body only | 117 | 0 | 3 |
| push only | 0 | 120 | 0 |
| body + push | 70 | **19** | 31 |

- **False positives: 0/120 (95% CI 0.0–3.1%).** A push alone was never mistaken for a body change.
- **Masking: 19/120 = 15.8% (95% CI 10.4–23.4%).** Overall misattribution was 4.0%.
- **The margin was clean but thin.**
  - Settled scores with a body change: 0.0143–0.0986.
  - Without one: 0.0080–0.0129.
  - Gap +0.0014, against a threshold of 0.0138.

**Mechanism.** All 19 masked trials took the same route:

1. The pendulum never settled within 10 s.
2. The error stayed raised, but in fewer than the required 80% of late 1 s blocks.
3. The rule therefore fell through to its default verdict, "world".

18 of the 19 were length changes. Masking was concentrated in the 0.5 s pushes at 1 and 2 N·m (8/20 and 10/20).

The factorial files don't log the pendulum's angle, so *why* those trials never settled isn't measured here. An earlier ad-hoc look at `results_overlap/` traces suggested the pendulum had fallen, but that wasn't saved as a number.

**Question handed on:** the rigid result holds, with a thin margin and a known failure route through "never settles". Does it hold on a body that isn't rigid?

---

## 3. Experiment 2: the compliant grid

**Tested:** the same trials, rule and seed on a simulated compliant joint, at 4 stiffnesses × 3 sensor-noise levels.

- **Stiffness:** rigid, then k = 200, 30 and 4 N·m/rad.
- **The joint:** series-elastic, with hysteresis and damping. The motor and spring state are hidden from the robot.
- **Sensor noise:** σ = (0, 0), (0.002, 0.02) and (0.01, 0.1) on angle and angular velocity.
- **Recalibration:** a forward model was retrained on each body, and the body/world threshold was recalibrated on each body's own 50 undisturbed episodes. The alarm threshold stayed frozen at 0.04042.

**Found** (`results_compliant/summary.txt`). Rates are out of 120 per cell. The AUC is computed on settled trials only, comparing those with and without a body change.

| config | false positive | masking | missed, no push | gap | AUC | settled, body / no body |
|---|---|---|---|---|---|---|
| rigid, none | 0.0% | 15.8% | 0.0% | +0.0014 | 1.000 | 187 / 239 |
| rigid, low | 0.0% | 55.8% | 38.3% | −0.0167 | 0.930 | 193 / 238 |
| rigid, medium | 12.5% | 60.0% | 100.0% | −0.1084 | **0.291** | 192 / 225 |
| stiff, none | 2.5% | 70.0% | 24.2% | −0.0126 | 0.980 | 134 / 196 |
| stiff, low | 0.8% | 78.3% | 64.2% | −0.0541 | 0.830 | 51 / 185 |
| stiff, medium | 96.7% | 2.5% | 2.5% | −0.0241 | 0.767 | 6 / 5 |
| medium, none | 4.2% | 97.5% | 100.0% | −0.0271 | 0.955 | 7 / 185 |
| medium, low | 15.0% | 100.0% | 100.0% | −0.0281 | **0.373** | 13 / 192 |
| medium, medium | 100.0% | 0.8% | 0.0% | n/a | n/a | 1 / 0 |
| soft, none | 21.7% | 100.0% | 100.0% | −0.0401 | **0.202** | 172 / 177 |
| soft, low | 43.3% | 99.2% | 100.0% | −0.0410 | **0.286** | 166 / 180 |
| soft, medium | 100.0% | 0.0% | 0.0% | n/a | n/a | 0 / 0 |

- **The gap turns negative** in every configuration where it can be computed, except rigid with no noise.
- **Four configurations invert outright** (AUC below 0.5): trials *without* a body change score higher than trials with one.
- **On medium and soft joints every body change is missed even with no push.**
- **At medium noise on compliant bodies the calibration episodes never settle** (stiff 0/50, medium 1/50, soft 0/50, from the per-config JSON). No threshold can be set, and the rule alarms on nearly everything, undisturbed runs included.

**Lesson:** this is not a threshold problem. The body/world threshold was already recalibrated per body by the rule's own procedure, and it still failed. Where the AUC is below 0.5, no threshold placement could fix it.

**Question handed on:** the compliant body's normal error is higher (normal balancing error 0.0107 rigid vs 0.0907 soft, from `findings.md`). Is the failure just a matter of scale?

---

## 4. Experiment 3: per-body calibration

**Tested:** scoring surprise *relative to each body's own healthy baseline* instead of in absolute terms. Each step's error was z-scored, (error − baseline mean) / baseline std, using that body's undisturbed calibration episodes. The same rule then ran on the standardised signal, on the same trials.

- **Thresholds:** carried over from the rigid body in z units (body 0.639, alarm 6.215) and held fixed.
- **Paired comparison:** the regenerated traces reproduced the logged trials exactly, with 0 mismatches in 6,360 trials.

**Found** (`results_calibrated/summary.txt`). Counts are out of 120.

| config | false positive old → new | masking old → new | missed, no push old → new | AUC old = new |
|---|---|---|---|---|
| rigid, low | 0 → 0 | 67 → 100 | 46 → 51 | 0.930 |
| rigid, medium | 15 → 0 | 72 → 120 | 120 → 120 | 0.291 |
| stiff, none | 3 → **12** | 84 → 84 | 29 → 28 | 0.980 |
| medium, low | 18 → 2 | 120 → 120 | 120 → 120 | 0.373 |
| soft, none | 26 → 0 | 120 → 120 | 120 → 120 | 0.202 |
| soft, low | 52 → 0 | 119 → 120 | 120 → 120 | 0.286 |

- **Every AUC is unchanged, as it must be.** Within one body, z-scoring is a fixed shift and scale, which can't reorder scores. An inverted AUC stays inverted.
- **The false positives that disappear aren't a fix.** In z units, the rigid alarm line (6.2 baseline std) is out of reach on bodies with a wider baseline. Outside rigid/none the alarm branch fired 0 times.
- **The rule now answers "world" almost everywhere.** Misattribution on medium, soft and medium-noise bodies sits at 49.8–50.6%, the same as always answering "world". On stiff/none, false positives went *up*, from 3 to 12.

**Lesson:** this is not a scale problem. The scores are in the wrong *order*, not just on the wrong scale.

**Question handed on:** if the order is wrong, maybe the model lacks information. The compliant joint has hidden spring and motor state that one observation can't reveal. Does giving the model that information put the scores in order?

---

## 5. Experiment 4a: a forward model with history

**Tested:** a forward model fed the last k observations and actions (k = 2, 4, 8) instead of one, so it can infer hidden actuator state. Everything else was the same.

- **Training:** same training data, seed, widths, epochs and learning rate.
- **Controller held fixed:** the original single-state model still drives the pendulum, so trajectories are identical and only the surprise signal changes.
- **Paired comparison:** the re-recorded episodes reproduced the old error traces bit for bit, and the logged verdicts exactly.

**Found** (`results_history/summary.txt`). "Held-out" is one-step error on 1,000 fresh random-action transitions.

| config | held-out single → k=4 | AUC single | k=2 | k=4 | k=8 | missed, no push single → k=4 |
|---|---|---|---|---|---|---|
| rigid, none | 0.0162 → 0.0213 | 1.000 | 1.000 | 0.661 | 0.993 | 0 → 60 |
| rigid, medium | 0.1267 → 0.1102 | 0.291 | 0.653 | 0.777 | 0.854 | 120 → 108 |
| stiff, none | 0.0066 → 0.0072 | 0.980 | 1.000 | 0.896 | 0.904 | 29 → 30 |
| medium, none | 0.0775 → **0.0105** | 0.955 | 0.998 | 0.983 | 0.615 | 120 → 118 |
| medium, low | 0.0817 → **0.0264** | **0.373** | 1.000 | **0.994** | 0.998 | 120 → 116 |
| soft, none | 0.0519 → **0.0118** | **0.202** | 0.302 | **0.047** | 0.479 | 120 → 120 |
| soft, low | 0.0578 → **0.0281** | **0.286** | 0.506 | **0.110** | 0.215 | 120 → 120 |

**What history did:**
- **The model got much more accurate on compliant bodies.** Held-out error fell 3–7× on medium and soft joints with no or low noise, and barely moved on rigid and stiff ones. That fits hidden actuator state being real and partly recoverable from a short history.
- **Two of the four inversions were fixed:** medium/low (0.373 → 0.994) and rigid/medium (0.291 → 0.777 at k = 4). The rigid body has no hidden state, so the rigid/medium gain more likely comes from the history window averaging out noise.

**What it didn't do:**
- **The soft body stayed inverted at every k.** soft/none was 0.302, 0.047 and 0.479, even though its model became about 4× more accurate.
- **Detection barely moved anywhere.** Body changes missed with no push stayed at 116–120/120 on medium and soft joints.

**Why detection didn't improve: two separate walls.**

- **Medium stiffness: the settle gate.** Body-change trials almost never settle (7/240 on medium/none, 13/240 on medium/low), so the settled score, however good, is computed for almost none of them. The rest go to the persistence branch, where error is low while the pendulum is moving, and they fall through to "world".
- **Soft body: not the gate.** Its trials do settle (172/240 body-change trials on soft/none). The wall there is inversion. Inside the settled regime, trials without a body change score higher than trials with one:
  - median settled score 0.0901 without a body change vs 0.0837 with one (single-state);
  - 0.1402 vs 0.1291 (k = 4);
  - *(from the per-trial CSV)*.
- **The settled regime is where the model is worst.** Balancing calm and upright is the regime the gate trusts, but on compliant bodies it's where error is highest. For example, on soft/none without a body change, the median settled error is 0.0901 against a median late-window error of 0.0444 in trials that never settled *(from the per-trial CSV)*. The history model raises settled-regime error further on every body, even while it lowers held-out error.

**Lesson:** this is not mainly a model-information problem. Better information fixed the model and two inversions, but not the soft body, and not detection. On medium stiffness the settle mechanism blocks detection. On soft, something in the settled error itself orders the trials the wrong way.

**Question handed on:** what if attribution doesn't wait to settle at all?

---

## 6. Experiment 4b: the shape feature, with no settle gate

**Tested:** replacing the settle and persistence rule with the *shape* of the error after onset. A push should spike and recover; a body change should rise and stay up.

- **The feature:** in a fixed window of W steps from the onset, r = log(mean error in the second half ÷ mean error in the first half).
- **The decision point:** set only on each body's 50 healthy calibration episodes. "Recovered" means r is more than 2 standard deviations below its healthy mean, and is called world; anything else is called body.
- **No settle gate:** every trial gets a score.
- **A secondary rule, fixed in advance ("+elev"),** also requires the late error to be at least 2 standard deviations above its healthy level.
- **Paired comparison:** the same recorded traces, re-scored only; all 6,360 reproduce the logged settled scores.

**Found** (`results_shape/summary.txt`, `summary.json`). Counts: detection is out of 240, false positives out of 120.

| config | AUC body-only vs push-only, W=20 / W=40 | median r at W=20: push only / body only / undisturbed | detected, old → shape+elev (W=20) | false positives, shape+elev (W=20) |
|---|---|---|---|---|
| rigid, none | 1.000 / 1.000 | −2.70 / 0.11 / 0.02 | 221 → 127 | 0 |
| stiff, none | 0.999 / 0.998 | −1.01 / 0.07 / −0.01 | 127 → 140 | 1 |
| medium, none | 0.996 / 1.000 | −0.77 / 0.37 / 0.01 | **3 → 167** | 16 |
| medium, low | 0.985 / 0.996 | −0.64 / 0.15 / 0.03 | **0 → 124** | 40 |
| soft, none | 0.796 / 0.935 | −0.60 / **0.03** / **−0.01** | 0 → **15** | 22 |
| soft, low | 0.847 / 0.990 | −0.56 / **0.05** / **0.04** | 1 → **43** | 50 |

**What removing the gate fixed:**
- **The shape separates pushes from non-pushes on every body.** Pushes recover clearly, with median r between −0.56 and −2.70. Body-only vs push-only AUC at W = 20 ranges from 0.783 (soft/medium) to 1.000 (rigid/none); on soft it's 0.796 at W = 20 and 0.935 at W = 40.
- **The old rule's inversion on soft (0.202, 0.286) is gone** for this comparison.
- **On medium stiffness, where the gate had blocked everything, detection becomes real.** medium/none goes from 3 to 167 of 240.

**What it didn't fix:**
- **The shape can't tell "stays up" from "nothing happened".** Undisturbed runs have r ≈ 0, like a body change, so shape alone flags 113–120 of 120 undisturbed runs as body changes. The "+elev" rule is needed, and it cuts those to 1–10.
- **On the soft body, a body change is almost invisible in the one-step error.** Its median r (0.03, 0.05) is the same as an undisturbed run's (−0.01, 0.04). Its error doesn't rise above normal balancing error either. Once elevation is required, soft detection falls to 15/240 (soft/none) and 43/240 (soft/low) at W = 20, and to 21 and 39 at W = 40.
- **It's worse than the old rule on the rigid body.** Detection drops from 221 to 127, because a simultaneous push's decay dominates the window (masking 19 → 103).
- **Window length matters.** At W = 10, a 0.5 s push is still on during the window's second half, and nearly all of those pushes are called body (54–60 of 60 on most bodies). On soft at W ≥ 20, longer pushes give more false positives than short ones:
  - soft/none: 24/60 vs 13/60 at W = 20, 20 vs 10 at W = 40;
  - soft/low: 33 vs 25 at W = 20, 12 vs 5 at W = 40.

  No slow, gradually fading pushes exist in these trials, so that weak spot is untested.

**Lesson:** the settle gate was a real wall on medium stiffness, and removing it helps there. On the soft body, removing it reveals a lower wall: the body change hardly shows up in the error at all.

---

## 7. The core finding, by elimination

On a simulated soft series-elastic joint, under this controller, the four experiments rule out four explanations for why surprise attribution fails:

| explanation | ruled out by | evidence |
|---|---|---|
| wrong threshold | Exp 2 | already recalibrated per body; AUC < 0.5, so no threshold works |
| wrong scale | Exp 3 | z-scoring leaves every AUC unchanged; soft stays at 0.202 and 0.286 |
| model lacks memory of hidden state | Exp 4a | model 4× more accurate on soft (0.0519 → 0.0118); soft AUC still 0.047–0.479 |
| settle mechanism | Exp 4b | no gate at all; soft body changes still look like undisturbed runs (median r 0.03 vs −0.01) |

**What remains:** on the soft body, a mass or length change of the sizes tested barely changes the one-step prediction error during balancing. It doesn't change its level relative to normal balancing error, or its shape over time. **In this simulation, one-step prediction error is the wrong observable for detecting a body change on a compliant body.** It's a good observable for detecting *pushes* on every body tested, and for detecting body changes on rigid, stiff and (without the settle gate) medium joints.

**How far this claim goes:**
- It covers one-step error, from models trained on random-action data, during closed-loop balancing, under one controller, in simulation.
- It doesn't show that the body change is undetectable by *any* means, or that a real compliant joint behaves this way.
- Several plausible routes remain untested (section 8). A route that isn't eliminated is still open.

---

## 8. Open question: what observable would reveal it?

These are the next hypotheses. None has been tested.

1. **Multi-step (rollout) error.** A changed mass or length may change slow dynamics (period, sag, how the spring settles) more than the next 0.05 s step. Prediction error over a horizon of 0.5–2 s could make a small per-step difference add up.
2. **Model-parameter drift.** Instead of reading error, let a copy of the model adapt online and watch *how far its parameters have to move* to keep fitting. A push needs a brief, reversible move; a body change needs a lasting one.
3. **A different residual.** Compare commanded torque or acceleration against what the model expects for the observed motion, rather than the predicted state. On a compliant joint the spring may absorb a body change in the state while the torque needed to hold position still shifts.
4. **A baseline matched to the regime.** Train the model, or at least its normal-error baseline, on healthy *closed-loop balancing* data instead of random actions. Exp 4a showed that balancing is where the random-action model is worst, so a body change may only stand out against a baseline from that regime.

**A test that would end the arc:** on the soft body, find an observable whose body-only vs undisturbed AUC is clearly above 0.5 while keeping push-only separable. Exp 4b showed that one-step error fails the first half of that test on soft. The second half already passes.

---

## Caveats

1. **Simulated compliance, not hardware.** The joint parameters (`J_MOTOR` 0.05, `B_MOTOR` 0.5, `B_JOINT` 0.3, `HYST` 0.3, `SPRING_ZETA` 0.3, `N_SUBSTEPS` 10) were chosen, not measured from a device. The spring damper was added after a balance-only pilot whose numbers aren't logged.
2. **One training seed per model.** In Exp 4a the AUC swings between k values (for example rigid/none 1.000 → 0.661 → 0.993). These may be model-to-model noise; with one seed per k it can't be told apart from a real effect of k.
3. **Controller held fixed.** Every experiment after Exp 2 used the single-state model's controller, so that trajectories stay paired. A different controller would produce different trajectories and settling behaviour. That hasn't been tested.
4. **Medium-noise compliant configurations are degenerate.** Calibration episodes never settle (stiff 0/50, medium 1/50, soft 0/50), so the settle-based rule has no threshold. The rigid medium-noise body did calibrate (50/50).
5. **Rigid-tuned settings.** The calm limits, settle windows and alarm threshold were tuned on the rigid body and never re-tuned. Exp 3 changed the threshold units; Exp 4b removed the gate.
6. **One run per configuration, fixed seed.** The confidence intervals cover trial-to-trial variation, not reruns.
7. **Test conditions are narrow.** Body changes were mass ×1.3–1.7 and length ×1.2–1.5. Pushes were square pulses of 0.2 s or 0.5 s; no slow, fading pushes were tested.
8. **Some mechanism statements are interpretations.** These include: hidden actuator state explains the held-out gain; the rigid/medium gain comes from averaging noise; the history model partly adapts to the body change from its input. Each is labelled as an interpretation where it appears.

---

## Reproducibility

**Shared by all experiments:**
- **Versions:** Python 3.11.15, numpy 2.4.6, torch 2.14.0+cpu, gymnasium 1.3.0, logged in each folder's `versions_and_seed.json`.
- **Seeds:** master seed 90000; forward-model training seed 0.
- **Paired trials:** every experiment after Exp 1 re-scores or reproduces the same 530 trials per configuration. Each run checked itself against the logged results with 0 mismatches.

| experiment | code (in `surprise_attribution/`) | results |
|---|---|---|
| 1 Factorial, rigid | `run_factorial.py`, `factorial_config.py` | `results_factorial/` (`summary.txt`, `trials.csv`) |
| 2 Compliant grid | `run_compliant.py`, `compliant_config.py`, `compliant_env.py`, `compliant_trials.py` | `results_compliant/` (`summary.txt/json`, `trials_<config>.csv`, `sweep.png`); models in `compliant_models/` |
| 3 Calibration | `run_calibrated.py`, `calibrated_config.py` | `results_calibrated/` (`summary.txt/json`, `trials_<config>.csv`, `sweep_calibrated.png`) |
| 4a History | `run_history.py`, `history_config.py`, `history_model.py` | `results_history/` (`summary.txt/json`, `trials_<config>.csv`, `sweep_history.png`); models in `history_models/` |
| 4b Shape | `run_shape.py`, `shape_config.py` | `results_shape/` (`summary.txt/json`, `trials_<config>.csv`, `sweep_shape.png`) |

**The rule under test:** `alarm_classifier.py` and `classifier.py`, unchanged throughout. The frozen alarm threshold is in `results_alarm/chosen_alarm.json`.

**Not committed (regenerable):**
- `results_calibrated/traces/` and `results_history/episodes/` are per-step traces, excluded from git because of their size.
- `run_calibrated.py` and `run_history.py` recreate them exactly.
- Exp 4b reads `results_history/episodes/`, so run 4a first.
