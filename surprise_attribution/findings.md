# Findings: simultaneous disturbances and compliant bodies

This document records the results of two experiments on the surprise-attribution rule:

1. the **rigid factorial** (`run_factorial.py`, results in `results_factorial/`), and
2. the **compliant grid** (`run_compliant.py`, results in `results_compliant/`).

Every number below is copied from the logged result files: `summary.txt`, `summary.json`, the per-config `.json` files, the `trials*.csv` files and `versions_and_seed.json`. Where a statement rests on something that is *not* in those files, the text says so.

---

## The rule under test

`alarm_classifier.decide`, used unchanged in both experiments. It returns one of three verdicts:

1. **Settled path.** It waits for the pendulum to settle: at least 0.5 s continuously near upright and slow, starting 1 s or more after the onset. It averages the prediction error over 2 s of settled steps. If that average is above the **body/world threshold**, the verdict is **body**; otherwise it is **world**.
2. **Persistent path.** If the pendulum never settles within 10 s, it looks at the 1 s block-mean errors from 5 to 10 s. If at least 80% of them are above the **alarm threshold**, the verdict is **alarm**.
3. **Decayed path.** Otherwise the verdict is **world**.

How the thresholds are set:

- **Body/world threshold:** mean + 4 standard deviations of the settled score on **50 separate undisturbed episodes**, recalibrated for each body.
- **Alarm threshold:** **0.04042**, frozen. It was chosen on the rigid dev set (`results_alarm/chosen_alarm.json`).

How verdicts are scored:

- The rule has no "none" output. **"world" means "no persistent change, don't adapt"**, which is the correct verdict for both a push and for nothing happening.
- For a body change, **body or alarm** counts as detected, and **world** counts as missed.

---

## Experiment 1: rigid factorial

### What it tested

The question: can the rule tell a body change from a push when **both happen at the same moment, unannounced**? The body is the rigid Gymnasium `Pendulum-v1`. The design is a 2×2 factorial:

| factor | levels |
|---|---|
| body change | none · mass ×U[1.3, 1.7] · length ×U[1.2, 1.5] (half the body trials each) |
| world push | none · \|torque\| ∈ {1, 2, 4} N·m (random sign) × duration {short 0.2 s, medium 0.5 s} |
| overlap | push onset = body onset + U[−0.25, +0.25] s |

The design settings come from `factorial_config.py` (they are not repeated in the result files).

- **120 trials per cell.** Inside the push cells, the 6 magnitude × duration combinations are balanced at 20 each.
- **50 separate calibration episodes.** The CSV has 530 rows: 4 × 120 trials plus 50 calibration episodes.

### Results (`results_factorial/summary.txt`)

**Thresholds:** body/world 0.01377 (from 50 undisturbed episodes, all of which settled); alarm 0.04042 (frozen).

| ground truth | → body | → world | → alarm | n |
|---|---|---|---|---|
| neither | 0 | 120 | 0 | 120 |
| body only | 117 | 0 | 3 | 120 |
| world only | 0 | 120 | 0 | 120 |
| body + world | 70 | **19** | 31 | 120 |

| rate | value |
|---|---|
| overall misattribution | 19/480 = **4.0%** (95% CI 2.5–6.1%) |
| **false positive** (world only → body or alarm) | **0/120 = 0.0%** (95% CI 0.0–3.1%); 0 body, 0 alarm |
| **masking** (body + world → world) | **19/120 = 15.8%** (95% CI 10.4–23.4%) |
| body only → world (reference) | 0/120 = 0.0% |
| neither → body/alarm (reference) | 0/120 = 0.0% |

**By push magnitude and duration:**

| push | false positive | masking |
|---|---|---|
| 1 N·m, short | 0/20 | 0/20 |
| 1 N·m, medium | 0/20 | 8/20 = 40% |
| 2 N·m, short | 0/20 | 1/20 = 5% |
| 2 N·m, medium | 0/20 | 10/20 = 50% |
| 4 N·m, short | 0/20 | 0/20 |
| 4 N·m, medium | 0/20 | 0/20 |

Masking does **not** grow with push size. From `trials.csv`, the body+world verdict paths at 4 N·m were:

- **short pushes:** all 20 went to *alarm* via the persistent path;
- **medium pushes:** all 20 went to *body* via the settled path.

**Margin: the values the rule actually used.**

| group | settled | settled score min / median / max |
|---|---|---|
| body present | 187/240 | 0.0143 / 0.0313 / 0.0986 |
| body absent | 239/240 | 0.0080 / 0.0107 / 0.0129 |

- **Settled-score gap** (lowest body-present minus highest body-absent): **+0.0014**. That is clean separation, but narrow, with the threshold at 0.0138.
- **Never settled, body present (53 trials):** the fraction of late blocks above the alarm threshold ranged 0.00–1.00 (median 1.00), and late block medians ranged 0.0250–0.0849.
- **Never settled, body absent (1 trial):** fraction 0.00, block median 0.0236.

**Verdict paths per cell:**

| cell | settled | persistent | decayed |
|---|---|---|---|
| neither | 120 | 0 | 0 |
| body only | 117 | 3 | 0 |
| world only | 119 | 0 | 1 |
| body + world | 70 | 31 | 19 |

### Mechanism behind the masking

From `results_factorial/trials.csv`:

- **All 19 masked trials** took the *decayed* path: they never settled, and their error was not persistent enough to alarm.
- **18 of the 19 are length changes** (1 is a mass change).
- **Their late error sits around the alarm threshold.** The fraction of late 1 s blocks above it was 0.0, 0.2 ×8, 0.4 ×6 and 0.6 ×4, against the 0.8 required. Late block medians were 0.0250–0.0487, against an alarm threshold of 0.0404.

So the rule fails here because the error is raised but flickers around the alarm line, and it falls through the "decayed → world" branch.

**Not in these files:** *why* those pendulums never settled. The factorial CSV does not log the pendulum's angle. In the earlier overlap experiment (`results_overlap/`), an ad-hoc analysis of `results_overlap/traces.npz` found the masked pendulums lying fallen for the whole 5–10 s window, while their body-only and push-only twins stayed up. That analysis was not saved as a summary number. It is consistent with this experiment, but it was not re-measured here.

---

## Experiment 2: compliant grid

### What it tested

The question: does the separation survive when the body is **compliant and noisy** instead of rigid and clean? The same factorial (same master seed, same draws, same start states) was run on a **simulated** compliant pendulum (`compliant_env.py`):

- **Series-elastic joint.** The commanded torque drives a motor (inertia `J_MOTOR` 0.05, friction `B_MOTOR` 0.5). The motor reaches the link only through a spring of stiffness k, with a parallel damper at `SPRING_ZETA` 0.3 of critical damping.
- **Hysteresis.** The spring is ±30% (`HYST` 0.3) stiffer while loading than while unloading.
- **Joint damping.** `B_JOINT` 0.3.
- **Sensor noise.** Gaussian noise on the observed angle and angular velocity.
- **Integration.** 10 substeps per 0.05 s step (`N_SUBSTEPS`).
- **Same interface.** The robot observes only the link's [cos θ, sin θ, θ̇]; the motor and spring state are hidden.

**Grid** (stiffness from `summary.json`; noise levels from `compliant_config.py` and `summary.json`):

| stiffness | k (N·m/rad) | | noise | σ angle (rad) | σ velocity (rad/s) |
|---|---|---|---|---|---|
| rigid | (stock Pendulum-v1) | | none | 0 | 0 |
| stiff | 200 | | low | 0.002 | 0.02 |
| medium | 30 | | medium | 0.01 | 0.1 |
| soft | 4 | | | | |

**Procedure for each of the 12 configurations:**

1. A forward model is **retrained** on 2000 random-action transitions from that body, as sensed (with noise).
2. Rigid with no noise uses the original model and reproduces Experiment 1 exactly: same confusion matrix, gap +0.0014.
3. The rule sees the **observed (noisy)** angle and velocity, not the simulator's true state.
4. The body/world threshold is recalibrated per configuration (50 separate undisturbed episodes); the alarm threshold stays frozen at 0.04042.

### Results (`results_compliant/summary.json`, `summary.txt`)

**Definitions for the table:**

- **Held-out**: the model's one-step error on 1000 fresh random-action transitions from that body.
- **Normal error**: the mean settled score of the 50 undisturbed calibration episodes, i.e. the error while balancing with nothing happening.
- **FP**: world-only trials called body or alarm, out of 120, with the split (body / alarm).
- **Mask**: body+world trials called world, out of 120.
- **Missed (no push)**: body-only trials called world, out of 120.
- **Flagged undisturbed**: "neither" trials called body or alarm, out of 120.

| config | held-out | calib settled | normal error | body/world thr | FP (body/alarm) | mask | missed (no push) | flagged undisturbed | misattribution | gap | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rigid, none | 0.0162 | 50/50 | 0.0107 | 0.0138 | 0 (0/0) | 19 | 0 | 0 | 4.0% | **+0.0014** | **1.000** |
| rigid, low | 0.0311 | 50/50 | 0.0248 | 0.0354 | 0 (0/0) | 67 | 46 | 1 | 23.8% | −0.0167 | 0.930 |
| rigid, medium | 0.1265 | 50/50 | 0.1245 | 0.1867 | 15 (0/15) | 72 | 120 | 1 | 43.3% | −0.1084 | **0.291** |
| stiff, none | 0.0074 | 50/50 | 0.0506 | 0.0568 | 3 (3/0) | 84 | 29 | 0 | 24.2% | −0.0126 | 0.980 |
| stiff, low | 0.0263 | 50/50 | 0.0636 | 0.0830 | 1 (1/0) | 94 | 77 | 2 | 36.2% | −0.0541 | 0.830 |
| stiff, medium | 0.1220 | **0/50** | n/a | n/a | 116 (0/116) | 3 | 3 | **119** | 50.2% | −0.0241 | 0.767 |
| medium, none | 0.0786 | 50/50 | 0.0672 | 0.0919 | 5 (1/4) | 117 | 120 | 0 | 50.4% | −0.0271 | 0.955 |
| medium, low | 0.0824 | 50/50 | 0.0482 | 0.0631 | 18 (2/16) | 120 | 120 | 0 | 53.8% | −0.0281 | **0.373** |
| medium, medium | 0.1462 | **1/50** | 0.1367 | n/a | 120 (0/120) | 1 | 0 | **120** | 50.2% | n/a | n/a |
| soft, none | 0.0527 | 50/50 | 0.0907 | 0.1256 | 26 (0/26) | 120 | 120 | 0 | 55.4% | −0.0401 | **0.202** |
| soft, low | 0.0585 | 50/50 | 0.0690 | 0.1011 | 52 (0/52) | 119 | 120 | 0 | 60.6% | −0.0410 | **0.286** |
| soft, medium | 0.1377 | **0/50** | n/a | n/a | 120 (0/120) | 0 | 0 | **120** | 50.0% | n/a | n/a |

**Note on the "n/a" cells:** where calibration episodes never settled, `summary.json` logs `NaN` for the threshold and the gap. For medium/medium, the normal error comes from the single calibration episode that settled. AUC is P(settled score with body change > settled score without one); 0.5 means no separation, and below 0.5 means **inverted**.

### The headline numbers

1. **False positives** (world only → body or alarm), with no sensor noise:

   | rigid | stiff | medium | soft |
   |---|---|---|---|
   | 0/120 | 3/120 | 5/120 | 26/120 (21.7%) |

   With low noise: 0, 1, 18 and **52/120 (43.3%)**. With medium noise: 15/120 on rigid, and 116–120/120 on every compliant body.
2. **Masking** (body+world → world), with no noise: rigid 15.8%, stiff 70.0%, medium 97.5%, soft 100.0%.
3. **Body changes missed with no push at all**, with no noise: rigid 0/120, stiff 29/120, medium **120/120**, soft **120/120**.
4. **The settled-score margin.** It goes from **+0.0014** (rigid, no noise) to **negative in every other configuration where it could be computed.** The stiffest joint with no noise is already at −0.0126.
5. **AUC falls below 0.5, meaning the scores invert**, in four configurations: soft/none 0.202, soft/low 0.286, rigid/medium 0.291, medium/low 0.373. There, trials *without* a body change have *higher* settled scores than trials with one.
6. **Noise breaks it on the rigid body too.** On rigid, body changes missed with no push go 0 → 46 → **120/120**, AUC goes 1.000 → 0.930 → 0.291, and at medium noise 15/120 pushes alarm.

**False positives by push magnitude** (`by_push` in `summary.json`; world-only trials called body or alarm, out of 40 per magnitude):

| config | 1 N·m | 2 N·m | 4 N·m |
|---|---|---|---|
| soft, none | 0 | 10 | 16 |
| soft, low | 0 | 19 | 33 |
| medium, low | 3 | 4 | 11 |
| rigid, medium | 0 | 7 | 8 |
| stiff, none | 3 | 0 | 0 |

### Mechanisms

**1. The normal error rises with compliance, so a body change no longer stands out.**

- The error while balancing with nothing happening is **0.0107** on rigid. With no noise it is **0.0506** on stiff, **0.0672** on medium and **0.0907** on soft.
- On rigid, the settled scores *with* a body change ranged 0.0143–0.0986.
- So on compliant bodies, normal error alone is the size of a rigid-body change signal. The recalibrated thresholds rise accordingly (0.0568, 0.0919, 0.1256), and body changes fall below them.
- **This does not track held-out prediction error.** The stiff body's model is *more* accurate than rigid on random data (0.0074 vs 0.0162), yet its error while balancing is about 5× higher.
- *Interpretation, not measured:* the spring and motor states are hidden from the robot, and closed-loop balancing excites them in a way random-action data does not. No experiment isolated this.

**2. Many compliant trials never settle, so the settled score is often not used at all.**

`settled b/nb` counts trials that reached a settled verdict, with and without a body change:

| config | body present settled (of 240) | body absent settled (of 240) |
|---|---|---|
| stiff/none | 134 | 196 |
| stiff/low | 51 | 185 |
| medium/none | **7** | 185 |
| medium/low | **13** | 192 |

On medium stiffness, body-change trials almost never settle. They fall through the persistence branch and, below the alarm threshold, end up as world. That is why medium/none misses 120/120 body changes even though its settled-score AUC is 0.955 on the few that did settle.

**3. At medium noise on compliant bodies, the rule alarms on everything.**

- Calibration episodes did not settle (stiff 0/50, medium 1/50, soft 0/50), so no body/world threshold could be set.
- Every trial then went down the persistence branch.
- The noisy error level (held-out 0.12–0.15) is far above the frozen alarm threshold of 0.04042, so nearly every trial alarmed. That includes 119–120 of 120 **undisturbed** runs, all via the persistent path (from the trials CSVs).
- The low masking numbers there (0–3/120) are not detection. The rule has stopped discriminating. Misattribution sits at 50.0–50.2%.

**4. Why the scores invert (AUC < 0.5) is not diagnosed.**

- The inversion is logged (soft/none, soft/low, rigid/medium, medium/low).
- No analysis in these files explains why trials without a body change score higher. One possibility is that heavier or longer bodies move more slowly and predictably on a soft joint. That is untested and should not be cited as a finding.

---

## Caveats (as logged and as designed)

1. **Simulated compliance, not hardware.** The compliant body is a model: series-elastic + hysteresis + damping + Gaussian noise. The parameters (`J_MOTOR`, `B_MOTOR`, `B_JOINT`, `HYST`, `SPRING_ZETA`, `N_SUBSTEPS`) were chosen, not measured from a real joint. They are logged in `results_compliant/versions_and_seed.json`.
2. **The spring damper was added after a pilot, and the pilot isn't logged.** It was added after a balance-only pilot showed the undamped joint could not balance. The stiffness levels were picked so each joint balanced 6/6 in that pilot, before any attribution trial. **The pilot's numbers are not saved in any results file.** It was an ad-hoc check, and only its design consequence is recorded (the `SPRING_ZETA` comment in `compliant_env.py`).
3. **Rigid-tuned settings were held fixed.**
   - The alarm threshold stayed frozen at 0.04042 (chosen on the rigid dev set).
   - The "calm" limits (|θ| < 0.3 rad, |θ̇| < 0.5 rad/s) and the settle/persistence windows are the rigid values.
   - Only the body/world threshold was recalibrated per configuration, by the rule's own procedure. Recalibrating the others might change the result; that is untested.
4. **One trained model per configuration.** Model-to-model variation (training seed) is not measured.
5. **Rigid/no-noise uses the original model** (`forward_model.pt`). A retrained rigid model differed slightly in its weights, which I attributed to CPU thread count. That comparison was not logged; the original model was used so the earlier factorial is reproduced exactly.
6. **One run per configuration, fixed seed.** The confidence intervals in `summary.json` cover trial-to-trial variation, not reruns.
7. **The fallen-pendulum explanation for masking comes from the earlier overlap experiment's traces** and is not logged in the factorial results (see Experiment 1).

---

## Reproducibility

**Shared by both experiments** (`versions_and_seed.json` in each results folder):

| item | value |
|---|---|
| Python | 3.11.15 |
| numpy | 2.4.6 |
| torch | 2.14.0+cpu |
| gymnasium | 1.3.0 |
| master seed | 90000 (all trial seeds derive from it; same draws in every configuration) |

**Compliance parameters** (`results_compliant/versions_and_seed.json`): `J_MOTOR` 0.05, `B_MOTOR` 0.5, `B_JOINT` 0.3, `HYST` 0.3, `SPRING_ZETA` 0.3, `N_SUBSTEPS` 10.

**Code** (in `surprise_attribution/`):

| file | role |
|---|---|
| `run_factorial.py`, `factorial_config.py` | Experiment 1 |
| `run_compliant.py`, `compliant_config.py`, `compliant_env.py`, `compliant_trials.py`, `plot_compliant.py` | Experiment 2 |
| `alarm_classifier.py`, `classifier.py` | the rule (unchanged) |
| `results_alarm/chosen_alarm.json` | the frozen alarm threshold |
| `overlap_trials.py` | episode loop for Experiment 1 |
| `forward_model.pt` | the rigid model |
| `compliant_models/*.pt` | retrained models, one per non-rigid-none configuration |

**Results:**

| path | contents |
|---|---|
| `results_factorial/summary.txt` | Experiment 1 summary |
| `results_factorial/trials.csv` | 530 rows: per-trial ground truth, verdict, path, settle time, settled score, late-window persistence values, overlap timing |
| `results_factorial/versions_and_seed.json` | versions and seed |
| `results_compliant/summary.txt`, `summary.json` | all 12 configurations |
| `results_compliant/<config>.json` | per-configuration detail, including `by_push` and the confusion matrix |
| `results_compliant/trials_<config>.csv` | per-trial results, 12 files |
| `results_compliant/sweep.png` | false-positive, masking, no-push miss rate and AUC versus noise, per stiffness |
| `results_compliant/versions_and_seed.json` | versions, seed and compliance parameters |

**Commands:**

```
python run_factorial.py                       # Experiment 1
python run_compliant.py                       # Experiment 2, all configs (resumes from cached per-config JSON)
python run_compliant.py --only soft_low       # one configuration
python run_compliant.py --summary             # rebuild summary + figure from finished configs
```

---

## In one paragraph

On the rigid pendulum, the rule never mistook a push for a body change (0/120). But it missed 15.8% of body changes when a push landed at the same time, all through one route: never settled, error flickering below the alarm requirement. The settled scores separated by only +0.0014. On a simulated compliant body that margin disappears. Every compliant or noisy configuration has a negative gap, four configurations invert outright (AUC 0.202–0.373), and false positives climb to 21.7% (soft, no noise) and 43.3% (soft, low noise). On medium and soft joints, every body change is missed even with no push. Sensor noise does the same on the rigid body. At medium noise on compliant bodies the rule alarms on everything, undisturbed runs included. **The rule, with its rigid-tuned settings, is a rigid-body result. In this simulation it does not transfer to a compliant, noisy body.**
