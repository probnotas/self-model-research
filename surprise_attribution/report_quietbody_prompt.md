TASK: Add a new technical report subpage to my research site (EPI Labs / Embodied Predictive Intelligence), and fix one false claim elsewhere on the site. The report is the next one in an existing series and must look like it belongs: same template, fonts, layout, nav and formatting as the existing report pages.

STEP 1: FIND AND COPY THE EXISTING TEMPLATE
- Find the existing technical report pages on the site.
- Base the new page on the most recent one. Reuse its stylesheet, header/nav, masthead, abstract box, status callout, section numbering, callouts, tables, figure styling, limitation rows, reproducibility block and footer. Do not invent a new design.
- If the site has shared layouts, components or a CSS file, use those rather than copying styles inline.
- Only if you cannot find the existing template, use this spec (the template's actual values):
  - Fonts (Google Fonts): "Spectral" (300/400/600/700, italic 400) for text; "IBM Plex Mono" (400/600) for labels, data, captions and code.
  - Light colours: paper #FAFAFB, surface #FFFFFF, surface-2 #F1F2F5, ink #14161B, ink-2 #4B515D, ink-3 #757C89, rule #DCDFE5, rule-soft #EAECF0, accent #7C2D3A, accent-soft #F5E8EA, good #1F5F43, warn #8A5A16.
  - Dark colours (prefers-color-scheme: dark): paper #0F1114, surface #15181C, surface-2 #1C2026, ink #E3E6EB, ink-2 #A7AEBA, ink-3 #7C8492, rule #2A2F37, rule-soft #20252B, accent #D98E99, accent-soft #2C1A1E, good #6FBF95, warn #D8A75C.
  - Page: max-width 780px, centred, 26px side padding. Body 17.5px / 1.66.
  - Masthead: mono uppercase eyebrow (11px, letter-spacing .13em, accent); H1 Spectral 700, clamp(31px, 4.6vw, 44px); italic light subtitle 20px ink-2; byline row in mono 12px between two hairline rules.
  - Abstract: surface box, 1px border, 3px accent left border, mono uppercase "ABSTRACT" label.
  - Sections: H2 Spectral 700 23px with a mono accent section number and a hairline underneath; H3 Spectral 600 italic 18px.
  - Callouts: surface box, 3px accent left border, mono uppercase accent label.
  - Tables: mono 12.6px, right-aligned numbers, uppercase mono header row on surface-2, inside a bordered scroll wrapper.
  - Captions: mono 11.5px, ink-3, starting with a bold "Figure N." or "Table N.".
  - Limitations: numbered rows, 26px mono accent number column plus text, hairline separators.
  - Code/repro blocks: mono 12.4px on surface with a 1px border.

STEP 2: CREATE THE PAGE
- Route: match the existing reports' URL pattern. If unclear, use /reports/the-quiet-body.
- Metadata:
  - Eyebrow: Technical Report · Self Model Research · Rung [N], where N is the next number after the most recent report already on the site. Tell me which number you used.
  - Title: The Quiet Body: How a Soft Robot Joint Reveals a Body Change
  - Subtitle: Prediction error failed on a simulated compliant joint, and eliminating the explanations one by one found why: a body change makes the robot quieter, not more surprised. A pre-registered test on fresh trials confirmed it.
  - Byline: Pendulum-v1 · Surprise attribution · October 2026 · Armaan Sharma
  - HTML <title> and meta description consistent with the other reports.
- Figure:
  - The image is at /assets/quiet-body-replication.png (I will place it; reference exactly this path).
  - Place it as Figure 1 in Section 5, after Table 3, with a white background box so the PNG reads in dark mode.
  - Alt text: "Four panels, one per simulated joint (soft no-noise, soft low-noise, rigid, medium). Each dot is one fresh trial's change in motion versus the healthy baseline, grouped as undisturbed, body change, push, and body change plus push. On the soft joints, body-change trials sit below zero (quieter) and many push trials sit above (louder)."
  - Caption: use the Figure 1 caption from the report text below.

STEP 3: CONTENT
Use the report text in part (A) below VERBATIM. Rules:
- Do NOT change, round, recompute or add any number. Every number on the page must appear in (A) exactly as written.
- Do NOT soften the limitations or the scoped claims. Do NOT describe the compliant joint as hardware or real anywhere. It is a simulation.
- Do NOT turn this into a success story bigger than the text. In particular, keep the sentence that the self-model is not the detector.
- Render the "(callout)" blocks as callouts, the tables as tables, the limitations as numbered limitation rows, and the code block as a repro block.
- Remove the bracketed layout notes, e.g. "(callout directly below the abstract)". They are instructions for you, not content.
- Part (B) is my lab notebook for the whole arc. It is for checking numbers only. Do not add content from it to the page.

STEP 4: FIX THE FALSE HARDWARE CLAIM ON THE SITE
- Somewhere on the site (likely the landing page roadmap), the first rung is marked as done with something like "R1 ✓ real compliant hardware". That is false. Nothing in this research has run on hardware; all of it is simulation.
- Change that item so it is clearly NOT done. Use something like: "R1 ○ real compliant hardware — not started; all results so far are in simulation". Keep the site's existing style for not-done items if there is one.
- Search the whole site for any other claim that the work runs on real, physical or compliant hardware, and correct each one the same way.
- Change nothing else on those pages.

STEP 5: LINK IT INTO THE SERIES
- Add the new report wherever the existing reports are listed (nav, reports index, homepage section, footer), in order after the most recent one.
- Give the new page a series footer listing all reports, with the current one marked "(this report)" and not linked.
- If the older report pages have a series footer, add this report to it. Change nothing else on those pages.

STEP 6: CHECK BEFORE YOU FINISH
- The page renders in light and dark mode, and at phone width (~390px) with no horizontal page scroll. Only tables and code blocks may scroll inside their own boxes.
- The figure path is /assets/quiet-body-replication.png.
- Spot-check these numbers against (A):
  - rule B AUC body vs push: 0.984 (soft, no noise) and 0.969 (soft, low noise);
  - false positives: 3/120 and 0/120;
  - self-model AUC vs undisturbed on soft: 0.717 and 0.449;
  - master seeds: 90000 and 777000.
- "hardware" never describes this work except in negation, anywhere on the site.
- Headings, fonts and spacing visually match the previous report.
- Report back with:
  - the new page's URL/route;
  - the Rung number you used;
  - every file you created or changed;
  - every hardware claim you found and how you changed it;
  - anything you could not place cleanly;
  - a screenshot if you can take one.

==================================================================
(A) REPORT TEXT (use verbatim)
==================================================================

## Abstract

A robot's self-model, a learned forward model of its own dynamics, is surprised whenever a prediction fails. The surprise might mean its body changed (adapt) or that the world pushed it (don't). On a rigid simulated pendulum, a rule built on one-step prediction error makes that call well: it never mistook a push for a body change (0/120). On a simulated soft series-elastic joint, the same approach failed, with scores inverted (AUC 0.202 and 0.286). We then ruled out explanations one at a time:
- per-body thresholds and per-body rescaling;
- a model with memory of hidden actuator state;
- removing the settle gate;
- longer prediction horizons;
- a model trained on the balancing regime.

The mechanism turned out to be the reverse of the assumption behind every score: on the soft joint, a body change makes the pendulum move about three times less while balancing. Prediction error tracks motion, so after a body change the self-model looks *less* surprised. A score that reads the change in behaviour directly ("quieter than its healthy self = body change") was then tested in a pre-registered experiment on fresh trials. On both soft configurations it separated body changes from pushes (AUC 0.984 and 0.969) with 3 and 0 false positives in 120 pushes. The detector is the robot's behaviour, not its self-model's surprise. Everything here is simulation, not hardware.

## Status of this result (callout directly below the abstract)

The headline result comes from a single pre-registered test on fresh trials (seed 777000), with the rules and success criteria committed before those trials existed. Everything before it is exploratory: those trials were analysed many ways, so it supports only the mechanism, not the claim. The detector catches 87 of 120 body changes on the soft joint without noise but only 24 of 120 with low sensor noise. It misses most body changes when a push happens at the same moment, and the direction it relies on ("quieter") is specific to this soft joint.

## 1. Motivation

Online adaptation needs a gate in front of it. Before a self-model rewrites itself, it has to decide whether the surprise was about itself or about the world. Earlier reports in this series showed that, on a rigid pendulum, the time course of one-step prediction error makes that call well.

Real robots aren't rigid. Their actuators have springs, lag, friction and noisy sensors. This report asks whether the same idea survives a compliant joint, and when it doesn't, why not. The answer changed what we think the right signal is.

## 2. Setup

- **System.** Gymnasium Pendulum-v1, balancing upright under a cross-entropy-method model-predictive controller that plans through a learned forward model (an MLP trained on 2,000 random-action transitions).
- **Surprise.** The one-step prediction error ‖s′ − f(s, a)‖.
- **Compliant joint (simulated).** The motor drives the link through a spring of stiffness k, with hysteresis, a parallel damper and joint damping. The motor and spring state are hidden from the robot. Optional Gaussian sensor noise is added. Stiffness levels: rigid, k = 200, 30 and 4 N·m/rad ("soft").
- **Trials.** A 2×2 factorial per body: nothing / body change only / push only / body change and push together, 120 trials each, plus 50 undisturbed calibration episodes.
  - **Body change:** mass ×1.3–1.7 or length ×1.2–1.5.
  - **Push:** 1, 2 or 4 N·m for 0.2 s or 0.5 s.
- **Decision points.** Every threshold is set on the body's own healthy calibration episodes, never on test trials.

## 3. The arc of elimination

Table 1 summarises the experiments on the original trials (master seed 90000). Each tested one explanation for why attribution failed on the soft joint.

Table 1. What each experiment tested, and what it showed. AUC compares body-change trials with trials that have no body change; below 0.5 means the scores are inverted.

| experiment | explanation tested | result on the soft joint |
|---|---|---|
| Compliant grid | does the rigid rule transfer? | No. AUC 0.202 (no noise) and 0.286 (low noise), inverted |
| Per-body calibration | wrong scale | AUC unchanged; z-scoring can't reorder scores |
| History-input model | model lacks hidden-state memory | model 4× more accurate (held-out error 0.0519 → 0.0118), yet AUC 0.047–0.479 |
| Shape feature, no settle gate | the settle gate | pushes now separate, but body changes look like undisturbed runs (median log-ratio 0.03 vs −0.01) |
| Rollout error | one-step horizon too short | never usefully above 0.5 before the rollout diverges |
| Regime-matched model | model trained on the wrong regime | error floor drops below the motion (0.0900 → 0.0278), but AUC stays inverted (0.236) |

The regime diagnostic explains the middle of the arc. With real motion, the soft model beats a trivial "nothing changes" predictor (ratio 0.29). During balancing it loses (ratio 1.61), because its error floor (about 0.08–0.10) is larger than the motion itself.

## 4. The mechanism: the body change makes the robot quieter

The regime-matched model removed the resolution problem, yet the scores stayed inverted. So we measured the motion directly, in the window 1–5 s after the disturbance.

Table 2. Median step-to-step motion of the pendulum, 1–5 s after onset (soft joint, no noise, seed 90000).

| trial type | median motion |
|---|---|
| undisturbed | 0.0501 |
| body change | 0.0150 |
| body change (mass only) | 0.0158 |
| body change (length only) | 0.0109 |

A heavier or longer link on the soft spring cuts the pendulum's movement to about a third. Across trials, prediction error tracks that motion (correlation 0.68–0.72). So after a body change the self-model's error goes *down*, and every score that read "more error = body change" read the soft joint backwards.

*Why* a heavier link on a soft spring moves less hasn't been isolated. Slower natural dynamics, or the controller fighting the spring less, are plausible, but neither has been tested.

## 5. Pre-registered test on fresh trials

### The rules, fixed before the data existed

Before generating any new trial, we committed:
- **Seed:** master seed 777000, never used before.
- **Configurations:** soft joint without and with low sensor noise as the targets; rigid and medium stiffness as controls.
- **Window:** 1–5 s after onset.
- **Score:** each trial's log mean motion as a z-score against the body's own 50 fresh healthy episodes.
- **Rules:**
  - **Rule A:** body change if |z| ≥ 2.
  - **Rule B:** body change if z ≤ −2 ("quieter").
  - **Rule C:** rule A on the self-model's error instead of motion.
- **Success criteria, on both soft configurations:**
  - **S1:** rule A AUC against undisturbed ≥ 0.75.
  - **S2:** rule B AUC against pushes ≥ 0.75, with at most 15 false positives in 120 pushes.

The fresh trials were scored once.

### Result: replicated

Table 3. Fresh trials, seed 777000. Rule B is "quieter than healthy = body change". Counts are out of 120.

| body | rule A: AUC vs undisturbed | rule B: AUC vs undisturbed | rule B: AUC vs push | rule B: body changes caught | rule B: false positives on pushes | rule B: body changes missed when a push coincides | rule C (self-model): AUC vs undisturbed / vs push |
|---|---|---|---|---|---|---|---|
| soft, no noise | 0.899 | 0.959 | 0.984 | 87 | 3 | 97 | 0.717 / 0.337 |
| soft, low noise | 0.831 | 0.944 | 0.969 | 24 | 0 | 110 | 0.449 / 0.346 |
| rigid (control) | 0.994 | 0.901 | 0.915 | 108 | 3 | 88 | 1.000 / 1.000 |
| medium (control) | 1.000 | 0.017 | 0.019 | 2 | 2 | 119 | 0.984 / 0.873 |

Both success criteria passed on both soft configurations. Median motion z after onset (undisturbed / body change / push):
- soft, no noise: +0.36 / −2.55 / +2.71;
- soft, low noise: +0.14 / −1.60 / +4.99.

On the medium joint the direction reverses (+13.94 after a body change), so "quieter" is not a general rule.

[FIGURE 1]

Figure 1. Change in motion against each body's healthy baseline, one dot per fresh trial (seed 777000); black bars are medians and dashed lines the pre-registered thresholds z = ±2. On the soft joint, body-change trials sit below zero. Push trials split: about half return to normal, the rest stay louder. On the medium joint, body changes go the other way. Simulated joint, not hardware.

## 6. Discussion

**The result is narrower than "attribution works on soft bodies".**
- **The self-model is not the detector.** Its own error, scored the same way, fails on the soft joint (rule C: 0.717 and 0.449 against undisturbed, 0.337 and 0.346 against pushes). What works is the robot's *behaviour* compared with its own healthy behaviour.
- **On the rigid and medium joints the self-model's error works well** (1.000 and 0.984). So the right signal depends on the body.

**Methodologically, the arc matters as much as the endpoint.** Every plausible fix aimed at the error signal (thresholds, scaling, memory, horizon, training data) failed for the same underlying reason, and finding that reason led to a signal that worked on data nobody had looked at.

**The open problems are concrete:**
- A real system has to learn, for its own body, which way its behaviour shifts after a body change.
- Body changes that coincide with a push are still mostly missed.
- Sensor noise cuts detection sharply (87 → 24 of 120).

## 7. Limitations

1. **Simulation, not hardware.** The compliant joint is a numerical model with chosen, not measured, parameters. Nothing here has run on a physical device.
2. **Direction is body-specific.** "Quieter = body change" holds on the soft and rigid joints tested and fails on the medium joint (AUC 0.019). It isn't a general rule.
3. **Detection is incomplete.** The soft joint with low noise catches 24 of 120 body changes. When a push coincides with the body change, 97–110 of 120 are missed.
4. **The self-model isn't the detector on the soft joint.** The replicated result is about behaviour; the self-model's surprise fails there.
5. **One controller, one model per body.** The controller was held fixed throughout. A different controller could change how much a body change alters the motion.
6. **Exploratory history.** The seed-90000 trials were analysed many ways; only the fresh-trial test supports a claim.
7. **Narrow test conditions.** Body changes were mass ×1.3–1.7 or length ×1.2–1.5; pushes were square pulses of 0.2 s or 0.5 s. Four configurations were replicated.
8. **Mechanism not isolated.** Why the soft joint moves less after a body change is an interpretation, not a measurement.

## 8. Reproducibility

```
Python 3.11.15 · numpy 2.4.6 · torch 2.14.0+cpu · gymnasium 1.3.0
master seeds: 90000 (exploratory arc) · 777000 (pre-registered replication)
model training seed: 0
pre-registration: surprise_attribution/PREREG_replication.md (committed before the fresh run)
replication:      surprise_attribution/run_replication.py → results_replication/
full arc:         surprise_attribution/findings_arc.md
```

## Conclusion

On a simulated soft robot joint, a body change doesn't make the self-model more surprised. It makes the robot quieter. Finding that took ruling out six other explanations. A score built on it passed a pre-registered test on fresh trials, separating body changes from pushes with almost no false alarms. It is a narrow, honest result: behaviour, not surprise, and simulation, not hardware. It points the next step at learning which way each body's behaviour shifts, and at a real joint.

==================================================================
(B) LAB NOTEBOOK (for checking numbers only; do not add to the page)
==================================================================

# An arc of elimination: surprise attribution on a compliant body

*In simulation only. The compliant body is a simulated series-elastic joint, not hardware.*

This document tells the story of the experiments as one argument. Each tested one explanation for a failure, ruled it out or narrowed it, and handed a sharper question to the next. The arc ends with a mechanism (on the soft body, a body change makes the robot *quieter*, not more surprised) and a pre-registered replication on fresh trials of a score built on that mechanism.

Every number below is copied from a results file in `surprise_attribution/`. A few numbers were computed directly from the per-trial CSVs rather than taken from a summary; those are marked *(from the per-trial CSV)*. Anything not logged is marked as such.

| step | experiment | results folder |
|---|---|---|
| 1 | Factorial, rigid body, simultaneous disturbances | `results_factorial/` |
| 2 | The same factorial on a compliant, noisy grid | `results_compliant/` |
| 3 | Per-body calibration (z-scoring) | `results_calibrated/` |
| 4a | History-input forward model | `results_history/` |
| 4b | Shape feature, no settle gate | `results_shape/` |
| 5 | Rollout (multi-step) error | `results_rollout/` |
| — | Diagnostic: error by motion regime | `results_regime/` |
| 6 | Regime-matched model | `results_regime_model/` |
| 7 | Behaviour-change score (exploratory) | `results_twosided/` |
| **R** | **Pre-registered replication, fresh seed** | **`results_replication/`** |

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


## 7. Interim conclusion after Experiment 4b (later refined)

After 4b, the four experiments had ruled out four explanations:

| explanation | ruled out by | evidence |
|---|---|---|
| wrong threshold | Exp 2 | already recalibrated per body; AUC < 0.5, so no threshold works |
| wrong scale | Exp 3 | z-scoring leaves every AUC unchanged; soft stays at 0.202 and 0.286 |
| model lacks memory of hidden state | Exp 4a | model 4× more accurate on soft (0.0519 → 0.0118); soft AUC still 0.047–0.479 |
| settle mechanism | Exp 4b | no gate at all; soft body changes still look like undisturbed runs (median r 0.03 vs −0.01) |

The interim conclusion was: *in this simulation, one-step prediction error is the wrong observable for detecting a body change on a compliant body.* Sections 8–11 tested that conclusion directly and refined it.

---

## 8. Experiment 5: rollout (multi-step) error

**Tested:** whether the problem is the one-step horizon. The same models were run free along the recorded actions for H = 1, 5, 10, 20, 40 and 100 steps, starting 1 s after onset when every push is over. Each trial averaged 8 start points. This was fully offline.

- **Checks:** H = 1 reproduced the recorded one-step error exactly (0 mismatches).
- **Divergence measure:** the healthy rollout error compared with a "nothing changes" predictor.

**Found** (`results_rollout/summary.txt`). AUC is body-only vs undisturbed.

| config | H=1 | 5 | 10 | 20 | 40 | 100 |
|---|---|---|---|---|---|---|
| medium, none | 0.962 | 0.999 | 0.991 | 0.959 | 0.452 | 0.063 |
| soft, none | 0.389 | 0.273 | 0.273 | 0.413 | 0.744 | 0.727 |
| soft, low | 0.401 | 0.322 | 0.354 | 0.387 | 0.480 | 0.450 |
| soft, medium | 0.447 | 0.247 | 0.207 | 0.270 | 0.280 | 0.222 |

**What happened on each body:**
- **Medium:** AUC peaks at H = 5–10, then collapses once the rollout blows up.
- **Soft/none:** reaches 0.744 only at H = 40, where the healthy rollout is already 12.5× worse than "nothing changes". It detects 5 of 240 body changes at the healthy threshold.
- **Soft/low:** never exceeds 0.5.

**The key fact:** on soft, the model's one-step error during balancing is already 1.88× (soft/none) and 1.56× (soft/low) worse than predicting no change at all.

**Lesson:** the horizon isn't the issue. The model doesn't track the soft body's balancing motion at any horizon.

---

## 9. Diagnostic: is the soft model broken everywhere, or only when quiet?

**Tested:** the model's one-step error against "nothing changes", split by motion regime at |θ̇| = 0.5 rad/s (the repo's `OMEGA_CALM`). Recorded data only.

- **Low-motion data:** the healthy episodes. They contain no high-motion steps, since all start within 0.2 rad of upright and balance.
- **High-motion data:** the post-push steps of push-only trials, when the body is unchanged and the push is off.

**Found** (`results_regime/summary.txt`):

| config | regime | model error | "nothing changes" | ratio |
|---|---|---|---|---|
| rigid, none | low motion | 0.0108 | 0.0503 | 0.21 |
| rigid, none | high motion | 0.0141 | 0.3565 | 0.04 |
| soft, none | low motion | 0.0843 | 0.0524 | **1.61** |
| soft, none | high motion | 0.0967 | 0.3309 | **0.29** |
| soft, low | low motion | 0.0675 | 0.0488 | **1.38** |
| soft, low | high motion | 0.1068 | 0.3107 | **0.34** |

**Lesson:** the soft model isn't globally broken; it beats "nothing changes" whenever the pendulum really moves. It has a flat error floor of about 0.08–0.10 at every motion level, 7–9× the rigid model's. During balancing that floor is larger than the motion itself.

---

## 10. Experiment 6: a regime-matched model

**Tested:** a forward model trained only on healthy balancing data.

- **Training data:** the pre-onset balancing transitions of every recorded trial (36,714–36,863 per configuration). These steps are never scored, so nothing leaks into the test. A second model on a random 2,000 transitions controls for data size.
- **Training settings:** the same architecture, settings and seed as before.
- **Controller unchanged:** the new model only scores the recorded trajectories.

**Found** (`results_regime_model/summary.txt`):

| config | motion | floor, original | floor, regime-matched | AUC body vs undisturbed, original → regime | AUC body vs push, original → regime |
|---|---|---|---|---|---|
| soft, none | 0.0477 | 0.0900 | **0.0278** | 0.074 → 0.236 | 0.538 → 0.115 |
| soft, low | 0.0429 | 0.0685 | **0.0327** | 0.246 → 0.337 | 0.374 → 0.153 |

**The floor dropped below the motion, but detection still failed.** The reason is in the motion itself: on the soft body, a body change cuts the pendulum's movement to about a third (median 0.0501 → 0.0150). Across trials, error tracks motion (r = 0.68–0.72), so the error *falls* after a body change.

**Outside balancing, the regime models are poor,** worse than "nothing changes" after pushes (0.44 vs motion 0.33). That is what broke push separation here.

**Lesson:** this explains the inverted AUCs since Exp 2. On the soft body, a body change makes the pendulum *quieter*, and every score built on "more error means body change" reads quieter as "less surprising".

---

## 11. Experiment 7: a behaviour-change score, then a pre-registered replication

**Tested:** whether *behaviour that differs from the body's own healthy baseline* separates body changes.

- **Signal:** the motion itself (‖s_{t+1} − s_t‖, no model), averaged over the window 1–5 s after onset.
- **Score:** a z-score against the body's 50 healthy calibration episodes.
- **Rules:** two-sided (rule A, |z| ≥ 2) and, after the exploratory run, signed (rule B, z ≤ −2, "quieter = body").

**Exploratory run, seed 90000** (`results_twosided/`). Soft/none reached AUC 0.912 body vs undisturbed with rule A, and 99% of body-change trials were quieter than normal. By then these trials had been analysed about seven ways, so this counted as a hypothesis only.

**Pre-registration.** `PREREG_replication.md` (commit `29193c5`) fixed the following before any fresh trial existed:
- **Seed:** 777000.
- **Configurations:** soft/none and soft/low as targets; rigid/none and medium/none as controls.
- **Rules A, B and C:** C is the self-model's error, scored two-sided.
- **Success criteria, on both soft configurations:**
  - **S1:** rule A AUC body vs undisturbed ≥ 0.75.
  - **S2:** rule B AUC body vs push ≥ 0.75, with ≤ 15/120 false positives.

**Fresh-trial result** (`results_replication/summary.txt`). Outcome: **REPLICATED.**

| config | rule A: AUC body vs undisturbed | rule B: AUC body vs undisturbed | rule B: AUC body vs push | rule B: body-only detected | rule B: false positives | rule B: masking | rule C (self-model): AUC body vs undisturbed / vs push |
|---|---|---|---|---|---|---|---|
| **soft, none** | **0.899** | 0.959 | **0.984** | 87/120 | **3/120** | 97/120 | 0.717 / 0.337 |
| **soft, low** | **0.831** | 0.944 | **0.969** | 24/120 | **0/120** | 110/120 | 0.449 / 0.346 |
| rigid, none (control) | 0.994 | 0.901 | 0.915 | 108/120 | 3/120 | 88/120 | 1.000 / 1.000 |
| medium, none (control) | 1.000 | 0.017 | 0.019 | 2/120 | 2/120 | 119/120 | 0.984 / 0.873 |

**Median motion z after onset** (undisturbed / body change / push):
- soft/none: +0.36 / −2.55 / +2.71
- soft/low: +0.14 / −1.60 / +4.99
- medium/none: +0.00 / +13.94 / +5.39 (the opposite direction)

---

## 12. The finding

**On a simulated soft series-elastic joint, a body change doesn't show up as surprise. It shows up as a lasting drop in how much the robot moves while balancing.**

- **The pre-registered test on fresh trials confirmed it.** A signed motion score separates body changes from undisturbed runs (AUC 0.944–0.959) and from pushes (0.969–0.984), with 0–3 false positives in 120 pushes.

**What this does not show:**
1. **The self-model isn't the detector here.** Its own error, scored the same way, fails on soft (0.717 and 0.449 vs undisturbed; 0.337 and 0.346 vs pushes). The signal is behavioural, and the arc's real contribution is finding that.
2. **It isn't a complete detector.**
   - soft/low catches only 24 of 120 body-only changes;
   - when a push lands at the same moment, masking is 97–110 of 120.
3. **The direction depends on the body.** Medium joints get louder (z +13.94), so "quieter = body" is specific to this soft joint. A real system would need to learn the direction for its own body.
4. **It's simulation, with one controller and one model per body.**

---

## 13. Open questions

1. **Learn the direction per body.** Can a robot work out which way its behaviour shifts after a body change from a few labelled examples, or from its own physics?
2. **Combine the signals.** On rigid and medium joints the self-model's error works; on soft, motion works. Is there a single rule that works across stiffness?
3. **Simultaneous pushes.** Masking is the largest remaining failure.
4. **Hardware.** None of this has touched a real joint.

---

## Caveats

1. **Simulated compliance, not hardware.** The joint parameters (`J_MOTOR` 0.05, `B_MOTOR` 0.5, `B_JOINT` 0.3, `HYST` 0.3, `SPRING_ZETA` 0.3, `N_SUBSTEPS` 10) were chosen, not measured from a device. The spring damper was added after a balance-only pilot whose numbers aren't logged.
2. **One training seed per model.** In Exp 4a the AUC swings between k values (for example rigid/none 1.000 → 0.661 → 0.993). These may be model-to-model noise.
3. **Controller held fixed.** Every experiment after Exp 2 used the single-state model's controller, including the replication. A different controller would produce different trajectories, and it may change how much a body change alters the motion.
4. **Medium-noise compliant configurations are degenerate.** Calibration episodes never settle (stiff 0/50, medium 1/50, soft 0/50), so the settle-based rule has no threshold. They weren't part of the replication.
5. **Rigid-tuned settings.** The calm limits, settle windows and alarm threshold were tuned on the rigid body and never re-tuned for the settle-based rule.
6. **Many analyses of the seed-90000 trials.** Experiments 2–7 all used the same 6,360 trials, which is why only the fresh-trial replication (seed 777000) supports a claim. Everything before it is exploratory.
7. **Narrow test conditions.**
   - Body changes: mass ×1.3–1.7 and length ×1.2–1.5.
   - Pushes: square pulses of 0.2 s or 0.5 s; no slow, fading pushes.
   - Replication configurations: soft/none, soft/low, rigid/none and medium/none only.
8. **Some mechanism statements are interpretations.** These include: hidden actuator state explains the held-out gain in 4a; *why* a heavier or longer link on a soft spring moves less. Each is labelled where it appears.

---

## Reproducibility

**Shared by all experiments:**
- **Versions:** Python 3.11.15, numpy 2.4.6, torch 2.14.0+cpu, gymnasium 1.3.0, logged in each folder's `versions_and_seed.json`.
- **Seeds:**
  - master seed 90000 for experiments 1–7;
  - **777000 for the replication**;
  - forward-model training seed 0;
  - regime-model subsample seed 0.

| experiment | code (in `surprise_attribution/`) | results |
|---|---|---|
| 1 Factorial, rigid | `run_factorial.py`, `factorial_config.py` | `results_factorial/` |
| 2 Compliant grid | `run_compliant.py`, `compliant_config.py`, `compliant_env.py`, `compliant_trials.py` | `results_compliant/`; models in `compliant_models/` |
| 3 Calibration | `run_calibrated.py`, `calibrated_config.py` | `results_calibrated/` |
| 4a History | `run_history.py`, `history_config.py`, `history_model.py` | `results_history/`; models in `history_models/` |
| 4b Shape | `run_shape.py`, `shape_config.py` | `results_shape/` |
| 5 Rollout | `run_rollout.py`, `rollout_config.py` | `results_rollout/` |
| Diagnostic | `diag_regime.py` | `results_regime/` |
| 6 Regime-matched model | `run_regime_model.py` | `results_regime_model/`; models in `regime_models/` |
| 7 Behaviour change | `run_twosided.py` | `results_twosided/` |
| **Replication** | **`PREREG_replication.md`, `run_replication.py`** | **`results_replication/`** |

**The original rule:** `alarm_classifier.py` and `classifier.py`. The frozen alarm threshold is in `results_alarm/chosen_alarm.json`.

**Not committed (regenerable):** the per-step traces in `results_calibrated/traces/`, `results_history/episodes/` and `results_replication/episodes/`. The run scripts recreate them exactly. Experiments 4b, 5, 6, 7 and the diagnostic read `results_history/episodes/`, so run 4a first.
