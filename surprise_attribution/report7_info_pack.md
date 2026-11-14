TASK FOR THE WEBSITE AGENT: write and publish Report #7 in my self-model research series on embodiedpredictiveintelligence.vercel.app.

Everything you need is below. **Part A** is how to build it, **Part B** is the verified facts (the only numbers you may use), **Part C** is the citations, and **Part D** is the writing rules. Do not use any number that isn't in Part B.

==================================================================
PART A: HOW TO BUILD IT
==================================================================
1. **Find the existing report pages** on the site (there are 6). Copy the most recent one's template exactly: masthead, eyebrow ("Technical Report · Self Model Research · Rung N"), abstract box, status callout, numbered sections, tables, figure styling, numbered limitation rows, reproducibility block, series footer. Do not invent a new design. Read 2 or 3 of the earlier reports first and match their tone and length (under 2000 words).
2. **Route:** follow the existing URL pattern. If unclear, use `/reports/controller-body-coupling`.
3. **Title:** use "Why Prediction Error Falls When the Body Changes". It is the most accurately scoped of the three options.
   - Suggested subtitle: "In simulation, on a soft compliant pendulum, a heavier body makes the controller push less and the arm move less, so the self-model looks less surprised. A decomposition of the ×0.30 motion drop."
   - Byline: "Pendulum-v1 · Compliant body · November 2026 · Armaan Sharma".
4. **Figures** (copy from the research repo into the site's assets folder):
   - Figure 1: `experiments/controller-coupling-check/motion_vs_time.png`
   - Figure 2: `experiments/controller-coupling-check/torque_vs_time.png`
   - Figure 3: `experiments/controller-coupling-check/prediction_error_vs_time.png`
   - Optional Figure 4: `experiments/resonance-check-in-env/ratio_vs_freq.png`

   Put each on a white background box so it reads in dark mode. Captions must say "in simulation".
5. **Link the report into the series:** reports index, nav, and the series footer on every report, with this one marked "(this report)".
6. **Publish:** commit and push to the site repo (Vercel auto-deploys). Then confirm the live URL loads, renders in light and dark mode, and has no horizontal scroll at 390 px width.
7. **Report back:** the live URL, the Rung number you used, the files changed, and any [VERIFY] markers still in the text.

==================================================================
PART B: VERIFIED FACTS (from github.com/probnotas/self-model-research, branch claude/surprise-attribution-experiment-ivcvjp)
==================================================================

## B1. What is simulated
**Everything is in simulation. Nothing has run on hardware.** Say "in simulation" in the summary, the abstract, the method and the limits.

- **Body:** Gymnasium Pendulum-v1 (m = 1 kg, l = 1 m, g = 10, dt = 0.05 s, torque limit ±2 N·m, speed limit 8 rad/s), task = **balancing upright**.
- **Compliance** (`surprise_attribution/compliant_env.py`):
  - **Series-elastic joint:** the motor drives the link only through a torsional spring, **k = 4 N·m/rad** (the "soft" setting).
  - **Spring hysteresis ±30%:** stiffer while loading, softer while unloading.
  - **Viscous damper parallel to the spring:** 0.3 of critical damping.
  - **Motor-side inertia:** 0.05 kg·m², with motor friction 0.5.
  - **Joint damping:** 0.3 /s.
  - **Integration:** 10 substeps per 0.05 s step.
  - **Sensor noise options:** none, or low (σ = 0.002 rad on angle, 0.02 rad/s on velocity). Noise only affects what the robot sees, not the physics.
  - These parameter values were chosen, not measured from a device.
- **Learned forward model ("self-model")** (`model.py`, `config.py`):
  - MLP: 4 inputs (cos θ, sin θ, θ̇, torque) → 64 → 64 → 3 outputs, ReLU, predicting the change in state.
  - Trained on **2,000 random-action transitions** from the healthy body (seed 0, reset every 200 steps), with 1,000 full-batch Adam epochs at learning rate 1e-3.
  - Held-out test: 1,000 fresh random-action transitions. Held-out error on the soft, no-noise body: 0.0519.
  - It never saw balancing data or a changed body.
- **Controller** (`planner.py`):
  - A CEM model-predictive controller that plans through the learned forward model: horizon 20 steps, 5 iterations, 100 candidate torque sequences, 10 elites.
  - Its cost is the predicted squared distance from upright-at-rest, with no torque penalty.
  - **Call it "a model-predictive controller planning through the learned self-model". It is NOT a separately trained policy.**
- **Episodes:**
  - 325 steps (16.25 s); start within ±0.2 rad and ±0.2 rad/s of upright.
  - Body change at a step drawn from 60–80 (3–4 s).
  - Scoring window: 1–5 s after the change (onset+20 to onset+100 steps).
- **Metrics:**
  - **Prediction error:** the one-step error, ‖s(t+1) − ŝ(t+1)‖.
  - **Motion:** mean ‖s(t+1) − s(t)‖ of the observed state over the window.
  - **Controller command:** mean |torque| over the window.

## B2. The inverted fault-detection signal (earlier experiments, seed 90000, exploratory)
These AUCs come from the 2×2 factorial: 120 trials per cell, body changes mass ×1.3–1.7 or length ×1.2–1.5. They are settled-score AUCs, body change vs no body change; below 0.5 means inverted.

| model | soft, no noise | soft, low noise | source |
|---|---|---|---|
| **one-step model (the standard self-model)** | **0.202** | **0.286** | `results_compliant/summary.txt` |
| 4-step-history model | 0.047 | **0.110** | `results_history/summary.txt` |
| regime-matched model (window score, body vs undisturbed) | 0.074 → 0.236 | 0.246 → 0.337 | `results_regime_model/summary.txt` |

**IMPORTANT:** "AUC = 0.11" is the soft, *low-noise* body scored with the *4-step-history* model. It is NOT the one-step error.
- For the one-step error, say AUC **0.202** (soft, no noise).
- If you want one headline sentence: "On the soft body, error-based detectors scored AUC between 0.05 and 0.34 across model variants (0.5 = chance), all inverted."

Also from earlier work (`findings_arc.md`, Exp 6 and the replication):
- **The motion drop in the main experiment:** on the soft body, a body change cut median motion from **0.0501 to 0.0150** (×0.30). This comes from the seed-90000 trials, with mixed mass and length changes.
- **Error tracks motion:** across trials, the forward model's error correlates with motion at **r = 0.68–0.72**. That is why less motion reads as less surprise.
- **Pre-registered replication** (`PREREG_replication.md`, commit `29193c5`, fresh seed 777000; this was Report 6's result): a signed "quieter = body" motion score separated body changes from undisturbed runs (AUC 0.959 soft/none, 0.944 soft/low) and from pushes (0.984, 0.969). The self-model's own error did not (0.717 and 0.449 vs undisturbed). Medium-stiffness joints get *louder* instead, so the direction is body-specific.

## B3. New: the passive-physics check (`experiments/resonance-check-in-env/`, commit `3d73bd6`)
**Question tested:** Prof. Shai Revzen (UMich BIRDS Lab) suggested on Nov 11, 2026 that the drop could be pure resonance detuning. Can the body alone produce ×0.30?

**Method:**
- The same compliant env, unchanged (k = 4, no noise).
- The controller is replaced by an open-loop torque 2·sin(ωt), at 15 frequencies from 0.1 to 10 rad/s, plus a fine grid of 2–6 rad/s in 0.05 steps.
- Each run: 30 s of settling, then amplitude measured over 10 s.
- Conditions: healthy vs mass ×1.5 (plus length ×1.35 as an extra).

**Found:**
- **Smallest mass ×1.5 amplitude ratio at any frequency: ×0.65** (fine grid ×0.67–0.92). Length ×1.35: ×0.48.
- **The resonance peak didn't move down:** 3.5 → 3.5 rad/s (fine grid 3.50 → 3.65). Pure spring-mass detuning would predict ×0.82.
- **Why:** swinging about the bottom, gravity's restoring torque scales with mass just like inertia does, so mass barely detunes this body.
- **Limitation:** an open-loop drive can't balance the pendulum upright, so this tests the hanging regime of the same body, not the balancing regime.

**Conclusion:** passive dynamics alone can't produce ×0.30 for a mass change.

The earlier simplified playground (`experiments/resonance-check/`, a plain torsional-spring pendulum, not my env) gave at best about ×0.45 at k = 4. Mention it only as a first sanity check.

## B4. New: the controller coupling check (`experiments/controller-coupling-check/`, commit `b905da5`)
**Method:**
- Same env (soft, no noise), same forward model file (`compliant_models/soft_none.pt`), same controller and same episode loop as the main experiment.
- **30 fresh seeds, 640000–640029,** not used by any other experiment. Each seed is run twice: healthy, and mass ×1.5 at the onset.
- The paired runs are bit-identical before the onset (checked).
- **Not formally pre-registered:** the decision rule was written in the README before the run, but committed in the same commit as the results. Describe it as "decision rule fixed before running", not "pre-registered".

**Found** (window = 1–5 s after the change):

| metric | healthy (mean ± SD) | mass ×1.5 (mean ± SD) | ratio of means | paired ratio (95% bootstrap CI) |
|---|---|---|---|---|
| motion | 0.0542 ± 0.0218 | 0.0165 ± 0.0051 | **×0.30** | ×0.32 (0.27–0.40) |
| controller torque magnitude (N·m) | 0.7742 ± 0.2829 | 0.3887 ± 0.0812 | **×0.50** | ×0.53 (0.46–0.63) |
| prediction error | 0.0922 ± 0.0074 | 0.0839 ± 0.0017 | **×0.91** | ×0.91 (0.89–0.94) |

- **The drop reproduces the main experiment:** motion medians 0.0589 → 0.0172 here, vs 0.0501 → 0.0150 in the main experiment.
- **The drop starts within about 0.5 s of the change and lasts the whole window.** All 30 seeds stay upright.
- **It isn't torque saturation:** torque is at the ±2 N·m limit on 0.1% of healthy steps and 0% of changed steps.
- **Pre-set rule outcome: A (controller output drops on the heavier body).**
- **The decomposition (ratio of means, both factors from this experiment):**

  | factor | ratio |
  |---|---|
  | controller torque, changed / healthy | ×0.50 |
  | motion per unit torque, changed / healthy (0.0425 vs 0.0701 motion per N·m; paired geometric mean ×0.61, per-seed range 0.39–0.92) | ×0.61 |
  | **product** | **×0.30** |

  The ×0.61 "motion per torque" factor is the passive body responding less. It agrees with the open-loop result in B3 (×0.65–0.67; 1/1.5 = 0.67).
- **CORRECTION to the outline I gave you:** both ×0.61 and ×0.50 come from THIS experiment (closed loop). Do NOT attribute ×0.61 to "pure sine drive" or ×0.50 to "closed-loop vs open-loop". The sine-drive experiment (B3) gave ×0.65 and corroborates the ×0.61 factor.

## B5. What is NOT established (put these in the report)
- **Which way the causation runs inside the loop.** The controller reacts to the state, so "less torque → less motion" and "less motion → smaller corrections → less torque" both fit, and probably reinforce each other. The replay experiment that would separate them has NOT been run: apply each seed's healthy torque sequence open-loop to the heavier body.
- **Scope:**
  - B3/B4 cover mass ×1.5 only, soft body, no sensor noise.
  - The main experiment also had length changes and low noise.
  - One body topology (a single-joint pendulum), one controller type, one forward-model training seed.
- **No AUC was computed in B3/B4.** AUC numbers come only from the seed-90000 experiments (B2) and the 777000 replication.
- **No theory or proof,** only simulation measurements.

==================================================================
PART C: CITATIONS
==================================================================
Use only these. Keep every [VERIFY] marker visible in the published text until I confirm.

1. **Bongard, J., Zykov, V., Lipson, H. (2006).** "Resilient machines through continuous self-modeling." *Science* 314(5802). A robot infers its own body model from action and sensing, and re-models after damage.
2. **Kwiatkowski, R., Lipson, H. (2019).** "Task-agnostic self-modeling machines." *Science Robotics* 4(26). A learned self-model, re-trained after a body change.
3. **Chen, B., Kwiatkowski, R., Vondrick, C., Lipson, H. (2022).** "Fully body visual self-modeling of robot morphologies." *Science Robotics* 7(68). [VERIFY volume/issue]
4. **Isermann, R. (2006).** *Fault-Diagnosis Systems: An Introduction from Fault Detection to Fault Tolerance.* Springer. The classical residual-based FDI framework: a fault shows up as a residual (model vs measurement) exceeding a threshold.
5. **Bechtle, S., Hammoud, B., Rai, A., Meier, F., Righetti, L.** "Leveraging forward model prediction error for learning control." arXiv 2020 / ICRA 2021. [VERIFY authors, year, venue, and the claim that it uses rigid bodies and is not a body-change study] It uses forward-model prediction error inside controller learning.
6. **SelfWAM (Pan et al., 2026).** [VERIFY: full title, authors, venue, and what it does. I could not confirm this paper. Do not describe its contents beyond "learned self-models integrated with world models" until verified.]
7. **Wang et al. (2026).** [VERIFY: full title, authors, venue. Same rule as 6.]

==================================================================
PART D: WRITING RULES
==================================================================
- **Under 2000 words.** Follow the structure: one-line summary, background, method, results, mechanism, honest limits, implications, next steps, reproducibility.
- **No em dashes anywhere.** Use commas, periods or colons.
- **"In simulation" early and often.** Never describe the joint as hardware or real.
- **No overclaiming:**
  - No "assumption broken", "overturns" or similar.
  - Scope every claim: one simulated soft pendulum, this controller, mass ×1.5.
  - Implications for humanoids or deployed robots must be phrased as hypotheses ("may", "would need testing").
- **Plain English for the mechanism**; symbols only in the method section.
- **Keep the honest-null voice of the series.**
  - Say plainly that the passive-physics explanation was tested and covers only part of the drop.
  - Say plainly that the causal direction inside the loop is still open.
- **Mechanism, in plain words:** the arm gets heavier, so the same push moves it less. The controller sees smaller wobbles, so it pushes more gently, and the arm moves even less. The self-model's error mostly tracks how much the arm moves, so the error goes down too. A detector that waits for error to go *up* reads the changed body as "fine".
- **Reproducibility block:**
  - Repo: https://github.com/probnotas/self-model-research (branch `claude/surprise-attribution-experiment-ivcvjp`)
  - Commands:
    ```
    python experiments/resonance-check-in-env/experiment.py
    python experiments/controller-coupling-check/experiment.py
    ```
  - Seeds: 90000 (exploratory arc), 777000 (pre-registered replication), 640000–640029 (coupling check). The open-loop check is deterministic, with start states seeds 0–4.
  - Logs: `results.json` and `episodes.npz` in each experiment folder. Pre-registration: `surprise_attribution/PREREG_replication.md` (commit `29193c5`).
