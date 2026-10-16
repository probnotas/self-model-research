# JEPA on the quiet soft body

*Simulation only, not hardware.* This is a side experiment, separate from the main arc in `surprise_attribution/`.

| file | what it is |
|---|---|
| `exploratory_run1_PREREG.md` | the criteria for the first, quick run (seed 0 only) |
| `exploratory_run1.py`, `exploratory_run1_results.json` | the code and results of that first run |
| `PREREG.md` | the follow-up's rules and success criteria, committed in `a934be8` before the follow-up ran |
| `run_followup.py` | the follow-up: persistence, push recovery, raw-space ablation, 3 seeds |
| `followup_results.json`, `followup.log` | all follow-up numbers |

## Results against `PREREG.md` (all 3 seeds)

| criterion | result |
|---|---|
| **Collapse check (all runs valid)** | **PASS.** The summaries' spread grew during training in every run. |
| **P: persistence**, late window (5–10 s): JEPA body-vs-push AUC ≥ 0.90 and ≤ 10/120 push false positives, on both soft configurations | **FAIL** on every seed. AUC 0.472–0.491; false positives 62–67/120. |
| **L: latent space, not window**: the raw-space ablation's error does not rise, or rises far less | **FAIL** on every seed. The ablation's error rises at least as clearly (AUC 1.000 vs undisturbed). |

**Why P failed: the pushes never ended for the pendulum.**
- **How many fell:** on the soft joint, 61 of 120 pushes on soft/none and 62 of 120 on soft/low left the pendulum unable to rebalance at all within the episode. That's every 2 N·m × 0.5 s and every 4 N·m push.
- **Where the false positives come from:** the JEPA's late-window false positives are 61/61 of the non-recovered pushes, but only 0–6 of the ~59 recovered ones.
- **Rigid control:** 115 of 120 pushes recover, and persistence works there (late-window AUC 0.966–0.989, 7–12 false positives).

**Why L failed.** A raw-space model with the same 10-step input and 10-step output also sees its error rise after a body change. So the improvement over the one-step model comes from the window and horizon (plus training on healthy balancing data), not from predicting in a latent space.

See the session summary for full tables. Numbers are given as the mean and the min–max over seeds 0, 1 and 2.
