# Inverse-dynamics mismatch and command gain on the soft pendulum

*Simulation only, not hardware.*

**Files:**
- `PREREG.md`: committed in `0ed3ead` before the run.
- `run_inverse.py`: the code.
- `results.json` and `run.log`: all numbers.

**The model:** an endpoint inverse-dynamics model, in the style of Toso, LeCun, Anderson and Bounou's EP-IDM. It sees 10 states before and 10 after a point and guesses the 10 commands in between. It was trained on healthy data and tested on the seed-777000 trials, with 3 seeds.

## Pre-registered criteria (early window, both soft configurations, all 3 seeds)

| criterion | result |
|---|---|
| **M1:** mismatch detects (AUC body vs undisturbed ≥ 0.75) | **FAIL.** soft/none 0.337 (inverted), soft/low 0.808 |
| **M2:** mismatch attributes (AUC body vs push ≥ 0.75, ≤ 15 false positives) | **FAIL.** 0.180 and 0.372; 62 and 67 false positives |
| **G:** command gain detects and attributes | **FAIL.** Detection passes (0.995 and 0.987), but body vs push is 0.508 and 0.514, with 65 and 60 false positives |

## What happened

- **Mismatch (|guessed − commanded|) falls into the same quiet-body trap as one-step error.** On soft/none it points the wrong way and tracks motion (r 0.57 within undisturbed runs).
- **Command gain is an excellent, almost motion-independent body-change detector.**
  - **What changes:** a heavier or longer link responds less to the same commands. The median gain drops from 0.87 to 0.50 on soft/none, and from 0.75 to 0.52 on soft/low.
  - **How well it detects:** it catches 119/120 and 104/120 body changes, with motion correlation −0.12 and 0.20.
- **The false positives are the pushes the pendulum never recovers from:** 60/61 and 60/62 non-recovered pushes are flagged, against only 1–4 of about 58 recovered ones. That's the same physical confound as in `experiments/jepa`.
- **Post-hoc observation, not a claim:** in the late window, fallen pendulums have a gain near zero (0.07 and 0.03), while body changes sit near 0.5. Separating "weaker response" from "no response" would need its own pre-registration and fresh trials.
