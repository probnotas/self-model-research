# Investor demo — damage detection

A <1 minute demo built from a real recorded run. Nothing here is illustrative:
every number and every trace on the page is read out of `demo_trace.json`.

    python3 record_demo.py          # MODE=control (default) or MODE=probe
    python3 build_page.py           # selects the run to show, writes demo.html

## Why this experiment and not the others

| Candidate | Verdict |
|---|---|
| Rung 1 sample efficiency | Control curve is flat within noise (4.32 @ 5000 vs 5.16 @ 50 samples, std ~2.1) and SAC wins on final quality (2.32). Not demo-safe. |
| Rung 2 damage recovery | The honest control kills it: pre-trained SAC degrades +26% on the damaged body with no adaptation; the model-based controller degrades +139%. |
| **Rung 2 damage detection** | **10/10 runs, 0 false alarms in 1,100 healthy steps, 0.38 s median latency, robust to every threshold rule tested.** |
| Rung 3 ensembles / meta-learning | 0 of 15 pairwise comparisons separate. Inconclusive. |

## Results

Model trained on 2,000 random-action transitions from the healthy body, never
retrained. At step 120 the mass is multiplied by 1.5 mid-episode.

Threshold rules, all calibrated on healthy data only and leave-one-out across
runs so no run is scored against a threshold fitted to itself:

| Rule | Detected | Median latency | False alarms |
|---|---|---|---|
| healthy mean + 3σ | 10/10 | 0.38 s | 0 / 1100 |
| healthy mean + 4σ | 10/10 | 0.40 s | 0 / 1100 |
| max healthy value | 10/10 | 0.33 s | 1 / 1100 |

Detection is independent of control quality: swing-up succeeded in only 3 of 10
runs, and latency does not track control cost (r = -0.30).

## The counter-intuitive result

Random-torque probing produces a much larger error jump (9.0x vs 3.8x) but is
the *worse* detector — 1/10 at a zero-false-alarm threshold, against 10/10 under
closed-loop control. Thrashing the body widens the healthy baseline faster than
it widens the damage signal. Run `MODE=probe python3 record_demo.py` to reproduce.

## What this does not show

Detection, not diagnosis. Not a control result (the planner balances in a
minority of runs, and the damage left control cost unchanged, 2.80 -> 2.81).
Simulation, one body, one damage type. Recovery after damage remains unsolved
in our own data (see Rung 2 and Rung 3).
