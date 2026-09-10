# Investor demos

**Use `story.html`** - detection and self-repair in one continuous run. The other
two are the experiments it was built from.

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


---

# Demo 2 — damage recovery (recommended for the pitch)

`recovery.html`. Three pendulums side by side from the same starting state:
healthy, motor cut to 50%, and the same broken body after the model retrained
on 25 transitions.

    python3 balance_probe.py      # which damage actually breaks balance-hold
    python3 recovery_probe.py     # the repair curve
    python3 record_recovery.py    # records the three runs -> recovery_trace.json
    python3 sac_balance.py        # the model-free control

## Why balance-hold and not swing-up

Swing-up from hanging is beyond this planner. `planner_sweep.py` scores success
rate (not mean cost, which is meaningless on a bimodal outcome) across horizons
and CEM budgets:

| horizon | iters | candidates | mean cost | swung up |
|---|---|---|---|---|
| 20 | 5 | 100 | 3.680 | 25% |
| 40 | 5 | 100 | 5.478 | 0% |
| 40 | 8 | 200 | 4.592 | 12% |
| 60 | 8 | 200 | 3.743 | 25% |
| 80 | 8 | 300 | 4.992 | 0% |
| 100 | 8 | 300 | 4.269 | 25% |

More horizon and more compute do not help, so this is compounding model error,
not a tuning problem. Balance-hold is a task the planner can actually do.

## Which damage breaks it

Balance-hold, healthy model, 10 starts:

| body | mean cost | held upright |
|---|---|---|
| healthy | 0.010 | 100% |
| mass +50% (the Rung 2 damage) | 0.217 | 90% |
| mass x2 | 12.751 | 0% |
| **motor at 50%** | **12.751** | **0%** |
| length +50% | 0.085 | 90% |

Mass +50% barely dents balance-hold, which is why Rung 2's damage never made a
legible demo. A weakened actuator does.

## Repair curve (motor at 50%)

| repair samples | on-policy | random probing |
|---|---|---|
| 10 | 0% | 30% |
| 25 | **80%** | 10% |
| 50 | 90% | 0% |
| 100 | 90% | 80% |
| 200 | 100% | 80% |
| 500 | 100% | 100% |

On-policy data (collected while the stale controller tried and failed) reaches
80% at 25 samples; random babbling needs 500 for the same result.

## The control that limits the claim

SAC, 20,000 steps on the healthy body, scored on the identical task:

| condition | mean cost | held upright |
|---|---|---|
| healthy body | 0.006 | 100% |
| motor at 50%, **no adaptation** | 3.964 | **80%** |

The model-free policy absorbs this damage without adapting at all, where the
model-based planner drops to 0%. The self-repair is real; "beats RL at staying
upright" is not available from this data. The defensible differentiators are
that the model is acquired with no reward function, is reusable across tasks,
and can detect its own damage (see demo 1) - a policy cannot tell you it is wrong.


---

# Demo 3 — detection AND self-repair in one run (USE THIS ONE)

`story.html`. One continuous 260-step recording, no cuts:

| act | steps | what happens |
|---|---|---|
| 1 | 0-59 | healthy body, healthy model — balances |
| 2 | 60-159 | actuator drops to 50% mid-run — notices in 0.13 s, falls, is stood back up at step 110, falls again |
| 3 | 160-259 | model fine-tuned on the 100 transitions from those two failed attempts — balances again, same broken motor |

    python3 record_story.py     # PH_A / PH_B / STEPS / RETRY / REPAIR_N via env

## Results across all 10 runs

| measure | result |
|---|---|
| upright at end of act 1 | 10/10 |
| attempts ending on the floor in act 2 | 20/20 |
| upright at end of act 3 | 8/10 |
| damage detected | 10/10, median 0.13 s |

Detection is unchanged by the threshold rule (all calibrated leave-one-out on
healthy data only): mean+3σ 10/10 at 0.13 s with 5 false alarms in 500 healthy
steps; mean+4σ and max-healthy both 10/10 at 0.13 s with 2.

## Why the repair data is collected this way

The repair set is the transitions from the failed attempts, and *where* they come
from matters more than how many there are:

| repair data | runs balancing afterwards |
|---|---|
| 25 from one continuous fall | 6/10 |
| 50 from one continuous fall | 6/10 |
| 60 from three 1 s attempts | 8/10 |
| 100 from two 2.5 s attempts | 8/10 |

Doubling the data from a single fall changes nothing; splitting it across
attempts that each start near upright is what helps, because that is the region
the controller has to be accurate in. A single 1 s attempt is also too short to
fall (only 6 of 30 ended fallen), which is why attempts are 2.5 s.

## Head to head with RL (`head_to_head.py`)

All conditions on the same damaged body from the same ten start states, so the
comparison is paired. Exact Wilcoxon signed-rank on cost, exact McNemar on the
hold rate (no scipy in this environment; both enumerated directly at n=10).

| condition | mean cost | held upright |
|---|---|---|
| model-based, healthy | 0.009 | 10/10 |
| model-based, stale model | 13.503 | 0/10 |
| **model-based, repaired on 100 moves** | **0.657** | **9/10** |
| SAC, healthy | 0.005 | 10/10 |
| SAC, no adaptation | 4.927 | 3/10 |
| **SAC, same 100 steps of adaptation** | **0.387** | **9/10** |

| comparison | result |
|---|---|
| stale vs repaired | 0/10 -> 9/10, McNemar **p=0.004**, cost p=0.002 |
| repaired vs SAC with matched budget | 9/10 vs 9/10, **p=1.00**, cost p=0.85 |
| repaired vs SAC unadapted | 9/10 vs 3/10, p=0.031 — but see caveat |
| healthy: model-based vs SAC | both 10/10; SAC lower cost, p=0.002 |

**The self-repair is significant. A sample-efficiency win over RL is not.** Given
the same 100 steps of experience on the broken body, the model-free policy
recovers just as well.

Caveat on "repaired beats unadapted SAC": unadapted SAC scored 8/10 in
`sac_balance.py` and 3/10 here, differing only in the start-state draw. Ten
starts is too few to pin that number, so the claim should not be leaned on. The
matched-budget tie does not depend on it.

## What survives

1. No reward function: 2,000 self-generated transitions vs 20,000 reward-driven
   steps for SAC. 10x fewer samples and no reward engineering.
2. The model is reusable across tasks; a policy is welded to the one it trained on.
3. Self-diagnosis has no model-free counterpart. A policy emits actions, not
   predictions, so it has nothing to compare against its body.
