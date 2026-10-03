"""The 3-way decision: body change / world event / ALARM.

Read decide() top to bottom; it is the whole rule.

    1. Look for SETTLED evidence for up to 10 s (not 5 s as before).
       Found -> judge exactly like the previous rule:
                mean error over 2 s of settled steps > body threshold -> BODY
                                                     otherwise        -> WORLD
       Waiting longer than before matters for big pushes: they DO settle, just
       later than 5 s allowed, and then they get a proper verdict.
    2. Never settled by 10 s -> check PERSISTENCE from 5 s to 10 s after onset:
       at least 80% of the 1 s blocks have mean error above the alarm threshold
       -> ALARM: "can't stabilise and still can't predict myself".
    3. Never settled, but error not persistently high -> WORLD: whatever
       happened has decayed; nothing says the body is wrong.

Why step 2 asks about time and persistence rather than just "didn't settle":
both a huge push and catastrophic damage can keep the pendulum from settling
for a while. Only damage keeps the MODEL wrong. Once the push has ended, the
physics is the physics the model learned, so its error falls back even while
the pendulum is still moving; with damage the error stays high as long as you
watch. Measuring from 5 s on, after pushes have had time to resolve, and
requiring the error to stay high in nearly every second, asks exactly "is
this still going on?".

What is reused unchanged: is_calm() from classifier.py (what "calm" means),
and the settle settings in classify_config.py. The previous rule itself is
classifier.calm_gated_score(..., calm_steps=2 * CALM_STEPS), used as is for
the comparison.
"""
import numpy as np

import alarm_config as ac
import classifier
import classify_config as cc

BODY, WORLD, ALARM = "body", "world", "alarm"


def settled_score(err, theta, thdot, onset):
    """Settled evidence, same definition as the previous rule, longer horizon.

    Counts transition t once the pendulum has been calm for more than
    SETTLE_STEPS consecutive transitions and t >= onset + MIN_LAG. Returns
    (score, decision_step) when SETTLED_NEEDED transitions are collected, or
    (None, onset + EXT_MAX_LAG) if that never happens.
    """
    picked, run = [], 0
    for t in range(onset + 1, onset + ac.EXT_MAX_LAG):
        run = run + 1 if classifier.is_calm(theta, thdot, t) else 0
        if t >= onset + cc.MIN_LAG and run > cc.SETTLE_STEPS:
            picked.append(err[t])
            if len(picked) == ac.SETTLED_NEEDED:
                return float(np.mean(picked)), t + 1
    return None, onset + ac.EXT_MAX_LAG


def late_blocks(err, onset):
    """Mean error of each 1 s block from ALARM_FROM to EXT_MAX_LAG after onset."""
    starts = range(onset + ac.ALARM_FROM, onset + ac.EXT_MAX_LAG, ac.ALARM_BLOCK)
    return np.array([err[s:s + ac.ALARM_BLOCK].mean() for s in starts])


def persistent(blocks, alarm_threshold):
    """True if at least PERSIST_FRAC of the blocks are above the alarm threshold."""
    return float(np.mean(blocks > alarm_threshold)) >= ac.PERSIST_FRAC


def decide(err, theta, thdot, onset, body_threshold, alarm_threshold):
    """The 3-way verdict. Returns (label, decision_step, path).

    path says which branch produced the verdict, so every outcome can be
    traced: "settled", "persistent" (alarm) or "decayed".
    """
    score, t = settled_score(err, theta, thdot, onset)
    if score is not None:
        return (BODY if score > body_threshold else WORLD), t, "settled"
    if persistent(late_blocks(err, onset), alarm_threshold):
        return ALARM, onset + ac.EXT_MAX_LAG, "persistent"
    return WORLD, onset + ac.EXT_MAX_LAG, "decayed"


def previous_rule(err, theta, thdot, onset, threshold):
    """The previous rule ("calm_long"), unchanged, for the paired comparison.

    Undecided (no settled evidence by 5 s) is acted on as WORLD, as before.
    """
    s, t, undecided = classifier.calm_gated_score(err, theta, thdot, onset,
                                                  calm_steps=2 * cc.CALM_STEPS)
    if undecided:
        return WORLD, t, "undecided"
    return (BODY if s > threshold else WORLD), t, "settled"
