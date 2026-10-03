"""The classifier: one number per trial, one comparison.

    score      = mean one-step prediction error over the late window
                 [onset + LATE_START, onset + LATE_END)
    prediction = "body" if score > threshold else "world"

That is the whole classifier. No learned parts, no other features. The reason
it should work: a world event ends, so once it is over the model's physics is
correct again and its error returns to normal. A body change does not end, so
the model stays wrong for as long as you look. Asking "is error still high a
second after the onset?" asks exactly "did the cause persist?".

What it is given: the true onset step. In a deployed system the onset would
come from a separate detector (error crosses a line). Here it is given, so this
evaluation measures attribution (body vs world), not detection.
"""
import numpy as np

import classify_config as cc

BODY, WORLD = "body", "world"


def late_window_score(errors, onset):
    """Mean prediction error over the late window after `onset`.

    This equals the trailing mean of the error, evaluated at onset + LATE_END,
    with a window of (LATE_END - LATE_START) steps. It only uses errors up to
    that step, so an online system could compute it at that moment.
    """
    lo, hi = onset + cc.LATE_START, onset + cc.LATE_END
    if hi > len(errors):
        raise ValueError(f"late window ends at step {hi} but the episode has "
                         f"{len(errors)} steps; increase EPISODE_MARGIN")
    return float(np.mean(errors[lo:hi]))


def classify(score, threshold):
    """Above the threshold: the surprise persisted, so call it a body change."""
    return BODY if score > threshold else WORLD


def calibrate_threshold(undisturbed_scores):
    """Threshold from undisturbed episodes only: mean + CALIB_SIGMAS * std.

    Uses no body-change or world-event data, so the evaluation is not tuned
    to the classes it is scored on.
    """
    s = np.asarray(undisturbed_scores, dtype=float)
    return float(s.mean() + cc.CALIB_SIGMAS * s.std(ddof=1))


# ---------------------------------------------------------------------------
# Improved rule: judge the model only while the pendulum is calm
# ---------------------------------------------------------------------------
# Why: the fixed 1-2 s window fails on big pushes because the pendulum is
# still swinging fast then, and the healthy model is less accurate at speed.
# That extra error is about the STATE, not about the body. Scoring only calm
# transitions removes it, so what is left is "is my model still wrong when
# everything is quiet?", which is exactly the body-change question.

def is_calm(theta, thdot, t):
    """Transition t (state t-1 -> state t) is calm if BOTH ends are near
    upright and slow. Checking both ends means the whole step happened in the
    regime the controller normally operates in.

    theta / thdot[t] are the state AFTER step t, so the state the transition
    started from is index t - 1.
    """
    return (abs(theta[t - 1]) < cc.THETA_CALM and abs(thdot[t - 1]) < cc.OMEGA_CALM and
            abs(theta[t]) < cc.THETA_CALM and abs(thdot[t]) < cc.OMEGA_CALM)


def calm_gated_score(err, theta, thdot, onset, calm_steps=None):
    """Mean error over the first `calm_steps` SETTLED transitions after onset + MIN_LAG.

    A transition t counts if it, and the SETTLE_STEPS transitions before it,
    were all calm. So the pendulum has to have been quiet for a while, not
    just pass through a quiet moment (e.g. the top of a swing).

    Returns (score, decision_step, undecided):
      score          the number compared to the threshold, or -inf if undecided
                     (-inf is below every threshold, so undecided is never
                     called "body change" whatever threshold is chosen)
      decision_step  the step at which the verdict is available
      undecided      True if fewer than `calm_steps` settled transitions
                     occurred before onset + MAX_LAG
    """
    n_needed = cc.CALM_STEPS if calm_steps is None else calm_steps
    picked, run = [], 0
    for t in range(onset + 1, onset + cc.MAX_LAG):
        run = run + 1 if is_calm(theta, thdot, t) else 0
        if t >= onset + cc.MIN_LAG and run > cc.SETTLE_STEPS:
            picked.append(err[t])
            if len(picked) == n_needed:
                return float(np.mean(picked)), t + 1, False
    return float("-inf"), onset + cc.MAX_LAG, True


def pre_onset_level(err, onset):
    """The episode's own normal error level just before the onset."""
    return float(np.mean(err[onset - cc.PRE_STEPS:onset]))
