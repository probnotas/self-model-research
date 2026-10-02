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
