"""Summary features that could separate a body change from a world event.

Deliberately simple and hand-defined - no learned classifier. Each feature is
computed from ONE episode's error series and only compares that episode with
its own pre-onset stretch, so nothing here needs the other conditions.

Windows (all in steps, all set in config.py):

    |---- PRE ----|-- POST --|........|-- LATE --|
             onset^          onset+LATE_LAG^

  (a) before vs after
      pre_mean   mean error over [onset - PRE_WINDOW, onset)
      post_mean  mean error over [onset, onset + POST_WINDOW)
      post/pre   how much bigger the error got right after the onset.
      Both a body change and a world event should push this well above 1;
      this feature says THAT something happened, not WHAT.

  (b) persistence
      late_mean  mean error over [onset + LATE_LAG, onset + LATE_LAG + LATE_WINDOW)
      late/pre   is the error still bigger long after the onset?
      elevated   late_mean > pre_mean + ELEVATED_SIGMAS * pre_std
      Hypothesis: a body change is permanent, so the model stays wrong ->
      elevated = True. A world event is over once the push stops and the
      controller recovers, so the model is right again -> elevated = False.

The threshold for (b) uses the episode's own pre-onset noise level (mean and
std), so it is not tuned by looking at the disturbed conditions. That is what
keeps it honest: the rule was fixed before seeing any post-onset data.
"""
import numpy as np

import config


def window_stats(err, start, length):
    seg = np.asarray(err[max(0, start):start + length], dtype=float)
    return float(seg.mean()), float(seg.std())


def summarise(err, onset=config.ONSET_STEP):
    """Compute the features for one episode's error series."""
    pre_mean, pre_std = window_stats(err, onset - config.PRE_WINDOW, config.PRE_WINDOW)
    post_mean, _ = window_stats(err, onset, config.POST_WINDOW)
    late_start = onset + config.LATE_LAG
    if late_start + config.LATE_WINDOW > len(err):
        raise ValueError("late window runs past the end of the episode: "
                         "increase EPISODE_STEPS or shorten LATE_LAG/LATE_WINDOW")
    late_mean, _ = window_stats(err, late_start, config.LATE_WINDOW)
    threshold = pre_mean + config.ELEVATED_SIGMAS * pre_std

    post_seg = np.asarray(err[onset:onset + config.POST_WINDOW], dtype=float)
    return {
        "pre_mean": pre_mean,
        "pre_std": pre_std,
        "post_mean": post_mean,
        "post_over_pre": post_mean / pre_mean,
        "peak_after_onset": float(post_seg.max()),
        "late_mean": late_mean,
        "late_over_pre": late_mean / pre_mean,
        "threshold": threshold,
        "still_elevated": bool(late_mean > threshold),
    }


def print_table(summaries):
    """Print the features for every condition as one aligned table."""
    o, lag = config.ONSET_STEP, config.LATE_LAG
    print(f"\nSummary features  (onset = step {o}; late window starts {lag} steps "
          f"= {lag * 0.05:.1f} s after onset)\n")
    head = (f"{'condition':<13}{'pre mean':>11}{'post mean':>11}{'post/pre':>10}"
            f"{'peak':>11}{'late mean':>11}{'late/pre':>10}{'threshold':>11}"
            f"{'still elevated?':>17}")
    print(head)
    print("-" * len(head))
    for name, f in summaries.items():
        print(f"{name:<13}{f['pre_mean']:>11.2e}{f['post_mean']:>11.2e}"
              f"{f['post_over_pre']:>10.2f}{f['peak_after_onset']:>11.2e}"
              f"{f['late_mean']:>11.2e}{f['late_over_pre']:>10.2f}"
              f"{f['threshold']:>11.2e}{'YES' if f['still_elevated'] else 'no':>17}")
    print(f"\n  threshold = pre mean + {config.ELEVATED_SIGMAS:g} x pre std "
          f"(each episode's own pre-onset noise)")
