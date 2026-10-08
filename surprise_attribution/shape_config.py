"""Shape-based attribution: early-vs-late error in a fixed window (run_shape.py).

SIMULATED compliant body, same as results_compliant/: a model of compliance,
not real hardware.

What changes versus results_compliant/: ONLY how the verdict is decided. The
trials, seeds, bodies, controller and error signal (the single-state model's
one-step error, the signal the old rule read) are reused from the recorded
episodes in results_history/episodes/ - nothing is re-simulated or retrained.

THE FEATURE (fixed before any result was looked at)
  Window: W steps starting at the disturbance onset the rule is told
          (err[onset .. onset+W-1]; same onset the old rule used).
  Early:  mean error over the first half of the window.
  Late:   mean error over the second half.
  Shape:  r = log(late / early).  r < 0: error came back down (recovers -> world);
          r >= 0: error stayed up or grew (stays elevated -> body).
          A log ratio, so it is symmetric (halving and doubling are equally far
          from 0) and does not depend on the body's overall error scale.
  No settle gate: the window always opens and every trial gets a score.

THE DECISION POINT (set on healthy data only, never on factorial trials)
  For each body, the 50 undisturbed calibration episodes (the same ones the old
  rule calibrated on) give the distribution of r when nothing happens.
  "Recovered" means r fell clearly below what healthy variation produces:
      tau = mean(r_healthy) - Z_SHAPE * sd(r_healthy)
      body  if r >= tau,  world if r < tau.
  This is the rule exactly as specified: shape only.

SECONDARY RULE (also fixed in advance, reported separately)
  The shape rule alone cannot tell "stays elevated" from "nothing happened"
  (both have r near 0). The secondary rule adds the obvious missing half -
  elevation - also calibrated on the same healthy episodes:
      body  if r >= tau  AND  late >= mean(late_healthy) + Z_ELEV * sd(late_healthy)
      world otherwise.

W_MAIN is the primary window; W_SWEEP is the requested sweep (0.5 s, 1 s, 2 s).
"""
W_MAIN = 20
W_SWEEP = (10, 20, 40)
Z_SHAPE = 2.0
Z_ELEV = 2.0
RESULTS = "results_shape"
EPISODES = "results_history/episodes"   # recorded traces reused, not regenerated
