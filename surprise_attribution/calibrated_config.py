"""Per-body calibration of the surprise signal (run_calibrated.py).

SIMULATED compliant body, same as results_compliant/: a model of compliance,
not real hardware.

What changes versus results_compliant/: ONLY the input to the rule. The rule
(alarm_classifier.decide), the trials, the seeds, the bodies and the trained
models are all reused. Instead of the raw one-step error e_t, the rule reads

    z_t = (e_t - mu_body) / spread_body

where mu_body and spread_body describe that body's own healthy error, measured
on its 50 undisturbed calibration episodes (the same episodes the old rule used
to set its body/world threshold).

Fixed before any compliant result was looked at:
  * BASELINE WINDOW: every per-step error in the rule's own decision window,
    steps onset+1 .. onset+EXT_MAX_LAG-1, of each calibration episode. This is
    defined whether or not the episode ever settles, so bodies whose calibration
    never settled (the medium-noise configs) still get a baseline.
  * PRIMARY SPREAD: mean and standard deviation (the formula as specified).
    Median and MAD (x1.4826) are run as a secondary check only.
  * THRESHOLDS IN z UNITS: the rule's two thresholds are carried over from the
    rigid, noise-free body, expressed in that body's own baseline units:
        z_body  = (rigid body/world threshold - mu_rigid) / spread_rigid
        z_alarm = (frozen alarm threshold     - mu_rigid) / spread_rigid
    and then held fixed for every body. Nothing is re-tuned on compliant data.
    On rigid/no-noise this reproduces the old verdicts exactly (by construction).
"""
REFERENCE_CONFIG = "rigid_none"
SPREADS = ("std", "mad")        # "std" is primary; "mad" is the robustness check
MAD_SCALE = 1.4826              # makes MAD comparable to a std under a normal
RESULTS = "results_calibrated"
TRACES = "traces"               # per-step traces (regenerated, not committed)
