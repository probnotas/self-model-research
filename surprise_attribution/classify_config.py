"""Every knob for the classifier evaluation lives here.

The shared physics / model / planner settings stay in config.py. This file only
holds what is specific to "classify body change vs world event over many
randomised trials". Time is in environment steps (dt = 0.05 s, 20 steps = 1 s).
"""

# ---------------------------------------------------------------------------
# 1. The classifier
# ---------------------------------------------------------------------------
# Late window, in steps AFTER the onset. The score of a trial is the mean
# one-step prediction error over [onset + LATE_START, onset + LATE_END), which
# is the trailing mean evaluated at onset + LATE_END over a window of
# (LATE_END - LATE_START) steps.
# Default 1-2 s after onset. Why this window: it has to start AFTER a world
# event has ended and the controller has had time to recover (otherwise a push
# still looks like a body change), but it should be as early as possible so the
# verdict comes quickly. The boundary analysis below shows what happens when a
# push lasts long enough to overlap this window.
LATE_START = 20            # 1.0 s after onset
LATE_END = 40              # 2.0 s after onset

# Decision threshold on the late-window score.
#   None  -> calibrate it from UNDISTURBED episodes only (see CALIB_* below).
#            This is the default because it never looks at either test class,
#            so the evaluation cannot be tuned to the answer.
#   float -> use this value as is (e.g. an operating point you picked from
#            the ROC curve).
THRESHOLD = None
# Calibrated threshold = mean + CALIB_SIGMAS * std of the late-window scores of
# undisturbed episodes. 4 sigma: under a normal approximation an undisturbed
# episode would cross it about 3 times in 100,000.
CALIB_SIGMAS = 4.0
N_CALIB = 50               # undisturbed episodes used to set the threshold

# ---------------------------------------------------------------------------
# 2. The trial generator
# ---------------------------------------------------------------------------
N_PER_CLASS = 100          # trials per class (body change, world event)

# Onset is drawn uniformly per trial (steps). Starting at 2 s gives the
# controller time to settle into balance before anything happens; ending at
# 4 s keeps episodes short. Episode length is set automatically to
# ONSET_MAX + LATE_END + EPISODE_MARGIN so every late window fits.
ONSET_MIN = 40             # 2.0 s
ONSET_MAX = 80             # 4.0 s
EPISODE_MARGIN = 5

# Body change: which physical parameter, and the multiplier range.
#   "l" = length, "m" = mass. Each body-change trial picks one at random.
#   Only increases are drawn (heavier / longer), uniform in the range.
# Mass only enters the torque term of Pendulum-v1, so small mass changes are
# expected to be the hardest to detect. That is a real property of the body,
# not a bug, and the boundary analysis measures it.
BODY_PARAMS = ("l", "m")
BODY_FACTOR_MIN = 1.2
BODY_FACTOR_MAX = 1.6

# World event: an external torque the model cannot see, then removed.
#   Magnitude uniform in [min, max] N*m with a random sign (push left/right).
#   Duration uniform integer in [min, max] steps.
# For scale, the controller's own torque limit is +/-2 N*m.
# The duration range ends at 0.5 s, so every push in the main evaluation ends
# at least 0.5 s before the late window starts.
WORLD_TORQUE_MIN = 0.5
WORLD_TORQUE_MAX = 2.0
WORLD_STEPS_MIN = 2        # 0.1 s
WORLD_STEPS_MAX = 10       # 0.5 s

# ---------------------------------------------------------------------------
# 3. Boundary analysis
# ---------------------------------------------------------------------------
N_BOUNDARY = 20            # trials per grid cell
# A setting counts as "reliably handled" if at least this fraction of its
# trials is classified correctly (detected for body, rejected for world).
RELIABLE_RATE = 0.9
# Body-change magnitudes to sweep, per parameter (multipliers).
BOUNDARY_BODY_FACTORS = (1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.6)
# World-event grid: torques (N*m) x durations (steps).
# The longest duration (15 steps = 0.75 s) is deliberately longer than the
# main evaluation's range, to find where rejection breaks.
BOUNDARY_TORQUES = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
BOUNDARY_DURATIONS = (2, 5, 10, 15)

# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------
# Each block of trials draws from its own seed range so calibration, main
# evaluation and boundary trials never share a start state.
SEED_CALIB = 10_000
SEED_MAIN = 20_000
SEED_BOUNDARY = 30_000
# |theta| above this (rad) at the end of the late window counts as "fell".
# Logged per trial so you can see whether a verdict was made on a pendulum
# that had already fallen over (a confound, not a feature of attribution).
FELL_ANGLE = 0.5
N_WORKERS = 4              # parallel episode workers (1 = no multiprocessing)
OUT_DIR = "results_classifier"
