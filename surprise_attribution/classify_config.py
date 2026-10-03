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

# ---------------------------------------------------------------------------
# 4. Improved rule: wait until calm (compare_rules.py)
# ---------------------------------------------------------------------------
# Diagnosis behind it: after a big push the pendulum swings fast for 1-2 s
# (|theta_dot| up to ~7 rad/s; ~0.05 while balancing), and the healthy model
# is somewhat less accurate at speed. A fixed 1-2 s window therefore catches
# the after-swing of the push, not a wrong model. A body change, by contrast,
# keeps the pendulum calm but keeps the model wrong.
# So the improved rule only scores transitions where the pendulum is CALM
# (near upright and slow, at both ends of the transition) AND has been calm
# for SETTLE_STEPS in a row, starting no earlier than MIN_LAG after onset, and
# decides as soon as it has CALM_STEPS of them.
# Why the settle requirement: on dev, the first calm steps right after a
# pendulum was knocked down and swung back up still had the controller
# working hard (|u| ~0.6-0.7 vs ~0.44 at rest) and error ~0.015 vs ~0.010,
# enough to cross the threshold. After 0.5 s of continuous calm it is gone.
# If the pendulum never gives CALM_STEPS settled steps before MAX_LAG, the
# rule has no clean evidence and returns "undecided", which is acted on as
# "world event": do NOT adapt without evidence. Undecided trials are counted
# and reported separately.
MIN_LAG = 20               # never decide earlier than 1 s after onset
MAX_LAG = 100              # never wait longer than 5 s after onset
SETTLE_STEPS = 10          # 0.5 s of continuous calm before a step can count
CALM_STEPS = 20            # 1 s worth of settled transitions to average
OMEGA_CALM = 0.5           # rad/s: "slow" (balancing runs sit around 0.05)
THETA_CALM = 0.3           # rad:   "near upright"
# Pre-onset window for the ratio variant (score divided by the episode's own
# normal error level): [onset - PRE_STEPS, onset).
PRE_STEPS = 30

# Fixed episode length for trace recording: latest onset + longest wait.
TRACE_STEPS = ONSET_MAX + MAX_LAG + EPISODE_MARGIN

# Rule selection protocol: candidates are compared on a DEVELOPMENT set and
# one is chosen by the criterion in compare_rules.py; it is then evaluated
# once on a TEST set with fresh seeds that played no part in the choice.
SEED_DEV = 40_000          # dev:  calib +0, main +1000, boundary +2000
SEED_TEST = 50_000         # test: calib +0, main +1000, boundary +2000
N_DEV_PER_CLASS = 50       # dev is smaller: it only has to rank the rules
N_DEV_CALIB = 30
N_DEV_BOUNDARY = 10
RESULTS_RULES = "results_rules"
