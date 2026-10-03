"""Every knob for the 3-way classifier: body change / world event / ALARM.

Shared settings (physics, model, planner) stay in config.py, and the settle
logic's own settings (what "calm" and "settled" mean) stay in
classify_config.py and are reused unchanged. This file only adds what the
alarm needs. Time is in steps (dt = 0.05 s, 20 steps = 1 s).

THE PROBLEM THIS SOLVES
The previous rule waited at most 5 s for the pendulum to settle. If it never
did, the verdict was "undecided", acted on as "don't adapt". On the test set,
the most severe body changes (x1.6) were exactly the ones that fell into this
bin: damage bad enough that the robot could not hold itself up was silently
ignored.

WHY TIME SEPARATES A BIG PUSH FROM SEVERE DAMAGE
Both can stop the pendulum settling in a short window. They differ in what
happens next, because of what the self-model is wrong about:
  - A push is over in under a second. Afterwards the body obeys the physics the
    model knows, so the controller recovers, and even before it fully settles
    the model predicts the motion well: error drops back near its normal level.
    (Diagnostic runs: 4 N*m for 1 s settled by about 3.5 s, error ~0.010-0.018
    after the first second.)
  - Severe damage does not end. The model is wrong about every step, whether
    the pendulum is upright or swinging, so the error stays high for as long
    as you watch. (Diagnostic runs: x1.6-1.7 error 0.04-0.29 in every 1 s block
    out to 10 s, often with the pendulum fallen and swinging.)
So: watch long enough that any push has had time to resolve, then ask whether
the error is STILL high. "Still high after the time a push needs" is the alarm.
"""

# ---------------------------------------------------------------------------
# 1. Extended observation
# ---------------------------------------------------------------------------
# How long to keep looking for settled evidence (and, failing that, for
# persistence) after the onset. The previous rule stopped at 5 s.
EXT_MAX_LAG = 200          # 10 s

# Settled evidence: same definition as the previous chosen rule ("calm_long"):
# transitions counted once the pendulum has been calm for SETTLE_STEPS
# (classify_config), from MIN_LAG after onset; the verdict needs this many.
SETTLED_NEEDED = 40        # 2 s of settled transitions

# ---------------------------------------------------------------------------
# 2. The alarm (only consulted if no settled verdict by EXT_MAX_LAG)
# ---------------------------------------------------------------------------
# Persistence is measured from ALARM_FROM to EXT_MAX_LAG after onset, in
# blocks of ALARM_BLOCK steps. ALARM_FROM is set past the old 5 s window: in
# the diagnostic runs every push, including 4 N*m for 1 s, had recovered by
# then, so error measured from here on is about the body, not the push.
ALARM_FROM = 100           # 5 s after onset
ALARM_BLOCK = 20           # 1 s blocks
# ALARM if at least this fraction of blocks has mean error above the alarm
# threshold. 0.8 rather than 1.0: a swinging damaged pendulum can pass through
# a brief well-predicted stretch, and one quiet second should not cancel the
# alarm. Fixed in advance, not tuned.
PERSIST_FRAC = 0.8
# Alarm threshold on 1 s block-mean error.
#   None  -> chosen on the dev set by the criterion in compare_alarm.py
#   float -> use as is
ALARM_THRESHOLD = None
# Candidate thresholds for the dev search: geometric grid. Lower end is just
# above the normal error level (~0.010); upper end well above anything a
# recovered push produces.
ALARM_GRID_LO = 0.011
ALARM_GRID_HI = 0.2
ALARM_GRID_N = 40

# ---------------------------------------------------------------------------
# 3. Trial categories (each stresses one failure mode)
# ---------------------------------------------------------------------------
#   body     moderate body change: length or mass x U[BODY_MIN, BODY_MAX]
#   world    ordinary push: |torque| U[0.5, 2] N*m, 0.1-0.5 s (as before)
#   severe   severe body change: length or mass x U[SEVERE_MIN, SEVERE_MAX]
#   bigpush  big long push: |torque| U[BIG_TQ_MIN, BIG_TQ_MAX], BIG_STEPS steps
#   calib    undisturbed (sets the body/world threshold; checks false alarms)
# Moderate and severe ranges meet at 1.5 so every trial has one true class.
BODY_MIN, BODY_MAX = 1.2, 1.5
SEVERE_MIN, SEVERE_MAX = 1.5, 1.7
BIG_TQ_MIN, BIG_TQ_MAX = 2.0, 4.0
BIG_STEPS_MIN, BIG_STEPS_MAX = 10, 20      # 0.5-1.0 s

# trials per category
N_DEV = {"calib": 30, "body": 40, "world": 40, "severe": 40, "bigpush": 40}
N_TEST = {"calib": 50, "body": 100, "world": 100, "severe": 100, "bigpush": 100}
# fresh seed blocks, disjoint from every earlier experiment
SEED_DEV = 60_000
SEED_TEST = 70_000

# episode length: latest onset + full extended window + margin
EPISODE_MARGIN = 5
RESULTS = "results_alarm"
