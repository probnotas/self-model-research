"""Every knob for the surprise-attribution experiment lives here.

Nothing else in this folder hard-codes a number that you might want to change
between runs. If you want to try a bigger push, a later onset, a different body
change or a longer "is it still elevated?" lag, edit this file only.

Time is measured in environment steps. Pendulum-v1 uses dt = 0.05 s, so
20 steps = 1 second.
"""

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# One seed controls the training data, the network init, the starting state of
# every episode and the planner's random samples. All three conditions use the
# SAME episode seed, so up to the disturbance step the three episodes are
# bit-identical. Any difference you see after the onset is caused by the
# disturbance and nothing else - that is what makes the comparison fair.
SEED = 0

# ---------------------------------------------------------------------------
# Forward model (the "self-model")
# ---------------------------------------------------------------------------
TRAIN_TRANSITIONS = 2000   # random-action transitions from the HEALTHY body
TRAIN_EPOCHS = 1000        # full-batch Adam epochs
LEARNING_RATE = 1e-3
HIDDEN = 64                # width of each of the two hidden layers
# True: the net outputs (next_state - state) and we add the state back.
# False: the net outputs next_state directly. The prediction-error signal is
# always computed in next-state space, so both options are directly comparable.
# Delta prediction is the default because it is the more accurate formulation
# for small-dt physics (the net only has to learn the small change).
PREDICT_DELTA = True
# Where the trained model is cached. Delete the file (or run with --retrain)
# to train a fresh one. To use YOUR model, see model.py -> load_model().
MODEL_PATH = "forward_model.pt"

# ---------------------------------------------------------------------------
# Controller (CEM model-predictive control, driven by the forward model)
# ---------------------------------------------------------------------------
HORIZON = 20               # planning horizon, steps
CEM_ITERS = 5              # refinement iterations per control step
CEM_CANDIDATES = 100       # action sequences sampled per iteration
CEM_ELITES = 10            # best sequences the Gaussian is refit to
CEM_INIT_STD = 1.0

# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------
EPISODE_STEPS = 240        # 12 s
# How each episode starts:
#   "upright" - within +/-START_RANGE rad (and rad/s) of the top. The controller
#               is holding balance when the disturbance hits, so the baseline
#               error is flat and low and any change is easy to see.
#   "random"  - Pendulum-v1's own reset (anywhere on the circle). The swing-up
#               transient adds its own error bumps to every condition, which
#               makes the signatures harder to read. Use it as a robustness check.
START_MODE = "upright"
START_RANGE = 0.2

# ---------------------------------------------------------------------------
# The disturbance (shared timing)
# ---------------------------------------------------------------------------
ONSET_STEP = 80            # step at which the body change / world event begins

# ---------------------------------------------------------------------------
# Condition 2: BODY CHANGE (permanent)
# ---------------------------------------------------------------------------
# Which physical parameter of the pendulum is changed, and by what factor.
# Valid names are the attributes Pendulum-v1 itself uses in its physics:
#   "l" - length (m), stock value 1.0
#   "m" - mass (kg),  stock value 1.0
#   "g" - gravity,    stock value 10.0
# Several can be changed at once, e.g. {"l": 1.5, "m": 1.5}.
#
# Note on mass: in Pendulum-v1's equation of motion, m appears ONLY in the
# torque term 3/(m l^2) * u - the gravity term 3g/(2l) sin(th) does not
# contain it. So a mass change is only visible while the controller is
# pushing (u != 0). While balancing near the top u is small, so a +50% mass
# change is a much quieter surprise than a +50% length change. Length is the
# default for that reason. Try {"m": 1.5} to see the quieter version.
BODY_CHANGE = {"l": 1.5}   # multiplicative factors applied to the stock values

# ---------------------------------------------------------------------------
# Condition 3: WORLD EVENT (transient)
# ---------------------------------------------------------------------------
# An external torque (N*m) added on top of the controller's torque for a few
# steps, then removed. It is applied inside the physics and is NOT part of the
# action the forward model sees, so the model has no way to predict it.
# For scale: the controller's own torque is limited to +/-2.0.
WORLD_TORQUE = 1.0
WORLD_DURATION = 5         # steps the push lasts (5 steps = 0.25 s)

# ---------------------------------------------------------------------------
# Summary features
# ---------------------------------------------------------------------------
# (a) before vs after: mean error in a window just before the onset vs a
#     window just after it.
PRE_WINDOW = 40            # steps [ONSET-PRE_WINDOW, ONSET)
POST_WINDOW = 20           # steps [ONSET, ONSET+POST_WINDOW)
# (b) persistence: mean error in a window that starts LATE_LAG steps after
#     the onset. LATE_LAG must be comfortably longer than WORLD_DURATION plus
#     the time the controller needs to recover, otherwise the world event has
#     not had a chance to subside and the test is meaningless.
LATE_LAG = 60              # steps after onset where the late window starts (3 s)
LATE_WINDOW = 20           # length of the late window
# "Still elevated" means the late-window mean exceeds the episode's own
# pre-onset mean by more than ELEVATED_SIGMAS pre-onset standard deviations.
ELEVATED_SIGMAS = 4.0

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUT_DIR = "results"        # CSVs, figure and summary go here
SMOOTH_WINDOW = 10         # trailing-mean window for the smoothed plot panel
