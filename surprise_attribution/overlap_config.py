"""Every knob for the simultaneous-disturbance test (body change + overlapping push).

WHY OVERLAP IS THE HARD CASE
Every earlier test gave the classifier one cause at a time. Its rule is:
wait until the pendulum has settled, then ask whether the model is still
wrong. That works because the two causes leave different traces over time:
a push's error ends, a body change's error does not. When both happen
together, their traces are added on top of each other:
  - the push can MIMIC a body change: a large or long push keeps the
    pendulum unsettled and the error high, so the push alone might be
    read as "body" (or "alarm"). That is the false positive, the
    safety-critical error: the robot would rewrite a correct self-model.
  - the push can MASK a body change: the push's large transient dominates the
    early error and delays settling. That is harmless if the rule waits it
    out, but it could push the settled window late, change where the pendulum
    settles, or (if it never settles) route the trial to the alarm/decay
    branch. Then a real body change could be called "world": a false negative.
Which one happens, if either, depends on timing: a push landing right on the
body-change onset hides the body change's onset inside the push's own spike.

DESIGN: PAIRED TRIPLETS
Each draw of (seed, start state, body change, push, timing) is run three ways:
  1 body only   2 push only   3 body + push
plus 4 undisturbed controls. Because 1 and 3 differ ONLY by the push, masking
is measured exactly: "detected without the push, missed with it" on the very
same body change. Likewise 2 and 3 differ only by the body change.

NOTHING HERE IS TUNED. The classifier is alarm_classifier.decide, unchanged,
with the alarm threshold frozen from its own dev stage and the body/world
threshold set by the same procedure as before (mean + 4 std of settled scores
on undisturbed episodes). With no free parameter to choose there is no dev
stage: one run on fresh seeds is the evaluation.
"""

# ---------------------------------------------------------------------------
# Disturbance draws (shared by all three members of a triplet)
# ---------------------------------------------------------------------------
BODY_ONSET_MIN = 60        # 3 s: leaves room for a push up to 1 s BEFORE it
BODY_ONSET_MAX = 80        # 4 s
BODY_PARAMS = ("l", "m")
BODY_FACTOR_MIN, BODY_FACTOR_MAX = 1.2, 1.6
PUSH_TORQUE_MIN, PUSH_TORQUE_MAX = 0.5, 4.0     # N*m, random sign
PUSH_STEPS_MIN, PUSH_STEPS_MAX = 2, 20          # 0.1-1.0 s

# Overlap offset = push onset - body onset, in steps, uniform.
#   negative: push lands BEFORE the body change (body changes mid-push or
#             while the pendulum is still recovering)
#   ~0:       push lands right ON the body-change onset
#   positive: push lands AFTER the body change, up to 2 s later, i.e. inside
#             the window where the rule is collecting settled evidence
OFFSET_MIN = -20           # -1.0 s
OFFSET_MAX = 40            # +2.0 s
# Bins for the breakdown (seconds, left-closed). Edges chosen to separate
# "before", "on top of", "shortly after" and "during the judging window".
OFFSET_BINS = (-1.0, -0.25, 0.25, 1.0, 2.0001)
OFFSET_BIN_NAMES = ("before (-1 to -0.25 s)", "on onset (+/-0.25 s)",
                    "shortly after (0.25-1 s)", "during judging (1-2 s)")
# Push-magnitude bins for the masking breakdown (|torque|, N*m) and duration
TORQUE_BINS = (0.5, 1.0, 2.0, 3.0, 4.0001)
DURATION_BINS = (2, 6, 11, 16, 21)              # steps: 0.1-0.25, 0.3-0.5, 0.55-0.75, 0.8-1.0 s
# Finer offset grid for the worst-case-timing curve (seconds)
OFFSET_FINE_STEP = 0.25

# ---------------------------------------------------------------------------
# Sizes and seeds
# ---------------------------------------------------------------------------
N_TRIPLETS = 200           # -> 200 body-only, 200 push-only, 200 body+push
N_UNDISTURBED = 50
SEED_BASE = 80_000         # fresh: disjoint from every earlier experiment
EPISODE_MARGIN = 5
RESULTS = "results_overlap"
