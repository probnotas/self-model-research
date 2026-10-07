"""Knobs for the 2x2 simultaneous-disturbance factorial (run_factorial.py).

Design: {body change, no body change} x {world push, no world push}.
  body change   mass x U[1.3, 1.7]  or  length x U[1.2, 1.5]   (half the body trials each)
  world push    |torque| in {1, 2, 4} N*m (random sign) x duration {short, medium}
  overlap       when both are present, the push starts within +/- OVERLAP_STEPS
                of the body change, so the two overlap
Every cell has N_PER_CELL trials; inside the world cells the 6 magnitude x
duration combinations are balanced exactly (N_PER_CELL / 6 each).

The rule under test is alarm_classifier.decide, unchanged:
  - body/world threshold: mean + 4 std of the settled score on N_CALIB
    SEPARATE undisturbed episodes (not the "neither" cell, so that cell is
    scored out-of-sample);
  - alarm threshold: frozen from results_alarm/chosen_alarm.json.
Nothing is tuned here, so there is no dev/test split: one run, fixed seed.

The body is the rigid Gymnasium Pendulum-v1, not a compliant body.
"""
MASTER_SEED = 90_000          # all trial seeds derive from this; logged in the output

N_PER_CELL = 120              # >= 100 required; divisible by 6 and by 2
N_CALIB = 50                  # separate undisturbed episodes for the body/world threshold

MASS_RANGE = (1.3, 1.7)
LENGTH_RANGE = (1.2, 1.5)
PUSH_TORQUES = (1.0, 2.0, 4.0)                    # N*m; 0 = the no-world cells
PUSH_DURATIONS = {"short": 4, "medium": 10}       # steps: 0.2 s, 0.5 s

BODY_ONSET_RANGE = (60, 80)   # steps (3-4 s): the controller is balancing by then
OVERLAP_STEPS = 5             # push onset = body onset + U[-5, +5] steps (+/- 0.25 s)

RESULTS = "results_factorial"
