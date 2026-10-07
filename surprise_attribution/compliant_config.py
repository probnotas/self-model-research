"""Grid for the compliant-body factorial (run_compliant.py).

SIMULATED compliance (series-elastic + hysteresis + viscous damping + sensor
noise; see compliant_env.py). A model of a soft joint, not real hardware.

Every configuration runs the SAME factorial trials as run_factorial.py (same
master seed, same draws, same start states), so rigid vs compliant is a paired
comparison. The rule is alarm_classifier.decide, unchanged; the body/world
threshold is recalibrated per configuration by the rule's own procedure (50
separate undisturbed episodes on that body); the alarm threshold stays frozen
at the value chosen on the rigid dev set.
"""
# stiffness (N*m/rad); None = rigid Pendulum-v1. Levels chosen only so the
# controller can balance (pilot: 6/6 upright at each), before any attribution run.
STIFFNESS = {"rigid": None, "stiff": 200.0, "medium": 30.0, "soft": 4.0}
# observation noise: (std of angle in rad, std of angular velocity in rad/s)
NOISE = {"none": (0.0, 0.0), "low": (0.002, 0.02), "medium": (0.01, 0.1)}

N_WORKERS = 4
RESULTS = "results_compliant"
