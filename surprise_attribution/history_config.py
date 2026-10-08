"""History-input forward model as the surprise signal (run_history.py).

SIMULATED compliant body, same as results_compliant/: a model of compliance,
not real hardware.

What changes versus results_compliant/: ONLY the model whose prediction error
the attribution rule reads. The new model sees the last K observed states and
K actions instead of one:

    input   [s_{t-K+1}, ..., s_t, a_{t-K+1}, ..., a_t]    (3K + K numbers)
    output  s_{t+1} - s_t                                  (3 numbers)

Same MLP widths (64, 64), same full-batch Adam, same epochs, same learning rate,
same training seed, and the same 2000 random-action transitions from that body
(compliant_trials.collect, same seed). Training windows never straddle a reset
(the data resets every 200 steps), so the first K-1 transitions of each
200-step segment are not used as targets: 2000 - 10*(K-1) training samples.

Held fixed so the comparison is paired:
  * the CONTROLLER: the pendulum is still driven by the single-state model, as
    in results_compliant/, so every trajectory, every observation and the
    calm/settle gating are identical. Only the error signal changes.
  * the RULE: alarm_classifier.decide, with its own per-body calibration of the
    body/world threshold (mean + 4 sd of settled scores on the 50 undisturbed
    calibration episodes) and the alarm threshold frozen at 0.04042 - exactly
    the procedure run_compliant.py used.

K values: K_MAIN is the primary; the others are the requested sweep.
"""
K_MAIN = 4
K_SWEEP = (2, 4, 8)
RESULTS = "results_history"
MODEL_DIR = "history_models"
EPISODES = "episodes"          # recorded obs/actions per config (regenerable, not committed)
TRAIN_THREADS = 1              # torch threads while training (determinism)
