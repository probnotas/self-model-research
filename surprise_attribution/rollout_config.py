"""Multi-step (rollout) prediction error as the surprise observable (run_rollout.py).

SIMULATED compliant body, same as results_compliant/: a model of compliance,
not real hardware.

Fully offline: the recorded episodes in results_history/episodes/ (observed
states + the actions actually taken) and the existing single-state forward
models (forward_model.pt for rigid_none, compliant_models/*.pt otherwise) are
reused. Nothing is simulated or trained. The only change is the error metric.

THE ROLLOUT (fixed before any result was looked at)
  From a start step t, seed the model with the real observed state s_t, then run
  it free: s^_{t+h+1} = f(s^_{t+h}, a_{t+h}) with the RECORDED actions, feeding
  its own prediction back in (no re-anchoring, no normalisation - the same way
  the CEM planner rolls the model forward). Rollout error over horizon H:
      e_H(t) = mean_{h=1..H} || s^_{t+h} - s_{t+h} ||
  H = 1 is exactly the one-step error, so H = 1 is the "old" observable measured
  in the same windows.

WHERE THE ROLLOUTS START
  START_LAG steps after the onset the rule is told, so every push is already
  over (pushes last <= 10 steps and begin <= 5 steps after onset). A rollout
  started before a push ends never sees the push and stays displaced forever,
  which would make pushes look permanently surprising for reasons that have
  nothing to do with the body. Starting after the push is the hypothesis's own
  premise: "once a push ends, the dynamics are the original ones".
  N_STARTS rollouts, START_STEP apart, are averaged per trial to reduce variance.
  (Last start + H_MAX stays inside every episode: onset <= 85, 85+20+70+100 <= 325.)

DECISION POINT (healthy data only)
  Per body and per H: threshold = mean + Z * sd of the score on the 50 healthy
  undisturbed calibration episodes. score >= threshold -> body, else world.

DIVERGENCE CHECK (the known risk of free-running rollouts)
  On the same healthy episodes, compare the model rollout with a trivial
  "nothing changes" predictor (s^_{t+h} = s_t). Where the model's rollout error
  exceeds the trivial predictor's, the rollout carries less information about
  the motion than doing nothing: flagged as diverged.

H_MAIN is fixed in advance as the primary horizon; the best H in the sweep is
also reported but is chosen after seeing the labels, so it is optimistic.
"""
H_SWEEP = (1, 5, 10, 20, 40, 100)
H_MAIN = 40
START_LAG = 20
N_STARTS = 8
START_STEP = 10
Z = 2.0
RESULTS = "results_rollout"
EPISODES = "results_history/episodes"
