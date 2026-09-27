"""MPC controller: Cross-Entropy Method planning through the forward model.

The controller is not the object of study here - it just gives the pendulum a
purposeful, closed-loop trajectory to follow while we watch the model's
surprise. It uses the SAME frozen forward model whose errors we are logging,
exactly as a self-model would be used by a real robot.

At each step:
  1. Keep a Gaussian over HORIZON-step torque sequences (start: mean 0, std 1).
  2. Sample CEM_CANDIDATES sequences, clip them to the torque limit [-2, 2].
  3. Roll every sequence forward through the model (all in one batch) and
     score it: summed squared distance from upright-and-still [1, 0, 0].
  4. Refit the Gaussian to the CEM_ELITES cheapest sequences; repeat CEM_ITERS
     times.
  5. Execute only the first torque of the final mean sequence (receding
     horizon), then replan from the new real state next step.

Randomness comes from numpy's global RNG, which run_experiment seeds at the
start of each episode, so the controller's choices are reproducible.
"""
import numpy as np
import torch

import config

ACTION_LOW, ACTION_HIGH = -2.0, 2.0
_TARGET = torch.tensor([1.0, 0.0, 0.0])   # cos th = 1, sin th = 0, thdot = 0


def rollout_costs(model, state, acts):
    """Summed cost of each candidate sequence under the model.

    acts: (n_candidates, HORIZON, 1) float32. Returns (n_candidates,).
    """
    n = acts.shape[0]
    sim = torch.from_numpy(np.tile(np.asarray(state, dtype=np.float32), (n, 1)))
    acts_t = torch.from_numpy(acts)
    total = torch.zeros(n)
    with torch.no_grad():
        for h in range(acts.shape[1]):
            sim = model.predict_next(sim, acts_t[:, h, :])
            total += ((sim - _TARGET) ** 2).sum(dim=1)
    return total.numpy()


def choose_action(model, state):
    """One CEM-MPC decision: returns a torque as a float32 array of shape (1,)."""
    H = config.HORIZON
    mean = np.zeros((H, 1), dtype=np.float32)
    std = np.full((H, 1), config.CEM_INIT_STD, dtype=np.float32)
    for _ in range(config.CEM_ITERS):
        noise = np.random.randn(config.CEM_CANDIDATES, H, 1).astype(np.float32)
        acts = np.clip(mean + std * noise, ACTION_LOW, ACTION_HIGH).astype(np.float32)
        costs = rollout_costs(model, state, acts)
        elites = acts[np.argsort(costs)[:config.CEM_ELITES]]
        mean, std = elites.mean(axis=0), elites.std(axis=0)
    return np.clip(mean[0], ACTION_LOW, ACTION_HIGH).astype(np.float32)
