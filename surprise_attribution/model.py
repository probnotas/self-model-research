"""The forward model ("self-model"): state + action -> predicted next state.

    input   [cos th, sin th, thdot, torque]    (4 numbers)
    output  predicted [cos th', sin th', thdot']   (3 numbers)

It is a plain MLP (4 -> 64 -> 64 -> 3, ReLU), trained with MSE on transitions
collected from the HEALTHY pendulum under random actions, then FROZEN. It is
never updated during the experiment. That matters: we are measuring how
surprised a fixed self-model is, not how fast it adapts.

------------------------------------------------------------------------------
SWAPPING IN YOUR OWN MODEL
------------------------------------------------------------------------------
The rest of the code touches the model through exactly one method:

    model.predict_next(states, actions) -> next_states

  states   torch.float32 tensor, shape (N, 3)   [cos th, sin th, thdot]
  actions  torch.float32 tensor, shape (N, 1)   torque
  returns  torch.float32 tensor, shape (N, 3)   predicted next state
           (in STATE space - if your net predicts deltas, add the state back
           inside this method, as ForwardModel does below)

It is called with N = 1 for the surprise signal and N = CEM_CANDIDATES for
planning. To use your model: make it expose predict_next() (a thin wrapper
class is fine), and return it from load_model() below. Nothing else changes.
------------------------------------------------------------------------------
"""
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import config


class ForwardModel(nn.Module):
    def __init__(self, hidden=config.HIDDEN, predict_delta=config.PREDICT_DELTA):
        super().__init__()
        self.predict_delta = predict_delta
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, states, actions):
        """Raw network output: the delta (if predict_delta) or the next state."""
        return self.net(torch.cat([states, actions], dim=1))

    def predict_next(self, states, actions):
        """Predicted next state, always in state space (see module docstring)."""
        out = self.forward(states, actions)
        return states + out if self.predict_delta else out


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_data(n, seed):
    """n transitions (s, a, s') from the stock, healthy Pendulum-v1.

    Actions are uniform random over [-2, 2]. Random actions visit a broad slice
    of the state space (hanging, swinging, near the top), so the model is not
    only accurate along the one trajectory the controller happens to follow.
    Episodes reset every 200 steps (Pendulum-v1's own time limit), which keeps
    re-sampling the starting angle.
    """
    env = gym.make("Pendulum-v1")
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    S, A, S2 = [], [], []
    for _ in range(n):
        a = env.action_space.sample()
        obs2, _, term, trunc, _ = env.step(a)
        S.append(obs); A.append(a); S2.append(obs2)
        obs = obs2
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    return (np.array(S, dtype=np.float32), np.array(A, dtype=np.float32),
            np.array(S2, dtype=np.float32))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(S, A, S2, seed=config.SEED, epochs=config.TRAIN_EPOCHS):
    """Fit the MLP with MSE, full-batch Adam.

    Full batch (all 2,000 transitions at once) is fine at this size and removes
    mini-batch ordering as a source of run-to-run variation.
    The loss is on whatever the network outputs: deltas if PREDICT_DELTA, else
    next states.
    """
    torch.manual_seed(seed)
    model = ForwardModel()
    s, a, s2 = map(torch.from_numpy, (S, A, S2))
    target = s2 - s if model.predict_delta else s2
    opt = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        loss = loss_fn(model(s, a), target)
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 250 == 0 or epoch == epochs - 1:
            print(f"    epoch {epoch:>4}  train MSE {loss.item():.3e}")
    model.eval()
    return model


def held_out_error(model, seed):
    """Mean one-step prediction error on fresh healthy-body transitions.

    Uses the same error definition as the experiment (Euclidean norm in state
    space), so you can see what "normal" surprise looks like on random data.
    """
    S, A, S2 = collect_data(1000, seed=seed)
    with torch.no_grad():
        pred = model.predict_next(torch.from_numpy(S), torch.from_numpy(A)).numpy()
    return float(np.linalg.norm(pred - S2, axis=1).mean())


# ---------------------------------------------------------------------------
# Save / load  (load_model is the swap point for your own model)
# ---------------------------------------------------------------------------

def save_model(model, path):
    torch.save({"state_dict": model.state_dict(),
                "predict_delta": model.predict_delta,
                "hidden": model.net[0].out_features}, path)


def load_model(path):
    """Return a frozen model exposing predict_next(). Replace this to use yours."""
    ckpt = torch.load(path)
    model = ForwardModel(hidden=ckpt["hidden"], predict_delta=ckpt["predict_delta"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def get_model(path, retrain=False):
    """Load the cached model, or train one on the healthy body and cache it."""
    if os.path.exists(path) and not retrain:
        print(f"  loading forward model from {path}")
        return load_model(path)
    print(f"  training forward model on {config.TRAIN_TRANSITIONS} "
          f"healthy-body transitions")
    S, A, S2 = collect_data(config.TRAIN_TRANSITIONS, seed=config.SEED)
    model = train_model(S, A, S2)
    save_model(model, path)
    print(f"  saved to {path}")
    return model
