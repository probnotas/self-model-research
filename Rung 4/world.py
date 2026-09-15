"""A pendulum whose body can change AND whose world can shove it.

Rungs 1-3 could only change the body. Telling "my body changed" apart from
"something pushed me" needs both, and they have to be injected through
different channels:

  body change   a physical parameter is permanently altered (m, torque_scale)
  world event   an external torque is added for a few steps, then removed

The external torque is added AFTER the actuator gain and is never part of the
commanded action, so the forward model - which sees only (state, action) - has
no way to predict it. That is what makes it a disturbance rather than a control.
"""
import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control.pendulum import angle_normalize

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Rung 3"))
from bodies import ParamPendulum


class WorldPendulum(ParamPendulum):
    """ParamPendulum plus an external disturbance torque."""

    def __init__(self, ext_torque=0.0, **kw):
        super().__init__(**kw)
        self.ext_torque = ext_torque

    def step(self, u):
        th, thdot = self.state
        g, m, l, dt = self.g, self.m, self.l, self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)
        u_eff = u * self.torque_scale + self.ext_torque      # the world's contribution

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th)
                            + 3.0 / (m * l**2) * u_eff) * dt
        newthdot = newthdot - self.damping * thdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, False, {}


def make(params=None, max_episode_steps=300):
    env = WorldPendulum(**(params or {}))
    return gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)


def verify_equivalence(n=400, seed=0):
    """With no disturbance and stock parameters this must match Pendulum-v1 exactly."""
    a = gym.make("Pendulum-v1")
    b = make({})
    oa, _ = a.reset(seed=seed); ob, _ = b.reset(seed=seed)
    assert np.allclose(oa, ob), "reset states differ"
    a.action_space.seed(seed)
    worst = 0.0
    for _ in range(n):
        u = a.action_space.sample()
        oa, ra, ta, tra, _ = a.step(u)
        ob, rb, tb, trb, _ = b.step(u)
        worst = max(worst, float(np.abs(oa - ob).max()), abs(float(ra - rb)))
        if ta or tra:
            oa, _ = a.reset(); ob, _ = b.reset()
    a.close(); b.close()
    return worst


if __name__ == "__main__":
    w = verify_equivalence()
    print(f"max |Pendulum-v1 - WorldPendulum| over 400 steps: {w:.3e}")
    print("PASS" if w == 0.0 else ("CLOSE" if w < 1e-12 else "FAIL"))
