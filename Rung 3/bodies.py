"""Parameterised pendulum bodies for Rung 3.

Rung 2 could only perturb mass, because it modified the stock environment's
attributes in place. Meta-learning needs a *distribution* of bodies, so this
subclasses PendulumEnv and reimplements step() with four physical parameters:

    m             pendulum mass          (stock 1.0)
    l             pendulum length        (stock 1.0)
    damping       linear viscous drag    (stock 0.0 — absent from the stock env)
    torque_scale  actuator gain          (stock 1.0)

With the stock values the dynamics are bit-identical to Pendulum-v1; this is
checked by `verify_equivalence()` and asserted at import time in the driver.
"""
import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control.pendulum import PendulumEnv, angle_normalize

# Ranges the meta-learner is trained over. Deliberately NARROWER than the test
# damages below, so every evaluated body is an extrapolation rather than an
# interpolation — see METRICS note in the driver.
META_RANGES = {
    "m":            (0.7, 1.3),
    "l":            (0.7, 1.3),
    "damping":      (0.0, 0.2),
    "torque_scale": (0.7, 1.3),
}

# Test bodies. All four sit outside META_RANGES on the axis they perturb.
BODIES = {
    "healthy":  dict(),
    "mass":     dict(m=1.5),             # the Rung 2 damage
    "length":   dict(l=1.5),
    "damping":  dict(damping=0.5),
}


class ParamPendulum(PendulumEnv):
    """Pendulum with settable physical parameters and viscous damping."""

    def __init__(self, m=1.0, l=1.0, damping=0.0, torque_scale=1.0,
                 render_mode=None, g=10.0):
        super().__init__(render_mode=render_mode, g=g)
        self.m = m
        self.l = l
        self.damping = damping
        self.torque_scale = torque_scale

    def step(self, u):
        th, thdot = self.state
        g, m, l, dt = self.g, self.m, self.l, self.dt

        # Clip first (actuator limit), then apply gain: a weaker actuator still
        # saturates at the same commanded value, it just delivers less torque.
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)
        u_eff = u * self.torque_scale

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th)
                            + 3.0 / (m * l**2) * u_eff) * dt
        newthdot = newthdot - self.damping * thdot * dt      # viscous drag
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, False, {}


def make_body(params=None, max_episode_steps=200):
    """Build a TimeLimit-wrapped ParamPendulum from a params dict."""
    params = params or {}
    env = ParamPendulum(**params)
    return gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)


def body_fn(name_or_params):
    """An env factory (zero-arg callable) for se.evaluate's env_fn hook."""
    p = BODIES[name_or_params] if isinstance(name_or_params, str) else name_or_params
    return lambda: make_body(p)


def sample_task(rng):
    """Draw one body uniformly from the meta-training distribution."""
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in META_RANGES.items()}


def verify_equivalence(n=400, seed=0):
    """ParamPendulum at stock parameters must match Pendulum-v1 exactly.

    Everything downstream assumes this; if it drifts, every comparison against
    the Rung 1/2 numbers becomes invalid.
    """
    a = gym.make("Pendulum-v1")
    b = make_body({})
    obs_a, _ = a.reset(seed=seed)
    obs_b, _ = b.reset(seed=seed)
    assert np.allclose(obs_a, obs_b), "reset states differ"
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
    print(f"max |stock - ParamPendulum| over 400 steps: {w:.3e}")
    print("PASS" if w == 0.0 else ("CLOSE" if w < 1e-12 else "FAIL"))
    rng = np.random.default_rng(0)
    print("\nsample meta-training tasks:")
    for _ in range(3):
        t = sample_task(rng)
        print("  " + "  ".join(f"{k}={v:.3f}" for k, v in t.items()))
    print("\ntest bodies (all outside the meta ranges on their axis):")
    for k, v in BODIES.items():
        print(f"  {k:>8}: {v if v else 'stock'}")


def collect(env_fn, n, seed):
    """Random-action transitions from a body. Mirrors Rung 1's collect_data."""
    env = env_fn()
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
    return np.array(S), np.array(A), np.array(S2)
