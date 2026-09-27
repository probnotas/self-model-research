"""Environment: stock Gymnasium Pendulum-v1, plus the two ways to disturb it.

The whole experiment rests on injecting the two kinds of surprise through two
DIFFERENT channels, so it is worth being precise about both:

  body change   A physical parameter of the pendulum (length, mass, gravity) is
                permanently altered. We do this by overwriting the attribute that
                Pendulum-v1's own step() reads - e.g. env.unwrapped.l - so the
                stock physics code keeps running, just on a different body.

  world event   An external torque is added to the equation of motion for a few
                steps, then set back to zero. It is added AFTER the controller's
                torque has been clipped, and it never appears in the action that
                the forward model is given. From the model's point of view it is
                an invisible hand.

Why a wrapper is needed at all: Pendulum-v1 has no "external torque" input.
The only torque it knows about is the action, and anything we put there would
(1) be clipped to +/-2 and (2) be visible to the forward model as part of its
input, which would make it a control, not a disturbance. So during a push this
wrapper evaluates Pendulum-v1's equation of motion itself, with one extra term.
Whenever the external torque is zero it simply calls the stock step(), so the
baseline and body-change conditions run the literal Gymnasium code.
verify_physics() checks that our re-implementation matches the stock code
exactly when the extra term is zero.
"""
import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control.pendulum import angle_normalize

import config


class DisturbablePendulum(gym.Wrapper):
    """Pendulum-v1 with a permanent-body-change hook and an external-torque hook."""

    def __init__(self, env):
        super().__init__(env)
        # External torque currently acting on the pendulum (N*m). 0 = none.
        self.ext_torque = 0.0

    # -- the two disturbance channels ----------------------------------------

    def change_body(self, factors):
        """Permanently scale physical parameters, e.g. {"l": 1.5} = length +50%.

        Pendulum-v1's step() reads self.g, self.m and self.l on every call, so
        overwriting them changes the physics from the next step on and forever
        after. The forward model is NOT told; it still believes in the old body.
        """
        base = self.env.unwrapped
        for name, factor in factors.items():
            if name not in ("g", "m", "l"):
                raise ValueError(f"unknown body parameter {name!r}; use g, m or l")
            setattr(base, name, getattr(base, name) * factor)

    def set_external_torque(self, torque):
        """Start (torque != 0) or stop (torque == 0) an external push."""
        self.ext_torque = float(torque)

    # -- stepping --------------------------------------------------------------

    def step(self, action):
        if self.ext_torque == 0.0:
            # No push: run the unmodified Gymnasium code path.
            return self.env.step(action)
        # Push active: evaluate the same physics with the extra torque term.
        return physics_step(self.env.unwrapped, action, self.ext_torque)


def physics_step(base, action, ext_torque):
    """Pendulum-v1's step(), line for line, plus an external torque term.

    Stock equation of motion (theta = 0 is upright):
        thdot' = thdot + (3g/(2l) sin(th) + 3/(m l^2) * u) * dt
    Ours:
        thdot' = thdot + (3g/(2l) sin(th) + 3/(m l^2) * (u + ext_torque)) * dt

    The external torque enters exactly where a torque physically would (same
    lever arm, same inertia), but it is added after the controller's torque is
    clipped and is not part of the action - so it is invisible to the model.
    """
    th, thdot = base.state
    g, m, l, dt = base.g, base.m, base.l, base.dt

    u = np.clip(action, -base.max_torque, base.max_torque)[0]
    base.last_u = u
    costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

    newthdot = thdot + (3 * g / (2 * l) * np.sin(th)
                        + 3.0 / (m * l**2) * (u + ext_torque)) * dt
    newthdot = np.clip(newthdot, -base.max_speed, base.max_speed)
    newth = th + newthdot * dt

    base.state = np.array([newth, newthdot])
    return base._get_obs(), -costs, False, False, {}


def make_env(max_steps=None):
    """A fresh, healthy Pendulum-v1 wrapped so it can be disturbed.

    Each condition gets its own env so a body change can never leak from one
    episode into the next.
    """
    env = gym.make("Pendulum-v1", max_episode_steps=max_steps or config.EPISODE_STEPS)
    return DisturbablePendulum(env)


def reset_env(env, seed):
    """Reset and return the first observation [cos th, sin th, thdot].

    START_MODE "upright" overwrites the stock random start with one near the
    top, so the controller is already balancing when the disturbance arrives.
    """
    obs, _ = env.reset(seed=seed)
    if config.START_MODE == "upright":
        rng = np.random.default_rng(seed)
        r = config.START_RANGE
        env.unwrapped.state = np.array([rng.uniform(-r, r), rng.uniform(-r, r)])
        obs = env.unwrapped._get_obs()
    return obs


def verify_physics(n=500, seed=0):
    """Check physics_step(..., ext_torque=0) reproduces stock Pendulum-v1 exactly.

    Two identical envs, the same random actions; one steps with Gymnasium's
    code, the other with ours. Returns the largest difference seen in any
    observation or reward. It should be exactly 0.
    """
    a = gym.make("Pendulum-v1").unwrapped
    b = gym.make("Pendulum-v1").unwrapped
    a.reset(seed=seed); b.reset(seed=seed)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        u = rng.uniform(-3, 3, size=1).astype(np.float32)   # includes out-of-range
        oa, ra, *_ = a.step(u)
        ob, rb, *_ = physics_step(b, u, 0.0)
        worst = max(worst, float(np.abs(oa - ob).max()), abs(float(ra - rb)))
    return worst


if __name__ == "__main__":
    w = verify_physics()
    print(f"max |stock Pendulum-v1 - physics_step(ext=0)| over 500 steps: {w:.3e}")
    print("PASS" if w == 0.0 else "FAIL")
