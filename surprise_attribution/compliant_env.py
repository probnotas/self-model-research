"""A SIMULATED compliant pendulum: series-elastic joint + hysteresis + sensor noise.

This is a model of compliance, not real hardware. It exists to ask whether the
attribution rule's margin survives the things a real soft joint adds, before
anything is built.

Same interface as before: action = commanded torque u in [-2, 2]; observation =
[cos th, sin th, th_dot] of the LINK. Everything below is hidden from the robot.

Physics (per step of dt = 0.05 s, integrated in N_SUBSTEPS semi-implicit Euler
substeps because a stiff spring needs a small integration step):

    motor:   phi_dd = (u - tau_s - B_MOTOR * phi_d) / J_MOTOR
    spring:  tau_s  = k * d * (1 + HYST * tanh(d * d_dot / HYST_EPS)) + c * d_dot,  d = phi - th
             (c = SPRING_ZETA * critical damping: a viscoelastic element)
             (a) series-elastic: the link only feels torque through the spring
             (b) hysteresis: stiffer while the spring is loading (d, d_dot same sign),
                 softer while unloading, so the return path differs from the push path
    link:    th_dd = 3g/(2l) sin th + 3/(m l^2) (tau_s + tau_ext) - B_JOINT * th_dot
             (velocity-dependent damping on the joint; tau_ext = world push)
    (c) sensor noise: Gaussian noise on the observed angle and angular velocity.

Body changes (m, l) and the world push (tau_ext) use the same hooks as the rigid
env (change_body / set_external_torque), so trial schedules are unchanged.

stiffness=None gives the stock rigid Pendulum-v1 (plus optional noise); with
noise 0 it is bit-identical to the existing env, which verify_rigid() checks.
"""
import gymnasium as gym
import numpy as np

import env_setup

N_SUBSTEPS = 10
J_MOTOR = 0.05        # motor-side inertia (kg m^2, reflected)
B_MOTOR = 0.5         # motor-side viscous friction
B_JOINT = 0.3         # joint viscous damping (1/s)
HYST = 0.3            # +/-30 % stiffness change between loading and unloading
HYST_EPS = 1e-3       # smooths the loading/unloading switch
# Damper in parallel with the spring, as a fraction of critical damping
# (2 * sqrt(k * J_MOTOR)). Without it a stiff spring against a light motor rings
# at ~7 Hz, faster than the 20 Hz controller can act, and the joint cannot
# balance at all. Chosen ONLY on whether the joint can balance, before any
# attribution trial was run.
SPRING_ZETA = 0.3


class CompliantPendulum(env_setup.DisturbablePendulum):
    def __init__(self, env, stiffness=None, noise=(0.0, 0.0), noise_seed=0):
        super().__init__(env)
        self.k = stiffness                       # N*m/rad; None = rigid
        self.noise_th, self.noise_w = noise      # std of angle (rad), velocity (rad/s) noise
        self.rng = np.random.default_rng(noise_seed)
        self.motor = None                        # (phi, phi_dot); set lazily after reset

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.motor = None
        return self._noisy(obs), info

    def observe(self):
        """Noisy observation of the current link state (used after reset_env sets the state)."""
        return self._noisy(self.env.unwrapped._get_obs())

    def _noisy(self, obs):
        if self.noise_th == 0.0 and self.noise_w == 0.0:
            return obs
        th = np.arctan2(obs[1], obs[0]) + self.rng.normal(0.0, self.noise_th)
        w = obs[2] + self.rng.normal(0.0, self.noise_w)
        return np.array([np.cos(th), np.sin(th), w], dtype=np.float32)

    def step(self, action):
        if self.k is None:
            obs, r, term, trunc, info = super().step(action)
            return self._noisy(obs), r, term, trunc, info
        base = self.env.unwrapped
        th, w = base.state
        if self.motor is None:                   # spring starts relaxed
            self.motor = [float(th), float(w)]
        phi, phi_d = self.motor
        u = float(np.clip(action, -base.max_torque, base.max_torque)[0])
        g, m, l = base.g, base.m, base.l
        h = base.dt / N_SUBSTEPS
        for _ in range(N_SUBSTEPS):
            d, d_dot = phi - th, phi_d - w
            tau_s = (self.k * d * (1.0 + HYST * np.tanh(d * d_dot / HYST_EPS))
                     + 2.0 * SPRING_ZETA * np.sqrt(self.k * J_MOTOR) * d_dot)
            phi_d += (u - tau_s - B_MOTOR * phi_d) / J_MOTOR * h
            phi += phi_d * h
            w += (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * (tau_s + self.ext_torque)
                  - B_JOINT * w) * h
            w = float(np.clip(w, -base.max_speed, base.max_speed))
            th += w * h
        self.motor = [phi, phi_d]
        base.state = np.array([th, w])
        base.last_u = u
        # elapsed-step bookkeeping for the TimeLimit wrapper is skipped on purpose:
        # episodes here have a fixed length and never rely on truncation
        return self._noisy(base._get_obs()), 0.0, False, False, {}


def make(cfg, max_steps, noise_seed=0):
    """cfg: dict(stiffness=None|float, noise=(sd_theta, sd_omega))."""
    env = gym.make("Pendulum-v1", max_episode_steps=max_steps)
    return CompliantPendulum(env, cfg["stiffness"], tuple(cfg["noise"]), noise_seed)


def verify_rigid(n=300, seed=0):
    """stiffness=None, no noise: identical to the existing DisturbablePendulum."""
    a = env_setup.make_env(max_steps=n)
    b = make(dict(stiffness=None, noise=(0.0, 0.0)), n)
    a.reset(seed=seed); b.reset(seed=seed)
    rng = np.random.default_rng(seed); worst = 0.0
    for t in range(n):
        if t == 100:
            a.set_external_torque(1.5); b.set_external_torque(1.5)
        if t == 110:
            a.set_external_torque(0.0); b.set_external_torque(0.0)
        u = rng.uniform(-2, 2, size=1).astype(np.float32)
        oa, *_ = a.step(u); ob, *_ = b.step(u)
        worst = max(worst, float(np.abs(oa - ob).max()))
    return worst
