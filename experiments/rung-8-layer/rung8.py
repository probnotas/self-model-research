"""Shared pieces for the Rung 8 four-way comparison (importable; no argparse).

SIMULATION ONLY. Everything reuses surprise_attribution/ unchanged:
  - body: compliant_env.CompliantPendulum, "soft" (k = 4), noise "none"
  - self-model AND the policy's planning model: compliant_models/soft_none.pt (not retrained)
  - policy: planner.choose_action (CEM-MPC through that model)

PerturbedCompliant re-states CompliantPendulum.step with each constant exposed as an
instance attribute, so the 7 perturbations can be switched on mid-episode. With no
perturbation it is bit-identical to CompliantPendulum (verify_identical() checks this).
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SA = os.path.join(HERE, "..", "..", "surprise_attribution")
sys.path.insert(0, SA)
import compliant_config as kc  # noqa: E402
import compliant_env as ce  # noqa: E402
import compliant_trials as ct  # noqa: E402
import config  # noqa: E402
import planner  # noqa: E402

NAME = "soft_none"
CFG = dict(stiffness=kc.STIFFNESS["soft"], noise=kc.NOISE["none"])
T_MAX, ONSET = 500, 50
TOL, HOLD = np.deg2rad(10.0), 50          # success: |theta| <= 10 deg for 50 consecutive steps
W = 20                                    # online window (steps) for the self-model's gain / error
PERTURBATIONS = {
    "healthy": {},
    "payload_add": {"m_mult": 1.5},
    "spring_soft": {"k_mult": 0.5},
    "spring_stiff": {"k_mult": 2.0},
    "damping_high": {"b_joint_mult": 3.0},
    "motor_weak": {"torque_mult": 0.6},
    "gravity_tilt": {"tilt": np.deg2rad(45.0)},
    "sensor_delay": {"delay": 2},
}


class PerturbedCompliant(ce.CompliantPendulum):
    """CompliantPendulum with switchable perturbations (applied by perturb())."""

    def __init__(self, env, stiffness, noise=(0.0, 0.0), noise_seed=0):
        super().__init__(env, stiffness, noise, noise_seed)
        self.k0 = stiffness
        self.b_joint, self.torque_mult, self.tilt, self.delay = ce.B_JOINT, 1.0, 0.0, 0
        self._buf = []

    def perturb(self, p):
        base = self.env.unwrapped
        if "m_mult" in p:
            self.change_body({"m": p["m_mult"]})            # the main experiment's own hook
        self.k = self.k0 * p.get("k_mult", 1.0)
        self.b_joint = ce.B_JOINT * p.get("b_joint_mult", 1.0)
        self.torque_mult = p.get("torque_mult", 1.0)
        self.tilt = p.get("tilt", 0.0)
        self.delay = p.get("delay", 0)
        _ = base

    def true_full_state(self):
        base = self.env.unwrapped
        th, w = base.state
        phi, phi_d = self.motor if self.motor is not None else (float(th), float(w))
        return np.array([th, w, phi, phi_d], dtype=np.float64)

    def step(self, action):
        # CompliantPendulum.step, line for line, with B_JOINT, the torque limit and gravity's
        # direction as attributes. tilt = 0 gives sin(th) exactly as in the original.
        base = self.env.unwrapped
        th, w = base.state
        if self.motor is None:
            self.motor = [float(th), float(w)]
        phi, phi_d = self.motor
        lim = base.max_torque * self.torque_mult
        u = float(np.clip(action, -lim, lim)[0])
        g, m, l = base.g, base.m, base.l
        h = base.dt / ce.N_SUBSTEPS
        for _ in range(ce.N_SUBSTEPS):
            d, d_dot = phi - th, phi_d - w
            tau_s = (self.k * d * (1.0 + ce.HYST * np.tanh(d * d_dot / ce.HYST_EPS))
                     + 2.0 * ce.SPRING_ZETA * np.sqrt(self.k * ce.J_MOTOR) * d_dot)
            phi_d += (u - tau_s - ce.B_MOTOR * phi_d) / ce.J_MOTOR * h
            phi += phi_d * h
            grav = np.sin(th) if self.tilt == 0.0 else np.sin(th - self.tilt)
            w += (3 * g / (2 * l) * grav + 3.0 / (m * l ** 2) * (tau_s + self.ext_torque)
                  - self.b_joint * w) * h
            w = float(np.clip(w, -base.max_speed, base.max_speed))
            th += w * h
        self.motor = [phi, phi_d]
        base.state = np.array([th, w])
        base.last_u = u
        obs = self._noisy(base._get_obs())
        self._buf.append(obs)
        self._buf = self._buf[-(self.delay + 1):]
        return self._buf[0] if self.delay else obs, 0.0, False, False, {}


def make_env(seed):
    import gymnasium as gym
    env = PerturbedCompliant(gym.make("Pendulum-v1", max_episode_steps=10 ** 6), CFG["stiffness"],
                             tuple(CFG["noise"]), noise_seed=seed + 1_000_000)
    env.reset(seed=seed)
    env.unwrapped.state = np.array([np.pi, 0.0])          # hanging straight down, at rest
    return env


def verify_identical(n=400, seed=0):
    """No perturbation: PerturbedCompliant == CompliantPendulum, step for step."""
    import gymnasium as gym
    a = ce.CompliantPendulum(gym.make("Pendulum-v1", max_episode_steps=10 ** 6), CFG["stiffness"])
    b = PerturbedCompliant(gym.make("Pendulum-v1", max_episode_steps=10 ** 6), CFG["stiffness"])
    a.reset(seed=seed); b.reset(seed=seed)
    rng = np.random.default_rng(seed); worst = 0.0
    for _ in range(n):
        u = rng.uniform(-2, 2, size=1).astype(np.float32)
        oa, *_ = a.step(u); ob, *_ = b.step(u)
        worst = max(worst, float(np.abs(oa - ob).max()))
    return worst


# ---------------------------------------------------------------- oracle (true-physics MPC)
def batched_step(x, u, p, base):
    """The same physics for many candidates at once. x: (n, 4) = th, w, phi, phi_d; u: (n,)."""
    th, w, phi, phi_d = x.T.copy()
    k = CFG["stiffness"] * p.get("k_mult", 1.0)
    bj = ce.B_JOINT * p.get("b_joint_mult", 1.0)
    m = base.m                                   # base.m already holds the true current mass (change_body)
    l, g, tilt = base.l, base.g, p.get("tilt", 0.0)
    u = np.clip(u, -base.max_torque * p.get("torque_mult", 1.0), base.max_torque * p.get("torque_mult", 1.0))
    h = base.dt / ce.N_SUBSTEPS
    for _ in range(ce.N_SUBSTEPS):
        d, d_dot = phi - th, phi_d - w
        tau_s = k * d * (1.0 + ce.HYST * np.tanh(d * d_dot / ce.HYST_EPS)) + 2.0 * ce.SPRING_ZETA * np.sqrt(k * ce.J_MOTOR) * d_dot
        phi_d = phi_d + (u - tau_s - ce.B_MOTOR * phi_d) / ce.J_MOTOR * h
        phi = phi + phi_d * h
        w = np.clip(w + (3 * g / (2 * l) * np.sin(th - tilt) + 3.0 / (m * l ** 2) * tau_s - bj * w) * h,
                    -base.max_speed, base.max_speed)
        th = th + w * h
    return np.stack([th, w, phi, phi_d], 1)


def oracle_action(x0, p, base):
    """planner.choose_action's CEM (same settings, same cost on [cos, sin, thdot]) but rolling
    out the TRUE physics with the TRUE post-change parameters from the TRUE full state."""
    H, n = config.HORIZON, config.CEM_CANDIDATES
    mean, std = np.zeros(H), np.full(H, config.CEM_INIT_STD)
    for _ in range(config.CEM_ITERS):
        acts = np.clip(mean + std * np.random.randn(n, H), planner.ACTION_LOW, planner.ACTION_HIGH)
        x = np.tile(x0, (n, 1)); cost = np.zeros(n)
        for t in range(H):
            x = batched_step(x, acts[:, t], p, base)
            cost += (np.cos(x[:, 0]) - 1) ** 2 + np.sin(x[:, 0]) ** 2 + x[:, 1] ** 2
        el = acts[np.argsort(cost)[:config.CEM_ELITES]]
        mean, std = el.mean(0), el.std(0)
    return np.array([np.clip(mean[0], planner.ACTION_LOW, planner.ACTION_HIGH)], dtype=np.float32)


# ---------------------------------------------------------------- self-model helpers
_MODEL = {}


def model():
    if "m" not in _MODEL:
        _MODEL["m"] = ct.get_model(NAME, CFG)[0]       # cached soft_none.pt, not retrained
    return _MODEL["m"]


def predict(s, u):
    with torch.no_grad():
        return model().predict_next(torch.from_numpy(np.asarray(s, np.float32))[None],
                                    torch.from_numpy(np.asarray(u, np.float32).reshape(1))[None])[0].numpy()


def success(theta):
    """Index where |theta| <= TOL first held for HOLD consecutive steps, else None."""
    ok = np.abs(np.arctan2(np.sin(theta), np.cos(theta))) <= TOL
    run = 0
    for t, o in enumerate(ok):
        run = run + 1 if o else 0
        if run >= HOLD:
            return t
    return None
