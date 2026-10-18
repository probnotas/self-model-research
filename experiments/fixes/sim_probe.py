"""Episodes with an active self-test ("probe") after the decision window.

Same physics, models, controller and loop as surprise_attribution/run_history.record_episode
up to step onset+200. Then the controller is switched off:
  SETTLE steps of zero command (let residual swing die down), then
  PROBE steps of a known command u = AMP * sin(2*pi*FREQ*t).
Hanging-calibration episodes: a healthy pendulum started hanging (theta = pi +/- 0.3,
theta_dot +/- 0.5), then the same settle + probe. SIMULATED body, not hardware.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SA = os.path.abspath(os.path.join(HERE, "..", "..", "surprise_attribution"))
sys.path.insert(0, SA)
import compliant_env as ce  # noqa: E402
import config  # noqa: E402
import model as model_lib  # noqa: E402
import planner  # noqa: E402

DT = 0.05
SETTLE, PROBE = 40, 60            # 2 s, 3 s
AMP, FREQ = 1.0, 1.5              # N*m, Hz
LOCK_SKIP = 20                    # skip the first 1 s of the probe (transient)


def probe_cmd(k):
    return np.array([AMP * np.sin(2 * np.pi * FREQ * k * DT)], dtype=np.float32)


def run(spec):
    """Controlled episode to onset+200, then settle + probe. Returns obs (T+1,3), act (T,1)."""
    cfg = spec["cfg"]
    T_ctrl = spec["onset"] + 200
    env = ce.make(cfg, max_steps=T_ctrl + SETTLE + PROBE + 5, noise_seed=spec["seed"] + 1_000_000)
    env.reset(seed=spec["seed"])
    m = _model(spec["model_path"])
    if spec.get("hanging"):
        rng = np.random.default_rng(spec["seed"])
        env.unwrapped.state = np.array([np.pi + rng.uniform(-0.3, 0.3), rng.uniform(-0.5, 0.5)])
        T_ctrl = 0
    elif config.START_MODE == "upright":
        rng = np.random.default_rng(spec["seed"]); r = config.START_RANGE
        env.unwrapped.state = np.array([rng.uniform(-r, r), rng.uniform(-r, r)])
    obs = env.observe()
    np.random.seed(spec["seed"])
    O, U = [obs], []
    for t in range(T_ctrl + SETTLE + PROBE):
        if not spec.get("hanging"):
            if spec["has_body"] and t == spec["body_onset"]:
                env.change_body({spec["param"]: spec["factor"]})
            if spec["has_push"]:
                if t == spec["push_onset"]:
                    env.set_external_torque(spec["torque"])
                if t == spec["push_onset"] + spec["duration"]:
                    env.set_external_torque(0.0)
        if t < T_ctrl:
            action = planner.choose_action(m, obs)
        elif t < T_ctrl + SETTLE:
            action = np.zeros(1, dtype=np.float32)
        else:
            action = probe_cmd(t - T_ctrl - SETTLE)
        obs, *_ = env.step(action)
        O.append(obs); U.append(action)
    env.close()
    out = dict(spec); out.pop("cfg")
    out["obs"], out["act"], out["T_ctrl"] = np.array(O, np.float32), np.array(U, np.float32), T_ctrl
    return out


_M = {}


def _model(p):
    if p not in _M:
        _M[p] = model_lib.load_model(p)
    return _M[p]


def _init():
    torch.set_num_threads(1)


def probe_amplitude(obs, T_ctrl):
    """Lock-in amplitude of theta_dot at the probe frequency, over the probe minus its first second."""
    s = T_ctrl + SETTLE + LOCK_SKIP
    w = obs[s + 1:T_ctrl + SETTLE + PROBE + 1, 2].astype(np.float64)
    k = np.arange(LOCK_SKIP, PROBE)
    ph = 2 * np.pi * FREQ * k * DT
    a, b = 2 * np.mean(w * np.sin(ph)), 2 * np.mean(w * np.cos(ph))
    return float(np.hypot(a, b))


def run_all(specs, workers=4):
    import multiprocessing as mpc
    res = []
    with mpc.get_context("fork").Pool(workers, initializer=_init) as pool:
        for i, r in enumerate(pool.imap(run, specs, chunksize=2), 1):
            res.append(r)
            if i % 100 == 0 or i == len(specs):
                print(f"    {i}/{len(specs)}", flush=True)
    return res
