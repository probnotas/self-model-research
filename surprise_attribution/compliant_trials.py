"""Models and episodes for the compliant-body experiment.

For every configuration (stiffness, noise):
  1. collect 2000 random-action transitions from THAT body, as the robot senses it
     (noisy observations if noise is on), with the same procedure as the
     original rigid model (random torques, reset every 200 steps);
  2. train the same MLP with the same seed and settings (model.train_model);
  3. report held-out one-step error on 1000 fresh transitions from that body.

Episodes use the same loop as overlap_trials.run_overlap_episode, with one
deliberate difference: the angle and angular velocity handed to the rule are
the OBSERVED (noisy) ones, not the simulator's true state. A real robot only
has its sensors; judging "calm" from true state would hide the effect of noise.
With noise 0 the two are identical.
"""
import os

import numpy as np
import torch

import compliant_env as ce
import config
import model as model_lib
import planner

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "compliant_models")
TRAIN_N, HELDOUT_N = config.TRAIN_TRANSITIONS, 1000


def collect(cfg, n, seed):
    """Random-action transitions (s, a, s') as the robot observes them."""
    env = ce.make(cfg, max_steps=10_000, noise_seed=seed + 7)
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    S, A, S2 = [], [], []
    for i in range(n):
        a = env.action_space.sample()
        obs2, *_ = env.step(a)
        S.append(obs); A.append(a); S2.append(obs2)
        obs = obs2
        if (i + 1) % 200 == 0:                    # same cadence as Pendulum-v1's time limit
            obs, _ = env.reset()
    return (np.array(S, dtype=np.float32), np.array(A, dtype=np.float32),
            np.array(S2, dtype=np.float32))


def get_model(name, cfg):
    """Train (or load cached) the forward model for one configuration."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"{name}.pt")
    if not os.path.exists(path):
        torch.manual_seed(config.SEED)
        S, A, S2 = collect(cfg, TRAIN_N, config.SEED)
        m = model_lib.train_model(S, A, S2, seed=config.SEED)
        model_lib.save_model(m, path)
    return model_lib.load_model(path), path


def heldout_error(m, cfg):
    S, A, S2 = collect(cfg, HELDOUT_N, config.SEED + 1)
    with torch.no_grad():
        pred = m.predict_next(torch.from_numpy(S), torch.from_numpy(A)).numpy()
    return float(np.linalg.norm(pred - S2, axis=1).mean())


def run_episode(m, cfg, spec, steps):
    """Same loop as the rigid experiments; returns error and OBSERVED angle/velocity."""
    env = ce.make(cfg, max_steps=steps, noise_seed=spec["seed"] + 1_000_000)
    env.reset(seed=spec["seed"])
    import env_setup
    if config.START_MODE == "upright":            # same start state as env_setup.reset_env
        rng = np.random.default_rng(spec["seed"]); r = config.START_RANGE
        env.unwrapped.state = np.array([rng.uniform(-r, r), rng.uniform(-r, r)])
    obs = env.observe()
    np.random.seed(spec["seed"])
    err, th, thd = [], [], []
    for t in range(steps):
        if spec["has_body"] and t == spec["body_onset"]:
            env.change_body({spec["param"]: spec["factor"]})
        if spec["has_push"]:
            if t == spec["push_onset"]:
                env.set_external_torque(spec["torque"])
            if t == spec["push_onset"] + spec["duration"]:
                env.set_external_torque(0.0)
        action = planner.choose_action(m, obs)
        with torch.no_grad():
            pred = m.predict_next(torch.from_numpy(obs.astype(np.float32))[None],
                                  torch.from_numpy(action)[None])[0].numpy()
        nxt, *_ = env.step(action)
        err.append(float(np.linalg.norm(nxt - pred)))
        th.append(float(np.arctan2(nxt[1], nxt[0])))   # observed angle, wrapped
        thd.append(float(nxt[2]))                       # observed angular velocity
        obs = nxt
    env.close()
    return np.array(err, np.float32), np.array(th, np.float32), np.array(thd, np.float32)


_MODELS = {}


def _init_worker():
    torch.set_num_threads(1)


def run_trial(spec):
    """Worker entry: spec carries cfg, model path and episode length."""
    if spec["model_path"] not in _MODELS:
        _MODELS[spec["model_path"]] = model_lib.load_model(spec["model_path"])
    out = dict(spec)
    out["err"], out["theta"], out["thdot"] = run_episode(
        _MODELS[spec["model_path"]], spec["cfg"], spec, spec["steps"])
    return out


def run_all(specs, workers=4, label=""):
    import multiprocessing as mp
    out = []
    with mp.get_context("fork").Pool(workers, initializer=_init_worker) as pool:
        for i, r in enumerate(pool.imap(run_trial, specs, chunksize=2), 1):
            out.append(r)
            if i % 100 == 0 or i == len(specs):
                print(f"    {label} {i}/{len(specs)}", flush=True)
    return out
