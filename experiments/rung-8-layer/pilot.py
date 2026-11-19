"""PILOT (run before the pre-registration was written): can Condition A swing up at all?

    python experiments/rung-8-layer/pilot.py

Healthy body only, Condition A (policy alone), 10 seeds (880000-880009, not reused in
the main run). Also checks PerturbedCompliant is bit-identical to CompliantPendulum.
Writes pilot_results.json. SIMULATION ONLY.
"""
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rung8 as R  # noqa: E402

SEEDS = range(880000, 880010)


def run(seed):
    torch.set_num_threads(1)
    env = R.make_env(seed)
    np.random.seed(seed)
    obs = env.observe()
    th, us = [], []
    for t in range(R.T_MAX):
        u = R.planner.choose_action(R.model(), obs)
        obs, *_ = env.step(u)
        th.append(float(env.unwrapped.state[0])); us.append(float(u[0]))
    th = np.array(th)
    s = R.success(th)
    return dict(seed=seed, success_step=s, max_height_deg=float(np.rad2deg(np.pi - np.abs(
        np.arctan2(np.sin(th), np.cos(th))).min())), final_angle_deg=float(np.rad2deg(np.arctan2(np.sin(th[-1]), np.cos(th[-1])))),
        mean_abs_u=float(np.mean(np.abs(us))))


def main():
    print("bit-identical check (max |diff|):", R.verify_identical())
    with mp.get_context("fork").Pool(4) as pool:
        res = pool.map(run, SEEDS)
    for r in res:
        print(r)
    n_ok = sum(r["success_step"] is not None for r in res)
    print(f"PILOT: Condition A, healthy, swing-up success {n_ok}/{len(res)}")
    json.dump(dict(success=n_ok, n=len(res), runs=res), open(os.path.join(HERE, "pilot_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
