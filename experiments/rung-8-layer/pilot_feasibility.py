"""PILOT feasibility check: oracle (true-physics CEM) at horizons 20/60 and Condition A at 60/100, healthy, 3 seeds.
Result (printed, not saved): all 0/3; best lift 14 deg above hanging. SIMULATION ONLY."""
import sys, numpy as np, torch, multiprocessing as mp
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rung8 as R
def run(args):
    kind, H, seed = args
    torch.set_num_threads(1); R.config.HORIZON = H
    env = R.make_env(seed); np.random.seed(seed); obs = env.observe(); th = []
    for t in range(R.T_MAX):
        u = R.oracle_action(env.true_full_state(), {}, env.unwrapped) if kind == "oracle" else R.planner.choose_action(R.model(), obs)
        obs, *_ = env.step(u); th.append(env.unwrapped.state[0])
    th = np.array(th); a = np.abs(np.arctan2(np.sin(th), np.cos(th)))
    return kind, H, seed, R.success(th), round(float(np.rad2deg(np.pi - a.min())), 1)
jobs = [(k, H, s) for k, H in (("oracle", 20), ("oracle", 60), ("A", 60), ("A", 100)) for s in (880000, 880001, 880002)]
with mp.get_context("fork").Pool(4) as p:
    for r in p.imap(run, jobs): print(r, flush=True)
