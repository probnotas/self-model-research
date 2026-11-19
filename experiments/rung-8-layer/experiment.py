"""Rung 8: policy alone vs + self-model filter vs + self-model adaptation vs oracle.

    python experiments/rung-8-layer/experiment.py pilot       # healthy A and D, seeds 880100-880109
    python experiments/rung-8-layer/experiment.py calibrate   # healthy A, seeds 881000-881019 -> threshold_B
    python experiments/rung-8-layer/experiment.py main        # refuses unless PREREG.md exists

Balance task: start within +/-0.2 rad (and rad/s) of upright, as in the main experiment;
perturbation switched on at step 50; 500 steps. Success = never past 90 deg, and
|theta| <= 10 deg for each of the final 50 steps.

Conditions (all reuse surprise_attribution/ unchanged; see rung8.py):
  A policy alone   CEM-MPC through soft_none.pt, command executed as is
  B filter         self-model gates each command on its online-corrected predicted mismatch
  C adaptation     self-model's online command gain g rescales the command by 1/g
  D oracle         same CEM, true physics, true post-change parameters, true full state
SIMULATION ONLY.
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

N_RETRIES, EXCLUDE = 3, 0.25           # filter: replans; candidates whose first torque is within 0.25 of a rejected one are excluded
C_RANGE = (0.5, 2.0)                    # adaptation: correction 1/g used only inside this range
SEEDS = dict(pilot=range(880100, 880110), calibrate=range(881000, 881020), main=range(882000, 882030))
CONDITIONS = ("A_policy", "B_filter", "C_adapt", "D_oracle")


def cem_excluding(s, excluded):
    """planner.choose_action's CEM loop, with candidates near rejected first torques excluded."""
    cfg, P = R.config, R.planner
    H = cfg.HORIZON
    mean = np.zeros((H, 1), dtype=np.float32)
    std = np.full((H, 1), cfg.CEM_INIT_STD, dtype=np.float32)
    for _ in range(cfg.CEM_ITERS):
        noise = np.random.randn(cfg.CEM_CANDIDATES, H, 1).astype(np.float32)
        acts = np.clip(mean + std * noise, P.ACTION_LOW, P.ACTION_HIGH).astype(np.float32)
        costs = P.rollout_costs(R.model(), s, acts)
        for x in excluded:
            costs = np.where(np.abs(acts[:, 0, 0] - x) < EXCLUDE, np.inf, costs)
        elites = acts[np.argsort(costs)[:cfg.CEM_ELITES]]
        mean, std = elites.mean(axis=0), elites.std(axis=0)
    return np.clip(mean[0], P.ACTION_LOW, P.ACTION_HIGH).astype(np.float32)


class SelfModelMonitor:
    """Online comparison of the frozen self-model with what actually happened.

    gain g: least-squares fit of the observed effect of the command to the predicted one over
    the last W steps, effect = next state minus the model's zero-torque prediction:
        g = sum (s' - f(s,0)) . (f(s,u) - f(s,0)) / sum |f(s,u) - f(s,0)|^2
    g ~ 1 when commands move the body as the self-model expects; g < 1 when they move it less."""

    def __init__(self):
        self.num, self.den, self.err = [], [], []
        self.pending = None

    def before(self, s, u):
        p_u, p_0 = R.predict(s, u), R.predict(s, np.zeros(1, np.float32))
        self.pending = (p_u, p_0)

    def after(self, s_next):
        p_u, p_0 = self.pending
        eff_pred = p_u - p_0
        self.num.append(float((s_next - p_0) @ eff_pred)); self.den.append(float(eff_pred @ eff_pred))
        self.err.append(float(np.linalg.norm(s_next - p_u)))
        self.num, self.den, self.err = self.num[-R.W:], self.den[-R.W:], self.err[-R.W:]

    def gain(self):
        if len(self.den) < R.W or sum(self.den) < 1e-9:
            return None
        return sum(self.num) / sum(self.den)


def mismatch(s, u, g):
    """|| self-model's corrected forecast - policy's own forecast || = |1-g| * |f(s,u) - f(s,0)|."""
    if g is None:
        return 0.0
    return abs(1.0 - g) * float(np.linalg.norm(R.predict(s, u) - R.predict(s, np.zeros(1, np.float32))))


def episode(args):
    cond, pert, seed, thr_b = args
    torch.set_num_threads(1)
    env = R.make_env(seed)
    rng = np.random.default_rng(seed)
    r = R.config.START_RANGE
    env.unwrapped.state = np.array([rng.uniform(-r, r), rng.uniform(-r, r)])   # main experiment's start
    np.random.seed(seed)
    obs = env.observe()
    p = R.PERTURBATIONS[pert]
    mon = SelfModelMonitor()
    th, U, mis, rejected_steps, retries, scale = [], [], [], 0, 0, []
    for t in range(R.T_MAX):
        if t == R.ONSET:
            env.perturb(p)
        if cond == "D_oracle":
            u = R.oracle_action(env.true_full_state(), p if t >= R.ONSET else {}, env.unwrapped)
        else:
            u = R.planner.choose_action(R.model(), obs)
            g = mon.gain()
            if cond == "B_filter":
                m = mismatch(obs, u, g); mis.append(m)
                if thr_b is not None and m >= thr_b:
                    rejected_steps += 1; excluded = [float(u[0])]
                    for _ in range(N_RETRIES):
                        retries += 1
                        u = cem_excluding(obs, excluded)
                        if mismatch(obs, u, g) < thr_b:
                            break
                        excluded.append(float(u[0]))
                    else:
                        u = np.zeros(1, np.float32)                  # safe no-op
            elif cond == "C_adapt" and g is not None and g > 0 and C_RANGE[0] <= 1.0 / g <= C_RANGE[1]:
                u = np.clip(u / g, -2.0, 2.0).astype(np.float32); scale.append(1.0 / g)
        if cond != "D_oracle":
            mon.before(obs, u)
        obs, *_ = env.step(u)
        if cond != "D_oracle":
            mon.after(obs)
        th.append(float(env.unwrapped.state[0])); U.append(float(u[0]))
    th = np.array(th)
    a = np.abs(np.arctan2(np.sin(th), np.cos(th)))
    ok = bool(a.max() < np.pi / 2 and (a[-R.HOLD:] <= R.TOL).all())
    return dict(cond=cond, pert=pert, seed=seed, success=ok, fell=bool(a.max() >= np.pi / 2),
                final_abs_deg=float(np.rad2deg(a[-R.HOLD:].max())), mean_abs_u_post=float(np.mean(np.abs(U[R.ONSET:]))),
                filter_reject_rate=rejected_steps / R.T_MAX if cond == "B_filter" else None,
                filter_retries=retries if cond == "B_filter" else None,
                mismatch_median=float(np.median(mis)) if mis else None, mismatch_all=mis if cond == "B_filter" and thr_b is None else None,
                adapt_mean_scale=float(np.mean(scale)) if scale else None,
                adapt_fraction=len(scale) / R.T_MAX if cond == "C_adapt" else None)


def pool_run(jobs):
    out = []
    with mp.get_context("fork").Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(episode, jobs, chunksize=1), 1):
            out.append(r)
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)
    return out


def main():
    mode = sys.argv[1]
    if mode == "pilot":
        res = pool_run([(c, "healthy", s, None) for c in ("A_policy", "D_oracle") for s in SEEDS["pilot"]])
        summ = {c: f"{sum(r['success'] for r in res if r['cond'] == c)}/{len(SEEDS['pilot'])}" for c in ("A_policy", "D_oracle")}
        print("PILOT (balance, healthy):", summ)
        json.dump(dict(success=summ, runs=res), open(os.path.join(HERE, "pilot_balance_results.json"), "w"), indent=1)
    elif mode == "calibrate":
        # B's threshold: 2 x the median per-step mismatch of the policy's own commands on the HEALTHY body
        res = pool_run([("B_filter", "healthy", s, None) for s in SEEDS["calibrate"]])
        allm = np.concatenate([r["mismatch_all"] for r in res])
        thr = 2.0 * float(np.median(allm[allm > 0])) if (allm > 0).any() else 0.0
        out = dict(threshold_B=thr, median_mismatch=thr / 2, n_steps=int((allm > 0).sum()),
                   healthy_success=f"{sum(r['success'] for r in res)}/{len(res)}", seeds=[SEEDS["calibrate"][0], SEEDS["calibrate"][-1]])
        print(out)
        json.dump(out, open(os.path.join(HERE, "calibration.json"), "w"), indent=1)
    elif mode == "main":
        if not os.path.exists(os.path.join(HERE, "PREREG.md")):
            sys.exit("refusing: PREREG.md must be written and committed before the main run")
        thr = json.load(open(os.path.join(HERE, "calibration.json")))["threshold_B"]
        jobs = [(c, p, s, thr) for p in R.PERTURBATIONS for c in CONDITIONS for s in SEEDS["main"]]
        res = pool_run(jobs)
        json.dump(dict(threshold_B=thr, runs=res), open(os.path.join(HERE, "main_runs.json"), "w"), indent=1)
        print("saved main_runs.json")


if __name__ == "__main__":
    main()
