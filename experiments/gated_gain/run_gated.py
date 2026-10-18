"""Fell / body / world with gated command gain, on fresh seed-888000 trials. Rules: PREREG.md.

    python experiments/gated_gain/run_gated.py

Records fresh episodes (same physics, models, controller and loop as before),
trains the inverse model on healthy seed-90000 data (3 seeds), scores once.
SIMULATED compliant body, not hardware.
"""
import json
import multiprocessing as mpc
import os
import platform
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SA = os.path.abspath(os.path.join(HERE, "..", "..", "surprise_attribution"))
sys.path.insert(0, SA)
sys.path.insert(0, os.path.join(HERE, "..", "inverse"))
sys.path.insert(0, os.path.join(HERE, "..", "jepa"))
import compliant_config as kc  # noqa: E402
import compliant_trials as ct  # noqa: E402
import overlap_trials as ot  # noqa: E402
import run_history as rh  # noqa: E402
import run_followup as J  # noqa: E402
import run_inverse as I  # noqa: E402

SEED = 888000
CONFIGS = {"soft_none": (4.0, "none"), "soft_low": (4.0, "low"),
           "rigid_none": (None, "none"), "medium_none": (30.0, "none")}
JUDGED = ("soft_none", "soft_low", "rigid_none")
LATE = (100, 200)
EP = os.path.join(HERE, "episodes")


def record(name, cfg, specs):
    path = os.path.join(EP, f"{name}.npz")
    if not os.path.exists(path):
        mp = rh.single_model_path(name, cfg)
        for s in specs:
            s.update(cfg=cfg, model_path=mp, steps=ot.STEPS, config=name)
        res = []
        with mpc.get_context("fork").Pool(kc.N_WORKERS, initializer=ct._init_worker) as pool:
            for i, r in enumerate(pool.imap(rh._worker, specs, chunksize=2), 1):
                res.append(r)
                if i % 100 == 0 or i == len(specs):
                    print(f"    {name} {i}/{len(specs)}", flush=True)
        os.makedirs(EP, exist_ok=True)
        np.savez_compressed(path, seed=[r["seed"] for r in res],
                            **{k: np.stack([r[k] for r in res]) for k in ("obs", "act", "err")})
    d = np.load(path)
    assert list(d["seed"]) == [s["seed"] for s in specs]
    return d


def main():
    s90 = J.specs_for(90000)
    s88 = J.specs_for(SEED)
    assert s88[0]["seed"] == SEED
    cell = np.array([s["cell"] for s in s88]); cal = cell == "calib"
    cells = ("neither", "body_only", "world_only", "body_world")
    res = {"versions": dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                            test_master_seed=SEED, train_master_seed=90000, model_seeds=[0, 1, 2],
                            preregistration="PREREG.md (commit 8ade83e)")}
    for name, (k, n) in CONFIGS.items():
        cfg = dict(stiffness=k, noise=kc.NOISE[n])
        d = record(name, cfg, [dict(s) for s in s88])
        obs, act = d["obs"], d["act"]
        th = np.abs(np.arctan2(obs[:, :, 1], obs[:, :, 0]))
        fell = np.array([th[i, s["onset"] + 180:s["onset"] + 200].max() >= np.pi / 2 for i, s in enumerate(s88)])
        # Rung 6 rule B context (early-window motion z vs fresh calibration, body if z <= -2)
        mot = np.linalg.norm(obs[:, 1:] - obs[:, :-1], axis=2)
        x = np.log(J.win(mot, s88, (20, 100))); zb = (x - x[cal].mean()) / x[cal].std(ddof=1)
        motion_late = J.win(mot, s88, LATE)
        d90 = np.load(os.path.join(SA, "results_history", "episodes", f"{name}.npz"))
        X, Y = I.train_set(d90["obs"], d90["act"], s90)
        out = {"fell": {c: int(fell[cell == c].sum()) for c in cells},
               "rung6_B": {"called_body": {c: int((zb[cell == c] <= -2).sum()) for c in cells},
                           "auc_body_push": J.auc(-zb[cell == "body_only"], -zb[cell == "world_only"])},
               "seeds": {}}
        for seed in (0, 1, 2):
            m, hl = I.train(X, Y, seed)
            P = I.per_step(m, obs, act)
            g = I.wsum(P["aha"], s88, LATE) / I.wsum(P["aa"], s88, LATE)
            lg = np.log(np.clip(g, 1e-6, None))
            z = (lg - lg[cal].mean()) / lg[cal].std(ddof=1)
            verdict = np.where(fell, "fell", np.where(np.abs(z) >= 2, "body", "world"))
            up = ~fell
            sel = lambda c: (cell == c) & up
            o = dict(heldout_mse=hl,
                     verdicts={c: {v: int((verdict[cell == c] == v).sum()) for v in ("body", "world", "fell")} for c in cells},
                     auc_upright_body_push=J.auc(np.abs(z[sel("body_only")]), np.abs(z[sel("world_only")]))
                     if sel("body_only").any() and sel("world_only").any() else None,
                     auc_upright_body_undist=J.auc(np.abs(z[sel("body_only")]), np.abs(z[sel("neither")]))
                     if sel("body_only").any() and sel("neither").any() else None,
                     median_gain={c: float(np.median(g[cell == c])) for c in cells},
                     median_gain_upright={c: float(np.median(g[sel(c)])) if sel(c).any() else None for c in cells},
                     r_motion_undist=float(np.corrcoef(np.abs(z[cell == "neither"]), motion_late[cell == "neither"])[0, 1]))
            out["seeds"][seed] = o
            v = o["verdicts"]
            print(f"{name} seed {seed}: body-only {v['body_only']} | push-only {v['world_only']} | undisturbed {v['neither']} | "
                  f"body+push {v['body_world']} | AUC upright body/push {o['auc_upright_body_push']} body/undist {o['auc_upright_body_undist']}",
                  flush=True)
        res[name] = out
        print(f"  {name}: fell per cell {out['fell']} | Rung 6 B called body {out['rung6_B']['called_body']}", flush=True)
    crit = {}
    for seed in (0, 1, 2):
        S = all(res[c]["seeds"][seed]["verdicts"]["world_only"]["body"] <= 10 and
                res[c]["seeds"][seed]["verdicts"]["neither"]["body"] <= 10 for c in CONFIGS)
        D = all(res[c]["seeds"][seed]["verdicts"]["body_only"]["body"] >= 80 for c in JUDGED)
        A = all((res[c]["seeds"][seed]["auc_upright_body_push"] or 0) >= 0.90 for c in JUDGED)
        crit[seed] = dict(S=S, D=D, A=A)
    res["criteria"] = dict(per_seed=crit, S=all(v["S"] for v in crit.values()),
                           D=all(v["D"] for v in crit.values()), A=all(v["A"] for v in crit.values()))
    res["criteria"]["PASS"] = res["criteria"]["S"] and res["criteria"]["D"] and res["criteria"]["A"]
    print(json.dumps(res["criteria"], indent=1))
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
