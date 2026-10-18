"""Fresh-trial test of fixes 1+2 (seed 999000). Rules: PREREG.md. Scored once.

    python experiments/fixes/run_fixes.py

SIMULATED compliant body, not hardware.
"""
import json
import os
import platform
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for p in ("../jepa", "../inverse"):
    sys.path.insert(0, os.path.join(HERE, p))
import sim_probe as SP  # noqa: E402
import run_followup as J  # noqa: E402
import run_inverse as I  # noqa: E402
sys.path.insert(0, SP.SA)
import compliant_config as kc  # noqa: E402
import factorial_config as fc  # noqa: E402
import run_history as rh  # noqa: E402

SEED, N_CAL, N_HANG = 999000, 150, 50
CONFIGS = {"soft_none": (4.0, "none"), "soft_low": (4.0, "low"),
           "rigid_none": (None, "none"), "medium_none": (30.0, "none")}
TG, TP = 2.5, 3.0
CELLS = ("neither", "body_only", "world_only", "body_world")
EP = os.path.join(HERE, "episodes")


def episodes(name, cfg, specs):
    path = os.path.join(EP, f"{name}.npz")
    if not os.path.exists(path):
        mp = rh.single_model_path(name, cfg)
        sp = [dict(s, cfg=cfg, model_path=mp) for s in specs]
        hang = [dict(seed=1999000 + i, cfg=cfg, model_path=mp, hanging=True, onset=0, cell="hang")
                for i in range(N_HANG)]
        res = SP.run_all(sp + hang)
        L = max(len(r["obs"]) for r in res)
        pad = lambda a, n: np.concatenate([a, np.full((n - len(a),) + a.shape[1:], np.nan, a.dtype)])
        os.makedirs(EP, exist_ok=True)
        np.savez_compressed(path, seed=[r["seed"] for r in res], T_ctrl=[r["T_ctrl"] for r in res],
                            obs=np.stack([pad(r["obs"], L) for r in res]),
                            act=np.stack([pad(r["act"], L - 1) for r in res]))
    return np.load(path)


def main():
    fc.N_CALIB = 50                     # seed-90000 training recordings had 50 calibration episodes
    s90 = J.specs_for(90000)
    fc.N_CALIB = N_CAL
    specs = J.specs_for(SEED)
    fc.N_CALIB = 50
    assert specs[0]["seed"] == SEED and sum(s["cell"] == "calib" for s in specs) == N_CAL
    n = len(specs)
    cell = np.array([s["cell"] for s in specs]); cal = cell == "calib"
    res = {"versions": dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                            test_master_seed=SEED, n_calib=N_CAL, n_hanging_calib=N_HANG, gain_threshold=TG,
                            probe_threshold=TP, probe=dict(settle=SP.SETTLE, probe=SP.PROBE, amp=SP.AMP, freq=SP.FREQ),
                            model_seeds=[0, 1, 2], preregistration="PREREG.md")}
    for name, (k, nz) in CONFIGS.items():
        cfg = dict(stiffness=k, noise=kc.NOISE[nz])
        d = episodes(name, cfg, specs)
        assert list(d["seed"][:n]) == [s["seed"] for s in specs]
        obs_all, act_all, Tc = d["obs"], d["act"], d["T_ctrl"]
        obs, act = obs_all[:n], act_all[:n]
        th = np.abs(np.arctan2(obs[:, :, 1], obs[:, :, 0]))
        fell = np.array([th[i, s["onset"] + 180:s["onset"] + 200].max() >= np.pi / 2 for i, s in enumerate(specs)])
        amp = np.array([SP.probe_amplitude(obs_all[i], Tc[i]) for i in range(len(obs_all))])
        la = np.log(amp); hang = np.arange(len(obs_all)) >= n
        zp = ((la - la[hang].mean()) / la[hang].std(ddof=1))[:n]
        d90 = np.load(os.path.join(J.SA, "results_history", "episodes", f"{name}.npz"))
        X, Y = I.train_set(d90["obs"], d90["act"], s90)
        out = {"fell": {c: int(fell[cell == c].sum()) for c in CELLS},
               "z_probe_fallen": {c: [float(np.min(zp[(cell == c) & fell])), float(np.median(zp[(cell == c) & fell])),
                                      float(np.max(zp[(cell == c) & fell]))] if ((cell == c) & fell).any() else None
                                  for c in CELLS},
               "seeds": {}}
        for seed in (0, 1, 2):
            m, _ = I.train(X, Y, seed)
            P = I.per_step(m, np.nan_to_num(obs), np.nan_to_num(act))
            g = I.wsum(P["aha"], specs, (100, 190)) / I.wsum(P["aa"], specs, (100, 190))
            lg = np.log(np.clip(g, 1e-6, None)); zg = (lg - lg[cal].mean()) / lg[cal].std(ddof=1)
            score = np.where(fell, np.abs(zp), np.abs(zg))
            verdict = np.where(fell, np.where(np.abs(zp) >= TP, "body", "world"), np.where(np.abs(zg) >= TG, "body", "world"))
            old = np.where(fell, "fell", np.where(np.abs(zg) >= 2.0, "body", "world"))
            o = dict(verdicts={c: {v: int((verdict[cell == c] == v).sum()) for v in ("body", "world")} for c in CELLS},
                     verdicts_by_branch={c: {"upright_body": int(((verdict == "body") & ~fell)[cell == c].sum()),
                                             "fell_body": int(((verdict == "body") & fell)[cell == c].sum())} for c in CELLS},
                     old_rule={c: {v: int((old[cell == c] == v).sum()) for v in ("body", "world", "fell")} for c in CELLS},
                     auc_body_push=J.auc(score[cell == "body_only"], score[cell == "world_only"]),
                     auc_body_undist=J.auc(score[cell == "body_only"], score[cell == "neither"]))
            out["seeds"][seed] = o
            v = o["verdicts"]
            print(f"{name} seed {seed}: body-only->body {v['body_only']['body']} | push->body {v['world_only']['body']} | "
                  f"undist->body {v['neither']['body']} | body+push->body {v['body_world']['body']} | "
                  f"AUC body/push {o['auc_body_push']:.3f} | branches {o['verdicts_by_branch']}", flush=True)
        print(f"  {name}: fell {out['fell']} | z_probe fallen (min/med/max) {out['z_probe_fallen']}", flush=True)
        res[name] = out
    crit = {}
    for seed in (0, 1, 2):
        V = lambda c: res[c]["seeds"][seed]
        crit[seed] = dict(
            S=all(V(c)["verdicts"]["world_only"]["body"] <= 10 and V(c)["verdicts"]["neither"]["body"] <= 10 for c in CONFIGS),
            D=all(V(c)["verdicts"]["body_only"]["body"] >= 80 for c in CONFIGS),
            A=all(V(c)["auc_body_push"] >= 0.90 for c in CONFIGS))
    res["criteria"] = dict(per_seed=crit, **{k: all(crit[s][k] for s in crit) for k in ("S", "D", "A")})
    res["criteria"]["PASS"] = all(res["criteria"][k] for k in ("S", "D", "A"))
    print(json.dumps(res["criteria"], indent=1))
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
