"""Regime-matched forward model: trained ONLY on healthy balancing transitions.

    python run_regime_model.py

Question: the random-action model has a flat error floor (~0.08-0.09 on the soft
body) that sits above the real step-to-step motion while balancing. Does a model
trained on the balancing regime itself drop that floor below the motion, and
does detection then work?

DATA (recorded episodes in results_history/episodes/; nothing is simulated):
  TRAIN   the PRE-ONSET part of every trial (steps before the disturbance onset;
          the body is unchanged and nothing has pushed it yet), keeping only
          balancing transitions: both ends calm by the repo's existing definition
          (|theta| < THETA_CALM 0.3 rad, |theta_dot| < OMEGA_CALM 0.5 rad/s).
          Why pre-onset and not the healthy episodes: the 120 "neither" episodes
          are the undisturbed TEST group and the 50 calibration episodes set the
          decision point; training on either would leak. Pre-onset steps are never
          scored, and every trial contributes them in the same way.
  FLOOR   held-out balancing transitions AFTER onset in the 50 calibration
          episodes (never trained on).
  SCORE   the same 480 factorial trials, after onset.
Training: model.train_model, unchanged (same MLP, Adam lr, 1000 full-batch
epochs, seed 0), one torch thread. Two regime-matched models per config:
  "regime_all"  every pre-onset balancing transition (~36.7k)
  "regime_2k"   a random 2000 of them (numpy seed 0) - the original model's
                data size, so "more data" is separated from "different data".
The controller is unchanged: the recorded trajectories were driven by the
original model; the new models only score them (paired comparison).

SCORE (fixed in advance): mean one-step error over the window onset+20 ..
onset+100 (1 s after onset, when every push is over, for 4 s). Decision point:
healthy calibration episodes only, threshold = mean + 2 sd -> body if above.
Secondary: the old settle rule's settled score (same gating, same trials).

SIMULATED compliant body, not hardware.
"""
import csv
import json
import os
import platform
import sys

import gymnasium
import numpy as np
import torch

import alarm_classifier as A
import classify_config as cc
import config
import factorial_config as fc
import metrics
import model as model_lib
import run_factorial as rf
from run_compliant import auc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_regime_model")
MODEL_DIR = os.path.join(HERE, "regime_models")
CONFIGS = ("rigid_none", "soft_none", "soft_low")
W0, W1 = 20, 100
Z = 2.0
SUB_N = 2000
LOG = []


def say(x=""):
    print(x, flush=True); LOG.append(x)


def calm(obs):
    th = np.arctan2(obs[..., 1], obs[..., 0]); w = obs[..., 2]
    c = (np.abs(th) < cc.THETA_CALM) & (np.abs(w) < cc.OMEGA_CALM)
    return c[..., :-1] & c[..., 1:]                     # transition k: obs[k] -> obs[k+1]


def errors(m, obs, act):
    n, T = act.shape[0], act.shape[1]
    with torch.no_grad():
        p = m.predict_next(torch.from_numpy(obs[:, :-1].reshape(-1, 3).astype(np.float32)),
                           torch.from_numpy(act.reshape(-1, 1).astype(np.float32))).numpy()
    return np.linalg.norm(p - obs[:, 1:].reshape(-1, 3), axis=1).reshape(n, T)


def rate(rs, pred):
    k = sum(pred(r) for r in rs); lo, hi = metrics.wilson(k, len(rs))
    return dict(k=k, n=len(rs), rate=k / len(rs), lo=lo, hi=hi)


def run_config(name):
    specs = rf.make_specs()
    d = np.load(os.path.join(HERE, "results_history", "episodes", f"{name}.npz"))
    assert list(d["seed"]) == [s["seed"] for s in specs]
    obs, act, err_rec = d["obs"], d["act"], d["err"]
    onset = np.array([s["onset"] for s in specs])
    T = act.shape[1]
    c = calm(obs)
    pre = np.arange(T)[None, :] < onset[:, None]
    tr = c & pre
    S, Aa, S2 = obs[:, :-1][tr], act[tr], obs[:, 1:][tr]
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(len(S), SUB_N, replace=False))
    torch.set_num_threads(1)
    os.makedirs(MODEL_DIR, exist_ok=True)
    models = {"original": model_lib.load_model(os.path.join(
        HERE, "forward_model.pt" if name == "rigid_none" else f"compliant_models/{name}.pt"))}
    for tag, idx in (("regime_all", np.arange(len(S))), ("regime_2k", sub)):
        path = os.path.join(MODEL_DIR, f"{name}_{tag}.pt")
        if not os.path.exists(path):
            m = model_lib.train_model(S[idx], Aa[idx], S2[idx], seed=config.SEED)
            model_lib.save_model(m, path)
        models[tag] = model_lib.load_model(path)
    E = {k: errors(m, obs, act) for k, m in models.items()}
    assert np.allclose(E["original"], err_rec, atol=1e-6), "original model does not reproduce recorded errors"

    # ---- check 1: floor on held-out balancing transitions (calibration episodes, after onset)
    is_cal = np.array([s["cell"] == "calib" for s in specs])
    post = (np.arange(T)[None, :] > onset[:, None]) & (np.arange(T)[None, :] < onset[:, None] + 200)
    held = c & post & is_cal[:, None]
    motion = np.linalg.norm(obs[:, 1:] - obs[:, :-1], axis=2)
    floor = {k: dict(mean=float(e[held].mean()), median=float(np.median(e[held]))) for k, e in E.items()}
    floor["motion_nothing_changes"] = dict(mean=float(motion[held].mean()), median=float(np.median(motion[held])))
    # also how the regime models do OUTSIDE balancing (known limitation): post-push high-motion steps
    w_abs = np.abs(obs[:, :-1, 2])
    is_push = np.array([s["cell"] == "world_only" for s in specs])
    after_push = np.array([[t >= s["push_onset"] + s["duration"] for t in range(T)] if s["cell"] == "world_only"
                           else [False] * T for s in specs])
    hi = after_push & (w_abs >= cc.OMEGA_CALM)
    outside = {k: float(e[hi].mean()) for k, e in E.items()}
    outside["motion_nothing_changes"] = float(motion[hi].mean())

    # ---- check 2: detection, same trials, same windows
    out = dict(config=name, n_train_all=int(len(S)), n_train_2k=SUB_N, n_floor_heldout=int(held.sum()),
               n_high_motion=int(hi.sum()), floor=floor, high_motion_error=outside, models={})
    rows = [s for s in specs if s["cell"] != "calib"]
    for k, e in E.items():
        sc = np.array([e[i, s["onset"] + W0:s["onset"] + W1].mean() for i, s in enumerate(specs)])
        settled = [A.settled_score(e[i], np.arctan2(obs[i, 1:, 1], obs[i, 1:, 0]), obs[i, 1:, 2], s["onset"])[0]
                   for i, s in enumerate(specs)]
        for s, v, st in zip(specs, sc, settled):
            s[f"score_{k}"] = float(v); s[f"settled_{k}"] = float("nan") if st is None else float(st)
        cal = sc[is_cal]; thr = float(cal.mean() + Z * cal.std(ddof=1))
        for s in specs:
            s[f"verdict_{k}"] = "body" if s[f"score_{k}"] >= thr else "world"
        g = lambda cell: [s[f"score_{k}"] for s in specs if s["cell"] == cell]
        sb = [s[f"settled_{k}"] for s in rows if s["has_body"] and not np.isnan(s[f"settled_{k}"])]
        sn = [s[f"settled_{k}"] for s in rows if not s["has_body"] and not np.isnan(s[f"settled_{k}"])]
        body = lambda s: s[f"verdict_{k}"] == "body"
        out["models"][k] = dict(
            threshold=thr,
            auc_body_vs_undisturbed=auc(g("body_only"), g("neither")),
            auc_body_vs_push=auc(g("body_only"), g("world_only")),
            auc_bodypush_vs_push=auc(g("body_world"), g("world_only")),
            settled_auc=auc(sb, sn), settled_n=(len(sb), len(sn)),
            median={cell: float(np.median(g(cell))) for cell in rf.CELLS},
            detection=rate([s for s in rows if s["has_body"]], body),
            detect_body_only=rate([s for s in rows if s["cell"] == "body_only"], body),
            false_positive=rate([s for s in rows if s["cell"] == "world_only"], body),
            masking=rate([s for s in rows if s["cell"] == "body_world"], lambda s: not body(s)),
            undisturbed_flagged=rate([s for s in rows if s["cell"] == "neither"], body))
    keys = ["cell", "seed", "has_body", "has_push", "param", "factor", "torque", "duration_name", "onset"]
    for k in E:
        keys += [f"score_{k}", f"settled_{k}", f"verdict_{k}"]
    with open(os.path.join(OUT, f"trials_{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config"] + keys); w.writeheader()
        w.writerows([{"config": name, **{k: s[k] for k in keys}} for s in specs])
    json.dump(out, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    versions = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                    gymnasium=gymnasium.__version__, master_seed=fc.MASTER_SEED, model_train_seed=config.SEED,
                    subsample_seed=0, subsample_n=SUB_N, window=[W0, W1], z=Z, train_threads=1,
                    train="pre-onset calm transitions of all 530 recorded trials (results_history/episodes)")
    json.dump(versions, open(os.path.join(OUT, "versions_and_seed.json"), "w"), indent=1)
    say("SIMULATED compliant body, not hardware. Recorded episodes only; controller unchanged.")
    say(f"versions/seed: {versions}")
    res = [run_config(n) for n in CONFIGS]
    for r in res:
        f = r["floor"]
        say(f"\n== {r['config']} == train transitions: all {r['n_train_all']}, subsample {r['n_train_2k']}; "
            f"held-out balancing transitions {r['n_floor_heldout']}")
        say(f"  FLOOR (held-out balancing, mean / median): motion ('nothing changes') "
            f"{f['motion_nothing_changes']['mean']:.4f} / {f['motion_nothing_changes']['median']:.4f}")
        for k in ("original", "regime_all", "regime_2k"):
            say(f"    {k:<11} {f[k]['mean']:.4f} / {f[k]['median']:.4f}   "
                f"(ratio to motion {f[k]['mean'] / f['motion_nothing_changes']['mean']:.2f})")
        say(f"  OUTSIDE balancing (post-push, |thdot| >= 0.5, n={r['n_high_motion']}): motion "
            f"{r['high_motion_error']['motion_nothing_changes']:.4f} | " +
            " | ".join(f"{k} {r['high_motion_error'][k]:.4f}" for k in ("original", "regime_all", "regime_2k")))
        say(f"  DETECTION (window score onset+{W0}..+{W1}; threshold mean+{Z:g}sd of calibration):")
        for k in ("original", "regime_all", "regime_2k"):
            m = r["models"][k]
            say(f"    {k:<11} AUC body/undisturbed {m['auc_body_vs_undisturbed']:.3f} | body/push {m['auc_body_vs_push']:.3f} | "
                f"settled AUC {m['settled_auc']:.3f} | detect {m['detection']['k']:>3}/240 (body-only {m['detect_body_only']['k']}/120) | "
                f"FP {m['false_positive']['k']:>3}/120 | mask {m['masking']['k']:>3}/120 | undisturbed flagged {m['undisturbed_flagged']['k']:>3}/120")
            say(f"                medians: body-only {m['median']['body_only']:.4f}, undisturbed {m['median']['neither']:.4f}, "
                f"push-only {m['median']['world_only']:.4f}, body+push {m['median']['body_world']:.4f}")
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    sys.exit(main())
