"""Does multi-step rollout error reveal a soft-body change that one-step error misses?

    python run_rollout.py

Offline re-scoring of the SAME recorded trials (results_history/episodes/) with
the SAME trained single-state models; see rollout_config.py for the rollout,
where it starts, the decision point and the divergence check. Sanity check:
the H = 1 rollout error must equal the recorded one-step error exactly.

SIMULATED compliant body (series-elastic + hysteresis + damping + sensor
noise), not real hardware.
"""
import csv
import json
import os
import platform
import sys

import gymnasium
import numpy as np
import torch

import compliant_config as kc
import factorial_config as fc
import metrics
import model as model_lib
import plot_rollout
import rollout_config as rc
import run_factorial as rf
from run_compliant import auc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, rc.RESULTS)
LOG = []
HMAX = max(rc.H_SWEEP)


def say(msg=""):
    print(msg, flush=True)
    LOG.append(msg)


def model_path(name):
    return os.path.join(HERE, "forward_model.pt" if name == "rigid_none"
                        else os.path.join("compliant_models", f"{name}.pt"))


def rollout_errors(m, obs, act, starts):
    """Per-step free-running errors, shape (n_trials, n_starts, HMAX), plus the trivial predictor's."""
    n, S = obs.shape[0], starts.shape[1]
    idx = np.arange(n)[:, None]
    s = torch.from_numpy(obs[idx, starts].reshape(n * S, 3).astype(np.float32))
    s0 = s.clone()
    err = np.empty((n * S, HMAX), np.float64); hold = np.empty_like(err)
    with torch.no_grad():
        for h in range(HMAX):
            a = torch.from_numpy(act[idx, starts + h].reshape(n * S, 1).astype(np.float32))
            s = m.predict_next(s, a)
            real = torch.from_numpy(obs[idx, starts + h + 1].reshape(n * S, 3).astype(np.float32))
            err[:, h] = torch.linalg.norm(s - real, dim=1).numpy()
            hold[:, h] = torch.linalg.norm(s0 - real, dim=1).numpy()
    return err.reshape(n, S, HMAX), hold.reshape(n, S, HMAX)


def rate(rs, pred):
    k = sum(pred(r) for r in rs); lo, hi = metrics.wilson(k, len(rs))
    return dict(k=k, n=len(rs), rate=k / len(rs), lo=lo, hi=hi)


def run_config(name, cfg):
    specs = rf.make_specs()
    d = np.load(os.path.join(HERE, rc.EPISODES, f"{name}.npz"))
    assert list(d["seed"]) == [s["seed"] for s in specs]
    obs, act, err1 = d["obs"], d["act"], d["err"]
    onset = np.array([s["onset"] for s in specs])
    starts = onset[:, None] + rc.START_LAG + rc.START_STEP * np.arange(rc.N_STARTS)[None, :]
    assert starts.max() + HMAX <= act.shape[1]
    m = model_lib.load_model(model_path(name))
    e, hold = rollout_errors(m, obs, act, starts)
    # H = 1 must be the recorded one-step error at those steps
    rec = err1[np.arange(len(specs))[:, None], starts]
    h1_mismatch = int(np.sum(np.abs(e[:, :, 0] - rec) > 1e-6))
    cum = np.cumsum(e, axis=2) / np.arange(1, HMAX + 1)            # mean over h = 1..H
    cumh = np.cumsum(hold, axis=2) / np.arange(1, HMAX + 1)
    finite = np.isfinite(e).all(axis=(1, 2))

    is_cal = np.array([s["cell"] == "calib" for s in specs])
    cell = np.array([s["cell"] for s in specs])
    out = dict(config=name, stiffness=cfg["stiffness"], noise=list(cfg["noise"]), h1_mismatches=h1_mismatch,
               nonfinite_trials=int((~finite).sum()), by_H={})
    for H in rc.H_SWEEP:
        score = cum[:, :, H - 1].mean(axis=1)
        hscore = cumh[:, :, H - 1].mean(axis=1)
        for s, v, hv in zip(specs, score, hscore):
            s[f"roll_{H}"] = float(v); s[f"hold_{H}"] = float(hv)
        cal = score[is_cal]
        thr = float(cal.mean() + rc.Z * cal.std(ddof=1))
        g = lambda c: score[cell == c]
        rows = [s for s in specs if s["cell"] != "calib"]
        body = lambda s: s[f"roll_{H}"] >= thr
        out["by_H"][H] = dict(
            threshold=thr,
            healthy_median=float(np.median(cal)), healthy_hold_median=float(np.median(hscore[is_cal])),
            diverged=bool(np.median(cal) > np.median(hscore[is_cal])),
            auc_body_vs_neither=auc(g("body_only"), g("neither")),
            auc_body_vs_push=auc(g("body_only"), g("world_only")),
            auc_bodyworld_vs_push=auc(g("body_world"), g("world_only")),
            median={c: float(np.median(g(c))) for c in rf.CELLS},
            detection=rate([s for s in rows if s["has_body"]], body),
            detect_body_only=rate([s for s in rows if s["cell"] == "body_only"], body),
            false_positive=rate([s for s in rows if s["cell"] == "world_only"], body),
            neither_flagged=rate([s for s in rows if s["cell"] == "neither"], body),
            masking=rate([s for s in rows if s["cell"] == "body_world"], lambda s: not body(s)),
            fp_by_duration={dn: rate([s for s in rows if s["cell"] == "world_only" and s["duration_name"] == dn], body)
                            for dn in fc.PUSH_DURATIONS},
        )
        for s in specs:
            s[f"verdict_{H}"] = "body" if body(s) else "world"
    keys = ["cell", "seed", "has_body", "has_push", "param", "factor", "torque", "duration_name", "onset"]
    for H in rc.H_SWEEP:
        keys += [f"roll_{H}", f"hold_{H}", f"verdict_{H}"]
    with open(os.path.join(OUT, f"trials_{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config"] + keys); w.writeheader()
        w.writerows([{"config": name, **{k: s[k] for k in keys}} for s in specs])
    json.dump(out, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(1)
    versions = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                    gymnasium=gymnasium.__version__, master_seed=fc.MASTER_SEED, h_sweep=list(rc.H_SWEEP),
                    h_main=rc.H_MAIN, start_lag=rc.START_LAG, n_starts=rc.N_STARTS, start_step=rc.START_STEP,
                    z=rc.Z, episodes=rc.EPISODES,
                    models="forward_model.pt (rigid_none), compliant_models/*.pt (single-state, unchanged)")
    json.dump(versions, open(os.path.join(OUT, "versions_and_seed.json"), "w"), indent=1)
    say("SIMULATED compliant body (series-elastic + hysteresis + damping + sensor noise): "
        "a model of compliance, not real hardware.")
    say(f"versions/seed: {versions}")
    shape = {r["config"]: r for r in json.load(open(os.path.join(HERE, "results_shape", "summary.json")))}
    results = []
    for s, k in kc.STIFFNESS.items():
        for n, nz in kc.NOISE.items():
            r = run_config(f"{s}_{n}", dict(stiffness=k, noise=nz))
            results.append(r)
            say(f"\n== {r['config']} == H=1 vs recorded one-step error: {r['h1_mismatches']} mismatches; "
                f"non-finite rollouts: {r['nonfinite_trials']}")
            say(f"  {'H':>4}{'healthy roll':>13}{'hold':>8}{'div':>5}{'AUC b/undist':>13}{'AUC b/push':>11}"
                f"{'med body':>10}{'med undist':>11}{'med push':>10}{'detect':>8}{'FP':>5}{'undist':>7}")
            for H in rc.H_SWEEP:
                m = r["by_H"][H]
                say(f"  {H:>4}{m['healthy_median']:>13.4f}{m['healthy_hold_median']:>8.4f}{'YES' if m['diverged'] else '-':>5}"
                    f"{m['auc_body_vs_neither']:>13.3f}{m['auc_body_vs_push']:>11.3f}{m['median']['body_only']:>10.4f}"
                    f"{m['median']['neither']:>11.4f}{m['median']['world_only']:>10.4f}{m['detection']['k']:>8}"
                    f"{m['false_positive']['k']:>5}{m['neither_flagged']['k']:>7}")
    say(f"\nSUMMARY: AUC body-only vs undisturbed (main test) and body-only vs push-only, by horizon H")
    say(f"  {'config':<15}" + "".join(f"{f'H={H}':>8}" for H in rc.H_SWEEP) + "   |" +
        "".join(f"{f'H={H}':>8}" for H in rc.H_SWEEP) + "   | shape4b b/push W=20")
    for r in results:
        say(f"  {r['config']:<15}" + "".join(f"{r['by_H'][H]['auc_body_vs_neither']:>8.3f}" for H in rc.H_SWEEP) + "   |" +
            "".join(f"{r['by_H'][H]['auc_body_vs_push']:>8.3f}" for H in rc.H_SWEEP) +
            f"   | {shape[r['config']]['shape']['20']['auc_bodyonly_vs_worldonly']:.3f}")
    say("  (left block: body-only vs undisturbed; right block: body-only vs push-only; H=1 = one-step error)")
    say(f"\nDIVERGENCE: healthy median rollout error / 'nothing changes' predictor, by H")
    for r in results:
        say(f"  {r['config']:<15}" + "".join(
            f"{r['by_H'][H]['healthy_median'] / r['by_H'][H]['healthy_hold_median']:>8.2f}" for H in rc.H_SWEEP))
    say("  (> 1 = the free-running model predicts the motion worse than assuming nothing changes)")
    json.dump(results, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    fig = plot_rollout.sweep(results, os.path.join(OUT, "auc_vs_horizon.png"))
    say(f"\nfigure: {os.path.relpath(fig, HERE)}")
    open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    sys.exit(main())
