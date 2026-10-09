"""Two-sided "did its behaviour change?" score, instead of "did error go up?".

    python run_twosided.py

Motivation (results_regime_model/): on the soft body a body change makes the
pendulum move ~3x LESS while balancing, so every score that assumed "body change
= more error" read it backwards. This asks whether behaviour that differs from
the body's own healthy baseline IN EITHER DIRECTION separates body changes.

Fixed before any result was looked at:
  WINDOW  steps onset+20 .. onset+100 (1 s after onset, when every push is over,
          for 4 s) - the same window as run_regime_model.py.
  SIGNALS primary   motion: mean ||s_{t+1} - s_t|| of the observed state (model-free)
          secondary model error: mean recorded one-step error of the original
                    single-state model (the signal of every earlier experiment)
  SCORE   x = log(window mean); z = (x - mean_healthy) / sd_healthy, using the
          50 healthy calibration episodes of that body (same window); score = |z|.
  DECIDE  body if |z| >= 2, else world. Healthy data only; no test-trial tuning.
Same recorded trials as every earlier experiment (results_history/episodes/);
nothing is simulated or trained. All 12 configs are scored; soft is the target.

Caveat stated up front: these 6,360 trials have now been analysed many times.
A positive result here is a hypothesis to confirm on fresh trials (new master
seed), not a finding.

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

import compliant_config as kc
import factorial_config as fc
import metrics
import run_factorial as rf
from run_compliant import auc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_twosided")
W0, W1, Z = 20, 100, 2.0
LOG = []


def say(x=""):
    print(x, flush=True); LOG.append(x)


def rate(rs, pred):
    k = int(sum(bool(pred(r)) for r in rs)); lo, hi = metrics.wilson(k, len(rs))
    return dict(k=k, n=len(rs), rate=k / len(rs), lo=lo, hi=hi)


def run_config(name):
    specs = rf.make_specs()
    d = np.load(os.path.join(HERE, "results_history", "episodes", f"{name}.npz"))
    assert list(d["seed"]) == [s["seed"] for s in specs]
    obs, err = d["obs"], d["err"].astype(np.float64)
    motion = np.linalg.norm(obs[:, 1:] - obs[:, :-1], axis=2).astype(np.float64)
    is_cal = np.array([s["cell"] == "calib" for s in specs])
    rows = [i for i, s in enumerate(specs) if s["cell"] != "calib"]
    out = dict(config=name, signals={})
    for sig, X in (("motion", motion), ("model_error", err)):
        x = np.log(np.array([X[i, s["onset"] + W0:s["onset"] + W1].mean() for i, s in enumerate(specs)]))
        mu, sd = x[is_cal].mean(), x[is_cal].std(ddof=1)
        z = (x - mu) / sd
        for i, s in enumerate(specs):
            s[f"{sig}_logmean"], s[f"{sig}_z"] = float(x[i]), float(z[i])
            s[f"{sig}_verdict"] = "body" if abs(z[i]) >= Z else "world"
        g = lambda cell, f=np.abs: f(z[[i for i, s in enumerate(specs) if s["cell"] == cell]])
        ident = lambda v: v
        body = lambda i: abs(z[i]) >= Z
        R = lambda cell: [i for i in rows if specs[i]["cell"] == cell]
        out["signals"][sig] = dict(
            healthy_mean_log=float(mu), healthy_sd_log=float(sd),
            auc_body_vs_undisturbed=auc(g("body_only"), g("neither")),
            auc_body_vs_push=auc(g("body_only"), g("world_only")),
            auc_bodypush_vs_push=auc(g("body_world"), g("world_only")),
            onesided_auc_body_vs_undisturbed=auc(g("body_only", ident), g("neither", ident)),
            median_z={cell: float(np.median(g(cell, ident))) for cell in rf.CELLS},
            frac_quieter_body_only=float(np.mean(g("body_only", ident) < 0)),
            detection=rate([i for i in rows if specs[i]["has_body"]], body),
            detect_body_only=rate(R("body_only"), body),
            detect_body_push=rate(R("body_world"), body),
            false_positive=rate(R("world_only"), body),
            masking=rate(R("body_world"), lambda i: not body(i)),
            undisturbed_flagged=rate(R("neither"), body),
            fp_by_duration={dn: rate([i for i in R("world_only") if specs[i]["duration_name"] == dn], body)
                            for dn in fc.PUSH_DURATIONS},
            fp_by_torque={f"{t:g}": rate([i for i in R("world_only") if abs(specs[i]["torque"]) == t], body)
                          for t in fc.PUSH_TORQUES},
        )
    keys = ["cell", "seed", "has_body", "has_push", "param", "factor", "torque", "duration_name", "onset"]
    for sig in ("motion", "model_error"):
        keys += [f"{sig}_logmean", f"{sig}_z", f"{sig}_verdict"]
    with open(os.path.join(OUT, f"trials_{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config"] + keys); w.writeheader()
        w.writerows([{"config": name, **{k: s[k] for k in keys}} for s in specs])
    json.dump(out, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    versions = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                    gymnasium=gymnasium.__version__, master_seed=fc.MASTER_SEED, window=[W0, W1], z=Z,
                    episodes="results_history/episodes (recorded; controller = original single-state model)")
    json.dump(versions, open(os.path.join(OUT, "versions_and_seed.json"), "w"), indent=1)
    say("SIMULATED compliant body, not hardware. Recorded trials only; nothing simulated or trained.")
    say(f"versions/seed: {versions}")
    res = []
    for s in kc.STIFFNESS:
        for n in kc.NOISE:
            r = run_config(f"{s}_{n}"); res.append(r)
    for sig in ("motion", "model_error"):
        say(f"\n=== signal: {sig}  (two-sided |z| vs healthy calibration, window onset+{W0}..+{W1}, body if |z| >= {Z:g}) ===")
        say(f"  {'config':<15}{'AUC b/undist':>13}{'(1-sided)':>10}{'AUC b/push':>11}{'AUC b+p/push':>13}"
            f"{'detect/240':>11}{'b-only':>7}{'b+push':>7}{'FP/120':>7}{'short':>6}{'medium':>7}{'undist':>7}"
            f"{'  median z: undist / body / push':>34}{'body quieter':>13}")
        for r in res:
            m = r["signals"][sig]
            say(f"  {r['config']:<15}{m['auc_body_vs_undisturbed']:>13.3f}{m['onesided_auc_body_vs_undisturbed']:>10.3f}"
                f"{m['auc_body_vs_push']:>11.3f}{m['auc_bodypush_vs_push']:>13.3f}{m['detection']['k']:>11}"
                f"{m['detect_body_only']['k']:>7}{m['detect_body_push']['k']:>7}{m['false_positive']['k']:>7}"
                f"{m['fp_by_duration']['short']['k']:>6}{m['fp_by_duration']['medium']['k']:>7}{m['undisturbed_flagged']['k']:>7}"
                f"{m['median_z']['neither']:>12.2f} /{m['median_z']['body_only']:>6.2f} /{m['median_z']['world_only']:>6.2f}"
                f"{m['frac_quieter_body_only']:>12.0%}")
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    sys.exit(main())
