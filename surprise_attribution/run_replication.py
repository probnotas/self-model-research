"""Fresh-trial replication of the behaviour-change score. Rules: PREREG_replication.md.

    python run_replication.py             # record fresh episodes (cached) and score once
    python run_replication.py --only soft_none

Fresh master seed 777000; same physics, models, controller and episode loop as
before. Nothing here may be tuned: the window, threshold and rules are the ones
committed in PREREG_replication.md before this run.

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
import compliant_trials as ct
import factorial_config as fc
import metrics
import overlap_trials as ot

fc.MASTER_SEED = 777000                      # fresh seed: must be set before make_specs is called
import run_factorial as rf                   # noqa: E402
import run_history as rh                     # noqa: E402
from run_compliant import auc                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_replication")
EP = os.path.join(OUT, "episodes")
CONFIGS = {"soft_none": (4.0, "none"), "soft_low": (4.0, "low"),
           "rigid_none": (None, "none"), "medium_none": (30.0, "none")}
TARGETS = ("soft_none", "soft_low")
W0, W1, Z = 20, 100, 2.0
LOG = []


def say(x=""):
    print(x, flush=True); LOG.append(x)


def record(name, cfg):
    path = os.path.join(EP, f"{name}.npz")
    specs = rf.make_specs()
    assert specs[0]["seed"] == 777000
    if not os.path.exists(path):
        mp = rh.single_model_path(name, cfg)
        for s in specs:
            s.update(cfg=cfg, model_path=mp, steps=ot.STEPS, config=name)
        import multiprocessing as mpc
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
    return specs, d


def rate(rs, pred):
    k = int(sum(bool(pred(r)) for r in rs)); lo, hi = metrics.wilson(k, len(rs))
    return dict(k=k, n=len(rs), rate=k / len(rs), lo=lo, hi=hi)


def score(name, specs, d):
    obs, err = d["obs"], d["err"].astype(np.float64)
    motion = np.linalg.norm(obs[:, 1:] - obs[:, :-1], axis=2).astype(np.float64)
    is_cal = np.array([s["cell"] == "calib" for s in specs])
    rows = [i for i, s in enumerate(specs) if s["cell"] != "calib"]
    R = lambda c: [i for i in rows if specs[i]["cell"] == c]
    out = dict(config=name, rules={})
    zs = {}
    for sig, X in (("motion", motion), ("model_error", err)):
        x = np.log(np.array([X[i, s["onset"] + W0:s["onset"] + W1].mean() for i, s in enumerate(specs)]))
        zs[sig] = (x - x[is_cal].mean()) / x[is_cal].std(ddof=1)
    rules = {"A_motion_two_sided": (np.abs(zs["motion"]), lambda z: abs(z) >= Z, zs["motion"]),
             "B_motion_signed": (-zs["motion"], lambda z: z <= -Z, zs["motion"]),
             "C_model_error_two_sided": (np.abs(zs["model_error"]), lambda z: abs(z) >= Z, zs["model_error"])}
    for rule, (sc, is_body, z) in rules.items():
        body = lambda i: is_body(z[i])
        g = lambda c: sc[R(c)]
        out["rules"][rule] = dict(
            auc_body_vs_undisturbed=auc(g("body_only"), g("neither")),
            auc_body_vs_push=auc(g("body_only"), g("world_only")),
            detection=rate([i for i in rows if specs[i]["has_body"]], body),
            detect_body_only=rate(R("body_only"), body), detect_body_push=rate(R("body_world"), body),
            false_positive=rate(R("world_only"), body), masking=rate(R("body_world"), lambda i: not body(i)),
            undisturbed_flagged=rate(R("neither"), body),
            median_z={c: float(np.median(z[R(c)])) for c in rf.CELLS})
        for i, s in enumerate(specs):
            s[f"{rule}_verdict"] = "body" if body(i) else "world"
    for i, s in enumerate(specs):
        s["motion_z"], s["model_error_z"] = float(zs["motion"][i]), float(zs["model_error"][i])
    keys = ["cell", "seed", "has_body", "has_push", "param", "factor", "torque", "duration_name", "onset",
            "motion_z", "model_error_z"] + [f"{r}_verdict" for r in rules]
    with open(os.path.join(OUT, f"trials_{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config"] + keys); w.writeheader()
        w.writerows([{"config": name, **{k: s[k] for k in keys}} for s in specs])
    json.dump(out, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default="")
    only = set(filter(None, ap.parse_args().only.split(",")))
    os.makedirs(OUT, exist_ok=True)
    versions = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                    gymnasium=gymnasium.__version__, master_seed=777000, window=[W0, W1], z=Z,
                    preregistration="PREREG_replication.md")
    json.dump(versions, open(os.path.join(OUT, "versions_and_seed.json"), "w"), indent=1)
    say("SIMULATED compliant body, not hardware. FRESH trials, master seed 777000. Rules: PREREG_replication.md.")
    res = {}
    for name, (k, n) in CONFIGS.items():
        if only and name not in only and not os.path.exists(os.path.join(EP, f"{name}.npz")):
            continue
        specs, d = record(name, dict(stiffness=k, noise=kc.NOISE[n]))
        res[name] = score(name, specs, d)
    for name, r in res.items():
        say(f"\n== {name} ==" + ("  [TARGET]" if name in TARGETS else "  [control]"))
        for rule, m in r["rules"].items():
            na = " (not applicable on medium; reported for completeness)" if rule.startswith("B") and name == "medium_none" else ""
            say(f"  {rule:<25} AUC body/undist {m['auc_body_vs_undisturbed']:.3f} | body/push {m['auc_body_vs_push']:.3f} | "
                f"detect {m['detection']['k']:>3}/240 (b-only {m['detect_body_only']['k']}, b+push {m['detect_body_push']['k']}) | "
                f"FP {m['false_positive']['k']:>3}/120 | mask {m['masking']['k']:>3}/120 | undist {m['undisturbed_flagged']['k']:>3}/120 | "
                f"median z undist/body/push {m['median_z']['neither']:+.2f}/{m['median_z']['body_only']:+.2f}/{m['median_z']['world_only']:+.2f}{na}")
    if all(t in res for t in TARGETS):
        say("\nPRE-REGISTERED SUCCESS TEST (both soft configs):")
        verdicts = {}
        for t in TARGETS:
            a, b = res[t]["rules"]["A_motion_two_sided"], res[t]["rules"]["B_motion_signed"]
            s1 = a["auc_body_vs_undisturbed"] >= 0.75
            s2 = b["auc_body_vs_push"] >= 0.75 and b["false_positive"]["k"] <= 15
            verdicts[t] = (s1, s2)
            say(f"  {t}: S1 (A AUC body/undist {a['auc_body_vs_undisturbed']:.3f} >= 0.75) {'PASS' if s1 else 'FAIL'} | "
                f"S2 (B AUC body/push {b['auc_body_vs_push']:.3f} >= 0.75 and FP {b['false_positive']['k']} <= 15) {'PASS' if s2 else 'FAIL'}")
        both = [s1 and s2 for s1, s2 in verdicts.values()]
        s1_all = all(s1 for s1, _ in verdicts.values())
        outcome = "REPLICATED" if all(both) else ("PARTIALLY REPLICATED" if any(both) or s1_all else "FAILED")
        say(f"  OUTCOME: {outcome}")
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    sys.exit(main())
