"""Does the attribution rule survive a (simulated) compliant, noisy body?

    python run_compliant.py                      # every (stiffness, noise) config; resumes from cache
    python run_compliant.py --only soft_low,soft_medium   # just these (shorter jobs)
    python run_compliant.py --summary            # rebuild summary/figure from finished configs only

For each configuration in compliant_config.py:
  1. train a forward model on that body's own random-action data (as sensed);
     rigid/no-noise uses the original model, so it reproduces run_factorial.py;
  2. report held-out one-step prediction error;
  3. run the exact 2x2 factorial from run_factorial.py (same seeds and draws);
  4. score it with the unchanged rule and record rates and margins.

This is a SIMULATED compliant body (series-elastic + hysteresis + damping +
Gaussian sensor noise), a model of compliance, not real hardware.
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
import alarm_config as ac
import classifier
import compliant_config as kc
import compliant_env as ce
import compliant_trials as ct
import factorial_config as fc
import metrics
import model as model_lib
import overlap_trials as ot
import plot_compliant
import run_factorial as rf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, kc.RESULTS)
LOG = []


def say(msg=""):
    print(msg, flush=True)
    LOG.append(msg)


def auc(pos, neg):
    """P(random body-present score > random body-absent score); 0.5 = no separation."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean(); eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def run_config(name, cfg, alarm_thr):
    cache = os.path.join(OUT, f"{name}.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    if name == "rigid_none":
        model_path = os.path.join(HERE, "forward_model.pt")
        m = model_lib.load_model(model_path)
    else:
        m, model_path = ct.get_model(name, cfg)
    heldout = ct.heldout_error(m, cfg)
    specs = rf.make_specs()
    for s in specs:
        s.update(cfg=cfg, model_path=model_path, steps=ot.STEPS, config=name)
    res = ct.run_all(specs, kc.N_WORKERS, name)

    calib = [r for r in res if r["cell"] == "calib"]
    cs = [A.settled_score(r["err"], r["theta"], r["thdot"], r["onset"])[0] for r in calib]
    finite = [s for s in cs if s is not None]
    body_thr = classifier.calibrate_threshold(finite) if len(finite) >= 2 else float("nan")
    rf.score(res, body_thr, alarm_thr)
    rows = [r for r in res if r["cell"] != "calib"]
    cell = lambda c: [r for r in rows if r["cell"] == c]

    def rate(rs, pred):
        k = sum(pred(r) for r in rs); lo, hi = metrics.wilson(k, len(rs))
        return dict(k=k, n=len(rs), rate=k / len(rs), lo=lo, hi=hi)

    sb = [r["settled_score"] for r in rows if r["has_body"] and r["settled"]]
    sn = [r["settled_score"] for r in rows if not r["has_body"] and r["settled"]]
    out = dict(
        config=name, stiffness=cfg["stiffness"], noise=list(cfg["noise"]), heldout_error=heldout,
        body_threshold=body_thr, calib_settled=len(finite), calib_n=len(calib),
        calib_score_mean=float(np.mean(finite)) if finite else float("nan"),
        confusion={c: {v: sum(r["verdict"] == v for r in cell(c)) for v in ("body", "world", "alarm")}
                   for c in rf.CELLS},
        false_positive=rate(cell("world_only"), lambda r: r["verdict"] != "world"),
        fp_body=sum(r["verdict"] == "body" for r in cell("world_only")),
        fp_alarm=sum(r["verdict"] == "alarm" for r in cell("world_only")),
        false_alarm_neither=rate(cell("neither"), lambda r: r["verdict"] != "world"),
        masking=rate(cell("body_world"), lambda r: r["verdict"] == "world"),
        miss_body_only=rate(cell("body_only"), lambda r: r["verdict"] == "world"),
        misattribution=rate(rows, lambda r: not r["correct"]),
        settled_body=len(sb), settled_nobody=len(sn),
        gap=(min(sb) - max(sn)) if sb and sn else float("nan"),
        body_min=min(sb) if sb else float("nan"), nobody_max=max(sn) if sn else float("nan"),
        settled_auc=auc(sb, sn),
        by_push={f"{tq:g}": dict(
            fp=sum(r["verdict"] != "world" for r in cell("world_only") if abs(r["torque"]) == tq),
            mask=sum(r["verdict"] == "world" for r in cell("body_world") if abs(r["torque"]) == tq),
            n=sum(abs(r["torque"]) == tq for r in cell("body_world"))) for tq in fc.PUSH_TORQUES},
    )
    keys = ["config", "cell", "seed", "has_body", "has_push", "param", "factor", "torque",
            "duration_name", "offset_s", "onset", "verdict", "correct", "path", "verdict_time_s",
            "settled", "settled_score", "settle_time_s", "late_block_median", "late_frac_above_alarm"]
    with open(os.path.join(OUT, f"trials_{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        w.writerows([{k: r[k] for k in keys} for r in res])
    json.dump(out, open(cache, "w"), indent=1)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default="", help="comma-separated config names, e.g. soft_low")
    ap.add_argument("--summary", action="store_true", help="only summarise finished configs")
    args = ap.parse_args()
    only = set(filter(None, args.only.split(",")))
    os.makedirs(OUT, exist_ok=True)
    assert ce.verify_rigid() == 0.0, "rigid mode of the compliant env differs from the existing env"
    versions = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                    gymnasium=gymnasium.__version__, master_seed=fc.MASTER_SEED,
                    compliance=dict(J_MOTOR=ce.J_MOTOR, B_MOTOR=ce.B_MOTOR, B_JOINT=ce.B_JOINT,
                                    HYST=ce.HYST, SPRING_ZETA=ce.SPRING_ZETA, N_SUBSTEPS=ce.N_SUBSTEPS))
    json.dump(versions, open(os.path.join(OUT, "versions_and_seed.json"), "w"), indent=1)
    alarm_thr = json.load(open(os.path.join(HERE, ac.RESULTS, "chosen_alarm.json")))["alarm_threshold"]
    say("SIMULATED compliant body (series-elastic + hysteresis + damping + sensor noise): "
        "a model of compliance, not real hardware.")
    say(f"versions/seed: {versions}")

    results = []
    for sname, k in kc.STIFFNESS.items():
        for nname, nz in kc.NOISE.items():
            name = f"{sname}_{nname}"
            cached = os.path.exists(os.path.join(OUT, f"{name}.json"))
            if (only and name not in only and not cached) or (args.summary and not cached):
                continue
            say(f"\n== {name} (stiffness {k}, noise {nz}) ==")
            r = run_config(name, dict(stiffness=k, noise=nz), alarm_thr)
            results.append(r)
            say(f"  held-out error {r['heldout_error']:.4f} | body thr {r['body_threshold']:.4f} | "
                f"FP {r['false_positive']['k']}/{r['false_positive']['n']} (body {r['fp_body']}, alarm {r['fp_alarm']}) | "
                f"masking {r['masking']['k']}/{r['masking']['n']} | body-only missed "
                f"{r['miss_body_only']['k']}/{r['miss_body_only']['n']} | neither flagged "
                f"{r['false_alarm_neither']['k']}/{r['false_alarm_neither']['n']} | gap {r['gap']:+.4f} | AUC {r['settled_auc']:.3f}")

    say("\nSUMMARY (one row per configuration)")
    say(f"  {'config':<15}{'heldout':>8}{'FP':>9}{'mask':>9}{'miss(b)':>9}{'flag(n)':>9}"
        f"{'misattr':>9}{'gap':>9}{'AUC':>7}{'settled b/nb':>14}")
    for r in results:
        say(f"  {r['config']:<15}{r['heldout_error']:>8.4f}{r['false_positive']['rate']:>9.1%}"
            f"{r['masking']['rate']:>9.1%}{r['miss_body_only']['rate']:>9.1%}"
            f"{r['false_alarm_neither']['rate']:>9.1%}{r['misattribution']['rate']:>9.1%}"
            f"{r['gap']:>+9.4f}{r['settled_auc']:>7.3f}{r['settled_body']:>7}/{r['settled_nobody']}")
    json.dump(results, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    figs = plot_compliant.sweep(results, os.path.join(OUT, "sweep.png"))
    say(f"\nfigure: {os.path.relpath(figs, HERE)}")
    open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    sys.exit(main())
