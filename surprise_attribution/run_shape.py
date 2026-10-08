"""Attribute by the SHAPE of the error after onset, with no settle gate.

    python run_shape.py

Paired comparison on the SAME recorded trials as results_compliant/ and
results_history/ (same configs, factorial, MASTER_SEED 90000, controller and
single-state error traces). Nothing is simulated or trained here: the traces
in results_history/episodes/ are re-scored. The old rule's verdicts come from
the logged results_compliant/ CSVs; every trace is first checked by
recomputing the old rule's settled score from it and comparing with the log.
See shape_config.py for the feature, the window and how the decision point
is set (healthy calibration episodes only).

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

import alarm_classifier as A
import compliant_config as kc
import factorial_config as fc
import metrics
import plot_shape
import run_factorial as rf
import shape_config as sc
from run_compliant import auc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, sc.RESULTS)
OLD = os.path.join(HERE, kc.RESULTS)
LOG = []


def say(msg=""):
    print(msg, flush=True)
    LOG.append(msg)


def load(name):
    """Specs + recorded single-state error traces + logged old-rule results."""
    specs = rf.make_specs()
    d = np.load(os.path.join(HERE, sc.EPISODES, f"{name}.npz"))
    assert list(d["seed"]) == [s["seed"] for s in specs]
    logged = list(csv.DictReader(open(os.path.join(OLD, f"trials_{name}.csv"))))
    bad = 0
    for s, e, th, w, row in zip(specs, d["err"], d["theta"], d["thdot"], logged):
        assert int(row["seed"]) == s["seed"]
        sc_, _ = A.settled_score(e, th, w, s["onset"])
        a = float(row["settled_score"])
        bad += not ((sc_ is None and np.isnan(a)) or (sc_ is not None and sc_ == a))
        s.update(config=name, err=e.astype(np.float64), old_verdict=row["verdict"], old_path=row["path"],
                 old_settled=row["settled"] == "True", old_settled_score=a)
    return specs, bad


def shape(err, onset, W):
    h = W // 2
    early = float(err[onset:onset + h].mean())
    late = float(err[onset + h:onset + W].mean())
    return early, late, float(np.log(late / early))


def rate(rs, pred):
    k = sum(pred(r) for r in rs); lo, hi = metrics.wilson(k, len(rs))
    return dict(k=k, n=len(rs), rate=k / len(rs) if rs else float("nan"), lo=lo, hi=hi)


def verdict_metrics(rows, v):
    """v(r) -> 'body' / 'world' / 'alarm'. Detection = body change called body or alarm."""
    cell = lambda c: [r for r in rows if r["cell"] == c]
    det = lambda r: v(r) in ("body", "alarm")
    out = dict(
        detection=rate([r for r in rows if r["has_body"]], det),
        detection_body_only=rate(cell("body_only"), det),
        detection_body_world=rate(cell("body_world"), det),
        false_positive=rate(cell("world_only"), det),
        masking=rate(cell("body_world"), lambda r: not det(r)),
        false_alarm_neither=rate(cell("neither"), det),
        misattribution=rate(rows, lambda r: det(r) != r["has_body"]),
        fp_by_duration={d: rate([r for r in cell("world_only") if r["duration_name"] == d], det)
                        for d in fc.PUSH_DURATIONS},
        fp_by_torque={f"{t:g}": rate([r for r in cell("world_only") if abs(r["torque"]) == t], det)
                      for t in fc.PUSH_TORQUES},
        mask_by_duration={d: rate([r for r in cell("body_world") if r["duration_name"] == d],
                                  lambda r: not det(r)) for d in fc.PUSH_DURATIONS},
    )
    return out


def run_config(name, cfg):
    res, bad = load(name)
    calib = [r for r in res if r["cell"] == "calib"]
    rows = [r for r in res if r["cell"] != "calib"]
    old_json = json.load(open(os.path.join(OLD, f"{name}.json")))
    out = dict(config=name, stiffness=cfg["stiffness"], noise=list(cfg["noise"]), trace_check_mismatches=bad,
               old=dict(settled_auc=old_json["settled_auc"], settled_body=old_json["settled_body"],
                        settled_nobody=old_json["settled_nobody"],
                        **verdict_metrics(rows, lambda r: r["old_verdict"])),
               shape={}, shape_elev={})
    for W in sc.W_SWEEP:
        for r in res:
            r[f"early_{W}"], r[f"late_{W}"], r[f"logr_{W}"] = shape(r["err"], r["onset"], W)
        hr = np.array([r[f"logr_{W}"] for r in calib]); hl = np.array([r[f"late_{W}"] for r in calib])
        tau = float(hr.mean() - sc.Z_SHAPE * hr.std(ddof=1))
        elev = float(hl.mean() + sc.Z_ELEV * hl.std(ddof=1))
        for r in res:
            stays = r[f"logr_{W}"] >= tau
            r[f"shape_{W}"] = "body" if stays else "world"
            r[f"shape_elev_{W}"] = "body" if (stays and r[f"late_{W}"] >= elev) else "world"
        s = lambda rs: [r[f"logr_{W}"] for r in rs]
        body = [r for r in rows if r["has_body"]]; nob = [r for r in rows if not r["has_body"]]
        cell = lambda c: [r for r in rows if r["cell"] == c]
        common = dict(
            tau=tau, elev_threshold=elev, healthy_logr_mean=float(hr.mean()), healthy_logr_sd=float(hr.std(ddof=1)),
            auc_all=auc(s(body), s(nob)),
            auc_bodyonly_vs_worldonly=auc(s(cell("body_only")), s(cell("world_only"))),
            auc_on_old_settled_subset=auc(s([r for r in body if r["old_settled"]]),
                                          s([r for r in nob if r["old_settled"]])),
            median_logr={c: float(np.median(s(cell(c)))) for c in rf.CELLS},
        )
        out["shape"][W] = dict(common, **verdict_metrics(rows, lambda r: r[f"shape_{W}"]))
        out["shape_elev"][W] = dict(common, **verdict_metrics(rows, lambda r: r[f"shape_elev_{W}"]))
    keys = ["config", "cell", "seed", "has_body", "has_push", "param", "factor", "torque", "duration_name",
            "offset_s", "onset", "old_verdict", "old_path", "old_settled", "old_settled_score"]
    for W in sc.W_SWEEP:
        keys += [f"early_{W}", f"late_{W}", f"logr_{W}", f"shape_{W}", f"shape_elev_{W}"]
    with open(os.path.join(OUT, f"trials_{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        w.writerows([{k: r[k] for k in keys} for r in res])
    json.dump(out, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    versions = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                    gymnasium=gymnasium.__version__, master_seed=fc.MASTER_SEED, w_sweep=list(sc.W_SWEEP),
                    w_main=sc.W_MAIN, z_shape=sc.Z_SHAPE, z_elev=sc.Z_ELEV,
                    traces=sc.EPISODES + " (single-state error, recorded by run_history.py)",
                    old_rule="logged verdicts in " + kc.RESULTS)
    json.dump(versions, open(os.path.join(OUT, "versions_and_seed.json"), "w"), indent=1)
    say("SIMULATED compliant body (series-elastic + hysteresis + damping + sensor noise): "
        "a model of compliance, not real hardware.")
    say(f"versions/seed: {versions}")
    results = []
    for s, k in kc.STIFFNESS.items():
        for n, nz in kc.NOISE.items():
            r = run_config(f"{s}_{n}", dict(stiffness=k, noise=nz))
            results.append(r)
            o = r["old"]
            say(f"\n== {r['config']} == trace check: {r['trace_check_mismatches']} mismatches of 530")
            say(f"  old rule      AUC(settled only, {o['settled_body']}/{o['settled_nobody']}) {o['settled_auc']:.3f} | "
                f"detect {o['detection']['k']:>3}/240 | FP {o['false_positive']['k']:>3} | mask {o['masking']['k']:>3} | "
                f"neither flagged {o['false_alarm_neither']['k']:>3} | misattr {o['misattribution']['rate']:.1%}")
            for W in sc.W_SWEEP:
                for tag in ("shape", "shape_elev"):
                    m = r[tag][W]
                    head = (f"AUC all {m['auc_all']:.3f} b-only/w-only {m['auc_bodyonly_vs_worldonly']:.3f} "
                            f"settled-subset {m['auc_on_old_settled_subset']:.3f}") if tag == "shape" else " " * 54
                    say(f"  {tag:<10} W={W:<3}{head} | detect {m['detection']['k']:>3}/240 | FP {m['false_positive']['k']:>3} "
                        f"(short {m['fp_by_duration']['short']['k']}/60, medium {m['fp_by_duration']['medium']['k']}/60) | "
                        f"mask {m['masking']['k']:>3} | neither flagged {m['false_alarm_neither']['k']:>3} | "
                        f"misattr {m['misattribution']['rate']:.1%}")
    W = sc.W_MAIN
    say(f"\nSUMMARY (W = {W}): old settle rule vs shape rule, same trials. Counts: detect of 240, others of 120")
    say(f"  {'config':<15}{'AUC old':>8}{'AUC new':>9}{'det old':>9}{'shape':>7}{'+elev':>7}{'FP old':>8}{'shape':>7}"
        f"{'+elev':>7}{'mask old':>10}{'shape':>7}{'+elev':>7}{'neither old':>13}{'shape':>7}{'+elev':>7}")
    for r in results:
        o, a, b = r["old"], r["shape"][W], r["shape_elev"][W]
        say(f"  {r['config']:<15}{o['settled_auc']:>8.3f}{a['auc_all']:>9.3f}{o['detection']['k']:>9}{a['detection']['k']:>7}"
            f"{b['detection']['k']:>7}{o['false_positive']['k']:>8}{a['false_positive']['k']:>7}{b['false_positive']['k']:>7}"
            f"{o['masking']['k']:>10}{a['masking']['k']:>7}{b['masking']['k']:>7}{o['false_alarm_neither']['k']:>13}"
            f"{a['false_alarm_neither']['k']:>7}{b['false_alarm_neither']['k']:>7}")
    json.dump(results, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    fig = plot_shape.sweep(results, os.path.join(OUT, "sweep_shape.png"))
    say(f"\nfigure: {os.path.relpath(fig, HERE)}")
    open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    sys.exit(main())
