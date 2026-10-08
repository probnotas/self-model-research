"""Per-body calibration: score surprise relative to the body's own healthy baseline.

    python run_calibrated.py                     # every config; resumes from cached traces
    python run_calibrated.py --only soft_low     # regenerate traces for just these configs
    python run_calibrated.py --summary           # score/summarise configs whose traces exist

A paired comparison on the SAME trials as results_compliant/: same configs,
same factorial (MASTER_SEED 90000), same start states, same trained models.
The only change is the signal the unchanged rule (alarm_classifier.decide)
reads: raw error e_t (old) vs z_t = (e_t - mu_body) / spread_body (new).
See calibrated_config.py for the baseline window and z thresholds.

Why traces are regenerated: results_compliant/ logged one row per trial
(settled score, verdict, ...) but not the per-step error traces, and z-scoring
needs the per-step signal. So the identical, deterministic pipeline from
run_compliant.py is re-run (same code, same cached models, same seeds, one
torch thread per worker as before), and every trial's raw settled score and
old verdict are checked against the logged CSV before anything new is scored.

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
import alarm_config as ac
import calibrated_config as kc2
import compliant_config as kc
import compliant_trials as ct
import factorial_config as fc
import metrics
import model as model_lib
import overlap_trials as ot
import plot_calibrated
import run_factorial as rf
from run_compliant import auc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, kc2.RESULTS)
TRACE_DIR = os.path.join(OUT, kc2.TRACES)
OLD = os.path.join(HERE, kc.RESULTS)
LOG = []


def say(msg=""):
    print(msg, flush=True)
    LOG.append(msg)


# ---------------------------------------------------------------------------
# 1. the same trials, with their per-step traces
# ---------------------------------------------------------------------------

def traces(name, cfg, regenerate):
    """Specs + per-step (err, theta, thdot) for one config, from cache or by re-running."""
    path = os.path.join(TRACE_DIR, f"{name}.npz")
    specs = rf.make_specs()
    if not os.path.exists(path):
        if not regenerate:
            return None
        model_path = (os.path.join(HERE, "forward_model.pt") if name == "rigid_none"
                      else ct.get_model(name, cfg)[1])       # cached model, not retrained
        for s in specs:
            s.update(cfg=cfg, model_path=model_path, steps=ot.STEPS, config=name)
        res = ct.run_all(specs, kc.N_WORKERS, name)
        os.makedirs(TRACE_DIR, exist_ok=True)
        np.savez_compressed(path, seed=[r["seed"] for r in res], err=np.stack([r["err"] for r in res]),
                            theta=np.stack([r["theta"] for r in res]),
                            thdot=np.stack([r["thdot"] for r in res]))
    d = np.load(path)
    assert list(d["seed"]) == [s["seed"] for s in specs], f"{name}: trace order differs from specs"
    for s, e, th, w in zip(specs, d["err"], d["theta"], d["thdot"]):
        s.update(config=name, err=e, theta=th, thdot=w)
    return specs


def check_reproduction(name, res):
    """Compare the re-run's raw settled scores and old verdicts with the logged CSV."""
    logged = list(csv.DictReader(open(os.path.join(OLD, f"trials_{name}.csv"))))
    assert len(logged) == len(res)
    bad_score = bad_verdict = 0
    for r, row in zip(res, logged):
        assert int(row["seed"]) == r["seed"]
        a, b = float(row["settled_score"]), r["raw_settled_score"]
        if not ((np.isnan(a) and np.isnan(b)) or a == b):
            bad_score += 1
        if row["verdict"] != r["old_verdict"]:
            bad_verdict += 1
        r["logged_verdict"], r["logged_settled_score"] = row["verdict"], a
    return bad_score, bad_verdict


# ---------------------------------------------------------------------------
# 2. per-body baseline from the undisturbed calibration episodes
# ---------------------------------------------------------------------------

def baseline(res):
    """Per-step error of the calibration episodes inside the rule's decision window."""
    e = np.concatenate([r["err"][r["onset"] + 1:r["onset"] + ac.EXT_MAX_LAG]
                        for r in res if r["cell"] == "calib"]).astype(np.float64)
    med = float(np.median(e))
    return dict(n_steps=int(e.size), mean=float(e.mean()), std=float(e.std(ddof=1)),
                median=med, mad=float(kc2.MAD_SCALE * np.median(np.abs(e - med))))


def centre_spread(b, kind):
    return (b["mean"], b["std"]) if kind == "std" else (b["median"], b["mad"])


# ---------------------------------------------------------------------------
# 3. scoring: the unchanged rule on a given signal
# ---------------------------------------------------------------------------

def apply_rule(r, sig, body_thr, alarm_thr, tag):
    s, t = A.settled_score(sig, r["theta"], r["thdot"], r["onset"])
    blocks = A.late_blocks(sig, r["onset"])
    v, _, path = A.decide(sig, r["theta"], r["thdot"], r["onset"], body_thr, alarm_thr)
    r.update({f"{tag}_verdict": v, f"{tag}_path": path,
              f"{tag}_settled_score": s if s is not None else float("nan"),
              f"{tag}_late_block_median": float(np.median(blocks)),
              f"{tag}_late_frac_above_alarm": float(np.mean(blocks > alarm_thr))})
    return s is not None, (t - r["onset"]) * 0.05 if s is not None else float("nan")


def metrics_for(rows, tag):
    cell = lambda c: [r for r in rows if r["cell"] == c]

    def rate(rs, pred):
        k = sum(pred(r) for r in rs); lo, hi = metrics.wilson(k, len(rs))
        return dict(k=k, n=len(rs), rate=k / len(rs), lo=lo, hi=hi)

    v = lambda r: r[f"{tag}_verdict"]
    sb = [r[f"{tag}_settled_score"] for r in rows if r["has_body"] and r["settled"]]
    sn = [r[f"{tag}_settled_score"] for r in rows if not r["has_body"] and r["settled"]]
    return dict(
        confusion={c: {x: sum(v(r) == x for r in cell(c)) for x in ("body", "world", "alarm")}
                   for c in rf.CELLS},
        false_positive=rate(cell("world_only"), lambda r: v(r) != "world"),
        fp_body=sum(v(r) == "body" for r in cell("world_only")),
        fp_alarm=sum(v(r) == "alarm" for r in cell("world_only")),
        masking=rate(cell("body_world"), lambda r: v(r) == "world"),
        miss_body_only=rate(cell("body_only"), lambda r: v(r) == "world"),
        false_alarm_neither=rate(cell("neither"), lambda r: v(r) != "world"),
        misattribution=rate(rows, lambda r: v(r) not in rf.CORRECT[r["cell"]]),
        settled_body=len(sb), settled_nobody=len(sn),
        gap=(min(sb) - max(sn)) if sb and sn else float("nan"),
        settled_auc=auc(sb, sn),
        paths={p: sum(r[f"{tag}_path"] == p for r in rows) for p in ("settled", "persistent", "decayed")},
    )


def score_config(name, res, old_body_thr, alarm_thr, z_thr):
    base = baseline(res)
    for r in res:
        r["settled"], r["settle_time_s"] = apply_rule(r, r["err"], old_body_thr, alarm_thr, "old")
        r["raw_settled_score"] = r["old_settled_score"]
        for kind in kc2.SPREADS:
            mu, sd = centre_spread(base, kind)
            tag = "new" if kind == "std" else "new_mad"
            apply_rule(r, (r["err"].astype(np.float64) - mu) / sd,
                       z_thr[kind]["body"], z_thr[kind]["alarm"], tag)
    bad_score, bad_verdict = check_reproduction(name, res)
    rows = [r for r in res if r["cell"] != "calib"]
    out = dict(config=name, baseline=base, old_body_threshold=old_body_thr, alarm_threshold=alarm_thr,
               z_thresholds=z_thr, repro_settled_score_mismatches=bad_score,
               repro_verdict_mismatches=bad_verdict, n_trials=len(res),
               old=metrics_for(rows, "old"), new=metrics_for(rows, "new"),
               new_mad=metrics_for(rows, "new_mad"))
    keys = ["config", "cell", "seed", "has_body", "has_push", "param", "factor", "torque",
            "duration_name", "offset_s", "onset", "settled", "settle_time_s",
            "raw_settled_score", "new_settled_score", "new_mad_settled_score",
            "old_verdict", "old_path", "new_verdict", "new_path", "new_mad_verdict", "new_mad_path",
            "old_late_block_median", "new_late_block_median",
            "old_late_frac_above_alarm", "new_late_frac_above_alarm", "new_mad_late_frac_above_alarm",
            "logged_settled_score", "logged_verdict"]
    ren = {"new_settled_score": "z_settled_score", "new_mad_settled_score": "z_mad_settled_score",
           "old_late_block_median": "raw_late_block_median", "new_late_block_median": "z_late_block_median"}
    with open(os.path.join(OUT, f"trials_{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[ren.get(k, k) for k in keys]); w.writeheader()
        w.writerows([{ren.get(k, k): r[k] for k in keys} for r in res])
    json.dump(out, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
    return out


# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default="", help="comma-separated configs whose traces to regenerate")
    ap.add_argument("--summary", action="store_true", help="never regenerate; score cached traces only")
    args = ap.parse_args()
    only = set(filter(None, args.only.split(",")))
    os.makedirs(OUT, exist_ok=True)
    versions = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                    gymnasium=gymnasium.__version__, master_seed=fc.MASTER_SEED,
                    reused_from=kc.RESULTS, models="forward_model.pt (rigid_none), compliant_models/*.pt")
    json.dump(versions, open(os.path.join(OUT, "versions_and_seed.json"), "w"), indent=1)
    alarm_thr = json.load(open(os.path.join(HERE, ac.RESULTS, "chosen_alarm.json")))["alarm_threshold"]
    say("SIMULATED compliant body (series-elastic + hysteresis + damping + sensor noise): "
        "a model of compliance, not real hardware.")
    say(f"versions/seed: {versions}")

    configs = [(f"{s}_{n}", dict(stiffness=k, noise=nz))
               for s, k in kc.STIFFNESS.items() for n, nz in kc.NOISE.items()]
    regen = lambda name: not args.summary and (not only or name in only or name == kc2.REFERENCE_CONFIG)

    # reference body first: its baseline fixes the z thresholds for every body
    ref_name, ref_cfg = configs[0]
    assert ref_name == kc2.REFERENCE_CONFIG
    ref = traces(ref_name, ref_cfg, regen(ref_name))
    ref_old = json.load(open(os.path.join(OLD, f"{ref_name}.json")))
    rb = baseline(ref)
    z_thr = {}
    for kind in kc2.SPREADS:
        mu, sd = centre_spread(rb, kind)
        z_thr[kind] = dict(body=(ref_old["body_threshold"] - mu) / sd, alarm=(alarm_thr - mu) / sd)
    say(f"\nreference baseline ({ref_name}): mean {rb['mean']:.5f} std {rb['std']:.5f} | "
        f"median {rb['median']:.5f} MAD {rb['mad']:.5f} | {rb['n_steps']} steps")
    say(f"z thresholds (carried over from {ref_name}, fixed for all bodies): "
        f"std: body {z_thr['std']['body']:.3f}, alarm {z_thr['std']['alarm']:.3f} | "
        f"mad: body {z_thr['mad']['body']:.3f}, alarm {z_thr['mad']['alarm']:.3f}")

    results = []
    for name, cfg in configs:
        res = ref if name == ref_name else traces(name, cfg, regen(name))
        if res is None:
            continue
        old_thr = json.load(open(os.path.join(OLD, f"{name}.json")))["body_threshold"]
        r = score_config(name, res, old_thr, alarm_thr, z_thr)
        results.append(r)
        b = r["baseline"]
        say(f"\n== {name} == baseline mean {b['mean']:.5f} std {b['std']:.5f} "
            f"(median {b['median']:.5f} MAD {b['mad']:.5f}); reproduction vs logged CSV: "
            f"{r['repro_settled_score_mismatches']} settled-score and "
            f"{r['repro_verdict_mismatches']} verdict mismatches of {r['n_trials']}")
        for tag in ("old", "new", "new_mad"):
            m = r[tag]
            say(f"  {tag:<8} FP {m['false_positive']['k']:>3}/120 (body {m['fp_body']}, alarm {m['fp_alarm']}) | "
                f"mask {m['masking']['k']:>3}/120 | miss(no push) {m['miss_body_only']['k']:>3}/120 | "
                f"flag(neither) {m['false_alarm_neither']['k']:>3}/120 | misattr {m['misattribution']['rate']:.1%} | "
                f"gap {m['gap']:+.4f} | AUC {m['settled_auc']:.3f} | paths {m['paths']}")

    say("\nSUMMARY: old (raw error, old thresholds) vs new (z vs own baseline, mean/std), same trials")
    say(f"  {'config':<15}{'FP old':>8}{'new':>6}{'mask old':>10}{'new':>6}{'miss old':>10}{'new':>6}"
        f"{'flag(n) old':>13}{'new':>6}{'AUC old':>9}{'new':>7}{'gap old':>10}{'gap new (z)':>13}")
    for r in results:
        o, n = r["old"], r["new"]
        say(f"  {r['config']:<15}{o['false_positive']['k']:>8}{n['false_positive']['k']:>6}"
            f"{o['masking']['k']:>10}{n['masking']['k']:>6}{o['miss_body_only']['k']:>10}"
            f"{n['miss_body_only']['k']:>6}{o['false_alarm_neither']['k']:>13}{n['false_alarm_neither']['k']:>6}"
            f"{o['settled_auc']:>9.3f}{n['settled_auc']:>7.3f}{o['gap']:>+10.4f}{n['gap']:>+13.3f}")
    say("  (counts out of 120 per cell)")
    json.dump(results, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    fig = plot_calibrated.sweep(results, os.path.join(OUT, "sweep_calibrated.png"))
    say(f"\nfigure: {os.path.relpath(fig, HERE)}")
    open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    sys.exit(main())
