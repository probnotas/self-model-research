"""3-way classifier (body / world / ALARM) vs the previous settle rule.

    python compare_alarm.py dev     # dev episodes; choose the alarm threshold
    python compare_alarm.py test    # fresh test episodes; evaluate once, paired

Discipline (same as compare_rules.py):
  - The only new free parameter, the alarm threshold, is chosen on DEV by the
    criterion below, written before any dev result was seen.
  - It is then frozen and the new rule is evaluated ONCE on TEST episodes with
    fresh seeds. Only test numbers are results.
  - The previous rule is scored on the IDENTICAL test episodes (paired).
  - The body/world threshold is set exactly as before for both rules:
    mean + 4 std of the settled score on that set's undisturbed episodes.

Alarm-threshold criterion (pre-stated):
  1. no false alarm on any dev world event (ordinary or big push) or any
     undisturbed dev episode;
  2. among those, the fewest dev severe-damage trials called "world" (missed);
  3. ties: the geometric midpoint of the tied thresholds, i.e. the most margin
     on both sides (if that midpoint itself violates 1 or 2, the tied grid
     value nearest it).

Correctness used in the summaries:
  body -> body, world -> world, severe -> alarm (ideal) or body (damage still
  acknowledged, adaptation triggered). severe -> world is a MISS: the dangerous
  case this change exists to remove.
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

import alarm_classifier as A
import alarm_config as ac
import alarm_trials
import classifier
import env_setup
import metrics
import model as model_lib
import plot_alarm
import run_experiment

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, ac.RESULTS)
DT = 0.05
TRUTHS = ("body", "world", "severe")
PREDS = (A.BODY, A.WORLD, A.ALARM)
LOG = []


def say(msg=""):
    print(msg, flush=True)
    LOG.append(msg)


# ---------------------------------------------------------------------------
# traces: simulate once, cache, reuse
# ---------------------------------------------------------------------------

def record(stage):
    path = os.path.join(OUT, f"{stage}_traces.npz")
    if os.path.exists(path):
        say(f"  loading cached {stage} traces")
        d = np.load(path)
        specs = json.loads(str(d["specs"]))
        for i, s in enumerate(specs):
            s["err"], s["theta"], s["thdot"] = d["err"][i], d["theta"][i], d["thdot"][i]
        return specs
    counts, base = (ac.N_DEV, ac.SEED_DEV) if stage == "dev" else (ac.N_TEST, ac.SEED_TEST)
    res = alarm_trials.run(alarm_trials.make_specs(counts, base), stage)
    keys = [k for k in res[0] if k not in ("err", "theta", "thdot")]
    np.savez_compressed(path, err=np.stack([r["err"] for r in res]),
                        theta=np.stack([r["theta"] for r in res]),
                        thdot=np.stack([r["thdot"] for r in res]),
                        specs=np.array(json.dumps([{k: r[k] for k in keys} for r in res])))
    return res


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def body_threshold_new(res):
    """Body/world threshold for the new rule: undisturbed settled scores."""
    s = [A.settled_score(r["err"], r["theta"], r["thdot"], r["onset"])[0]
         for r in res if r["category"] == "calib"]
    return classifier.calibrate_threshold([x for x in s if x is not None])


def body_threshold_prev(res):
    """Same procedure for the previous rule, from its own score function."""
    import classify_config as cc
    s = [classifier.calm_gated_score(r["err"], r["theta"], r["thdot"], r["onset"],
                                     calm_steps=2 * cc.CALM_STEPS)[0]
         for r in res if r["category"] == "calib"]
    return classifier.calibrate_threshold([x for x in s if np.isfinite(x)])


def run_new(res, body_thr, alarm_thr):
    return [A.decide(r["err"], r["theta"], r["thdot"], r["onset"], body_thr, alarm_thr) for r in res]


def run_prev(res, thr):
    return [A.previous_rule(r["err"], r["theta"], r["thdot"], r["onset"], thr) for r in res]


def is_correct(truth, pred):
    if truth == "severe":
        return pred in (A.ALARM, A.BODY)
    if truth == "none":
        return pred == A.WORLD
    return pred == truth


def confusion3(res, preds):
    m = {(t, p): 0 for t in TRUTHS for p in PREDS}
    for r, (p, _, _) in zip(res, preds):
        if r["truth"] in TRUTHS:
            m[(r["truth"], p)] += 1
    return m


def print_confusion(name, m):
    say(f"\n  {name}: 3x3 confusion (rows = truth, columns = verdict)")
    say(f"  {'':<16}" + "".join(f"{'-> ' + p:>12}" for p in PREDS))
    for t in TRUTHS:
        say(f"  {'true ' + t:<16}" + "".join(f"{m[(t, p)]:>12}" for p in PREDS))


def frac(k, n):
    lo, hi = metrics.wilson(k, n)
    return f"{k}/{n} = {k / n:.1%} [95% CI {lo:.1%}, {hi:.1%}]" if n else "n/a"


# ---------------------------------------------------------------------------
# dev: choose the alarm threshold
# ---------------------------------------------------------------------------

def stage_dev():
    say("[dev] choose the alarm threshold")
    res = record("dev")
    body_thr = body_threshold_new(res)
    say(f"  body/world threshold (dev undisturbed): {body_thr:.4g}")
    grid = np.geomspace(ac.ALARM_GRID_LO, ac.ALARM_GRID_HI, ac.ALARM_GRID_N)
    rows = []
    for thr in grid:
        preds = run_new(res, body_thr, thr)
        fa = sum(p == A.ALARM for r, (p, _, _) in zip(res, preds) if r["truth"] in ("world", "none"))
        sev = [(r, p) for r, (p, _, _) in zip(res, preds) if r["truth"] == "severe"]
        rows.append(dict(threshold=float(thr), false_alarms=fa,
                         severe_alarm=sum(p == A.ALARM for _, p in sev),
                         severe_body=sum(p == A.BODY for _, p in sev),
                         severe_missed=sum(p == A.WORLD for _, p in sev), n_severe=len(sev)))
    with open(os.path.join(OUT, "dev_alarm_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    feasible = [r for r in rows if r["false_alarms"] == 0]
    if not feasible:
        say("  no threshold gives zero false alarms on dev; using the highest grid value")
        chosen = float(grid[-1])
    else:
        best = min(r["severe_missed"] for r in feasible)
        tied = [r["threshold"] for r in feasible if r["severe_missed"] == best]
        mid = math.sqrt(min(tied) * max(tied))
        ok = lambda t: (sum(p == A.ALARM for r, (p, _, _) in zip(res, run_new(res, body_thr, t))
                            if r["truth"] in ("world", "none")) == 0 and
                        sum(p == A.WORLD for r, (p, _, _) in zip(res, run_new(res, body_thr, t))
                            if r["truth"] == "severe") == best)
        chosen = mid if ok(mid) else min(tied, key=lambda t: abs(math.log(t / mid)))
        say(f"  thresholds with zero false alarms: {min(r['threshold'] for r in feasible):.4g}"
            f" .. {max(r['threshold'] for r in feasible):.4g}")
        say(f"  fewest severe misses among them: {best}/{rows[0]['n_severe']}, "
            f"tied over {min(tied):.4g} .. {max(tied):.4g}")
    say(f"  chosen alarm threshold: {chosen:.4g}")
    json.dump({"alarm_threshold": chosen}, open(os.path.join(OUT, "chosen_alarm.json"), "w"))
    plot_alarm.dev_sweep(rows, chosen, os.path.join(OUT, "dev_alarm_sweep.png"))
    open(os.path.join(OUT, "dev_summary.txt"), "w").write("\n".join(LOG) + "\n")


# ---------------------------------------------------------------------------
# test: evaluate once, paired against the previous rule
# ---------------------------------------------------------------------------

def stage_test():
    path = os.path.join(OUT, "chosen_alarm.json")
    if ac.ALARM_THRESHOLD is not None:
        alarm_thr = float(ac.ALARM_THRESHOLD)
    elif os.path.exists(path):
        alarm_thr = json.load(open(path))["alarm_threshold"]
    else:
        sys.exit("run `python compare_alarm.py dev` first: the alarm threshold is chosen on dev")
    say(f"[test] fresh episodes; alarm threshold {alarm_thr:.4g} (frozen from dev)")
    res = record("test")
    thr_new, thr_prev = body_threshold_new(res), body_threshold_prev(res)
    say(f"  body/world thresholds from test undisturbed: new {thr_new:.4g}, previous {thr_prev:.4g}")
    new, prev = run_new(res, thr_new, alarm_thr), run_prev(res, thr_prev)

    m_prev, m_new = confusion3(res, prev), confusion3(res, new)
    print_confusion("PREVIOUS rule (settle, 5 s, undecided -> world)", m_prev)
    print_confusion("NEW rule (settle up to 10 s, then persistence -> ALARM)", m_new)

    def by(cat, preds):
        return [p for r, (p, _, _) in zip(res, preds) if r["category"] == cat]

    say("\n  (a) severe damage (x1.5-1.7): missed = called world")
    for name, preds in (("previous", prev), ("new", new)):
        s = by("severe", preds)
        say(f"    {name:<9} alarm {frac(s.count(A.ALARM), len(s))}; body {s.count(A.BODY)}; "
            f"MISSED {frac(s.count(A.WORLD), len(s))}")
    near = [i for i, r in enumerate(res) if r["category"] == "severe" and 1.55 <= r["factor"] <= 1.65]
    for name, preds in (("previous", prev), ("new", new)):
        s = [preds[i][0] for i in near]
        say(f"    x1.55-1.65 subset, {name:<9}: alarm {s.count(A.ALARM)}, body {s.count(A.BODY)}, "
            f"missed {s.count(A.WORLD)} (of {len(s)})")

    say("\n  (b) big long pushes (2-4 N*m, 0.5-1 s): any alarm is a new false alarm")
    for name, preds in (("previous", prev), ("new", new)):
        s = by("bigpush", preds)
        say(f"    {name:<9} ALARM {frac(s.count(A.ALARM), len(s))}; body {s.count(A.BODY)}; "
            f"world {s.count(A.WORLD)}")

    say("\n  (c) false positives: any world event (ordinary + big) called body or alarm")
    for name, preds in (("previous", prev), ("new", new)):
        w = by("world", preds) + by("bigpush", preds)
        k = sum(p != A.WORLD for p in w)
        say(f"    {name:<9} {frac(k, len(w))}  (body {w.count(A.BODY)}, alarm {w.count(A.ALARM)})")
    for name, preds in (("previous", prev), ("new", new)):
        c = by("calib", preds)
        say(f"    undisturbed, {name:<9}: body {c.count(A.BODY)}, alarm {c.count(A.ALARM)} (of {len(c)})"
            + ("  [body/world threshold set on these]" if True else ""))

    say("\n  moderate body changes (x1.2-1.5) detected:")
    for name, preds in (("previous", prev), ("new", new)):
        s = by("body", preds)
        say(f"    {name:<9} {frac(s.count(A.BODY) + s.count(A.ALARM), len(s))} "
            f"(body {s.count(A.BODY)}, alarm {s.count(A.ALARM)})")

    say("\n  how the new rule reached its verdicts (path) and how long it took:")
    for cat in ("body", "world", "severe", "bigpush", "calib"):
        idx = [i for i, r in enumerate(res) if r["category"] == cat]
        paths = [new[i][2] for i in idx]
        delays = [(new[i][1] - res[i]["onset"]) * DT for i in idx]
        say(f"    {cat:<8} settled {paths.count('settled'):>3}, persistent {paths.count('persistent'):>3}, "
            f"decayed {paths.count('decayed'):>3} | delay median {np.median(delays):.2f} s, "
            f"max {np.max(delays):.2f} s")

    dist = [i for i, r in enumerate(res) if r["truth"] in TRUTHS]
    b = sum(is_correct(res[i]["truth"], new[i][0]) and not is_correct(res[i]["truth"], prev[i][0]) for i in dist)
    c = sum(is_correct(res[i]["truth"], prev[i][0]) and not is_correct(res[i]["truth"], new[i][0]) for i in dist)
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
    say(f"\n  paired over {len(dist)} disturbed test trials (severe counted correct if alarm or body):"
        f" new right & previous wrong {b}, previous right & new wrong {c}, exact McNemar p = {p:.3g}")
    for i in dist:
        if is_correct(res[i]["truth"], prev[i][0]) and not is_correct(res[i]["truth"], new[i][0]):
            r = res[i]
            say(f"    new rule worse: {r['category']} {r['param']} x{r['factor']:.2f} "
                f"{r['torque']:+.2f} N*m {r['duration']} steps -> {new[i][0]} via {new[i][2]}")

    # per-trial CSV
    rows = []
    for r, pn, pp in zip(res, new, prev):
        bl = A.late_blocks(r["err"], r["onset"])
        rows.append({k: r[k] for k in ("category", "truth", "param", "factor", "torque",
                                       "duration", "onset", "seed")}
                    | {"new_verdict": pn[0], "new_path": pn[2], "new_delay_s": (pn[1] - r["onset"]) * DT,
                       "prev_verdict": pp[0], "prev_path": pp[2],
                       "late_block_min": float(bl.min()), "late_block_median": float(np.median(bl)),
                       "late_frac_above_alarm": float(np.mean(bl > alarm_thr))})
    with open(os.path.join(OUT, "test_trials.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    figs = [plot_alarm.confusions(m_prev, m_new, os.path.join(OUT, "confusion_3x3.png")),
            plot_alarm.category_outcomes(res, prev, new, os.path.join(OUT, "outcomes_by_category.png")),
            plot_alarm.error_over_time(res, alarm_thr, os.path.join(OUT, "error_over_time.png"))]
    say("\n  figures: " + ", ".join(os.path.relpath(f, HERE) for f in figs))
    open(os.path.join(OUT, "test_summary.txt"), "w").write("\n".join(LOG) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["dev", "test"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    assert env_setup.verify_physics() == 0.0, "push-capable physics differs from Pendulum-v1"
    model_lib.get_model(os.path.join(HERE, run_experiment.config.MODEL_PATH))
    stage_dev() if args.stage == "dev" else stage_test()


if __name__ == "__main__":
    main()
