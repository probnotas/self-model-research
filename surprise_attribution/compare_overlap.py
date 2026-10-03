"""Simultaneous disturbances: does the existing classifier survive overlap?

    python compare_overlap.py

The classifier is NOT changed or tuned here. It is alarm_classifier.decide
(settle up to 10 s, else persistence -> ALARM), with
  - the alarm threshold frozen from its own dev stage (results_alarm/chosen_alarm.json),
  - the body/world threshold set exactly as before: mean + 4 std of the
    settled score on this run's undisturbed episodes.
Because nothing is tuned, there is no dev stage: this single run on fresh
seeds is the evaluation. The previous settle rule (no alarm, undecided ->
world) is scored on the identical episodes for reference.

The two failure modes, measured separately:
  MIMICKING (false positive)  push only -> "body" or "alarm".
      A push on its own gets mistaken for damage. Safety-critical: the robot
      would rewrite a correct self-model.
  MASKING (false negative)    body + push -> "world", although the SAME body
      change without the push (its paired twin) is detected. The push hides
      real damage. Measured pair by pair, so it isolates the push's effect.

"Detected" for a body change means verdict body OR alarm: both acknowledge
damage. "world" on a body change is a miss.
"""
import csv
import json
import math
import os

import numpy as np

import alarm_classifier as A
import alarm_config as ac
import classifier
import classify_config as cc
import env_setup
import metrics
import model as model_lib
import overlap_config as oc
import overlap_trials as ot
import plot_overlap
import run_experiment

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, oc.RESULTS)
CATS = ("body_only", "push_only", "body_push", "undisturbed")
CAT_NAME = {"body_only": "(1) body only", "push_only": "(2) push only",
            "body_push": "(3) body + push", "undisturbed": "(4) undisturbed"}
LOG = []


def say(msg=""):
    print(msg, flush=True)
    LOG.append(msg)


def frac(k, n):
    lo, hi = metrics.wilson(k, n)
    return f"{k}/{n} = {k / n:.1%} [95% CI {lo:.1%}, {hi:.1%}]" if n else "0/0"


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


# ---------------------------------------------------------------------------
# traces
# ---------------------------------------------------------------------------

def record():
    path = os.path.join(OUT, "traces.npz")
    if os.path.exists(path):
        say("  loading cached traces")
        d = np.load(path)
        specs = json.loads(str(d["specs"]))
        for i, s in enumerate(specs):
            s["err"], s["theta"], s["thdot"] = d["err"][i], d["theta"][i], d["thdot"][i]
        return specs
    res = ot.run(ot.make_specs())
    keys = [k for k in res[0] if k not in ("err", "theta", "thdot")]
    np.savez_compressed(path, err=np.stack([r["err"] for r in res]),
                        theta=np.stack([r["theta"] for r in res]),
                        thdot=np.stack([r["thdot"] for r in res]),
                        specs=np.array(json.dumps([{k: r[k] for k in keys} for r in res])))
    return res


# ---------------------------------------------------------------------------
# the existing rules, applied unchanged
# ---------------------------------------------------------------------------

def verdicts(res):
    alarm_thr = json.load(open(os.path.join(HERE, ac.RESULTS, "chosen_alarm.json")))["alarm_threshold"]
    und = [r for r in res if r["category"] == "undisturbed"]
    s_new = [A.settled_score(r["err"], r["theta"], r["thdot"], r["onset"])[0] for r in und]
    thr_new = classifier.calibrate_threshold([s for s in s_new if s is not None])
    s_prev = [classifier.calm_gated_score(r["err"], r["theta"], r["thdot"], r["onset"],
                                          calm_steps=2 * cc.CALM_STEPS)[0] for r in und]
    thr_prev = classifier.calibrate_threshold([s for s in s_prev if np.isfinite(s)])
    say(f"  alarm threshold {alarm_thr:.4g} (frozen); body/world threshold from undisturbed: "
        f"current rule {thr_new:.4g}, previous rule {thr_prev:.4g}")
    for r in res:
        v, t, path = A.decide(r["err"], r["theta"], r["thdot"], r["onset"], thr_new, alarm_thr)
        r["verdict"], r["path"], r["delay_s"] = v, path, (t - r["onset"]) * 0.05
        r["prev_verdict"] = A.previous_rule(r["err"], r["theta"], r["thdot"], r["onset"], thr_prev)[0]
        r["detected"] = r["verdict"] in (A.BODY, A.ALARM)
        r["prev_detected"] = r["prev_verdict"] in (A.BODY, A.ALARM)


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def bin_index(x, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= x < edges[i + 1]:
            return i
    return None


def pairs(res):
    """triplet id -> {category: trial} for the three paired members."""
    out = {}
    for r in res:
        if r["triplet"] >= 0:
            out.setdefault(r["triplet"], {})[r["category"]] = r
    return out


def analyse(res, key="verdict"):
    det = "detected" if key == "verdict" else "prev_detected"
    by_cat = {c: [r for r in res if r["category"] == c] for c in CATS}

    say(f"\n  confusion per category (verdict counts: body / world / alarm)")
    for c in CATS:
        v = [r[key] for r in by_cat[c]]
        say(f"    {CAT_NAME[c]:<18} body {v.count(A.BODY):>4}   world {v.count(A.WORLD):>4}   "
            f"alarm {v.count(A.ALARM):>4}   (n={len(v)})")

    b1 = by_cat["body_only"]; b3 = by_cat["body_push"]; p2 = by_cat["push_only"]
    say("\n  body-change detection (body or alarm):")
    say(f"    (1) without push  {frac(sum(r[det] for r in b1), len(b1))}")
    say(f"    (3) with push     {frac(sum(r[det] for r in b3), len(b3))}")
    pr = pairs(res)
    masked = [t for t, m in pr.items() if m["body_only"][det] and not m["body_push"][det]]
    unmasked = [t for t, m in pr.items() if not m["body_only"][det] and m["body_push"][det]]
    say(f"    paired: MASKED (detected alone, missed with push) {len(masked)}; "
        f"rescued (missed alone, detected with push) {len(unmasked)}; "
        f"exact McNemar p = {mcnemar(len(masked), len(unmasked)):.3g}")
    fp = sum(r[key] in (A.BODY, A.ALARM) for r in p2)
    say(f"\n  MIMICKING: push only called body or alarm  {frac(fp, len(p2))}  <- safety-critical")
    say(f"    (body {sum(r[key] == A.BODY for r in p2)}, alarm {sum(r[key] == A.ALARM for r in p2)})")
    und = by_cat["undisturbed"]
    say(f"    undisturbed called body or alarm: {sum(r[key] != A.WORLD for r in und)}/{len(und)}"
        f" (body/world threshold set on these)")
    return masked, unmasked


def breakdowns(res):
    """Per offset bin / push magnitude bin, for the current rule."""
    pr = pairs(res)
    rows = []

    def table(label, keyfn, names):
        say(f"\n  by {label}:")
        say(f"    {'bin':<28}{'n':>4}{'det (1)':>10}{'det (3)':>10}{'masked':>8}{'FP (2)':>9}")
        for i, name in enumerate(names):
            ms = [m for m in pr.values() if keyfn(m["body_push"]) == i]
            if not ms:
                continue
            n = len(ms)
            d1 = sum(m["body_only"]["detected"] for m in ms)
            d3 = sum(m["body_push"]["detected"] for m in ms)
            mk = sum(m["body_only"]["detected"] and not m["body_push"]["detected"] for m in ms)
            fp = sum(m["push_only"]["verdict"] != A.WORLD for m in ms)
            say(f"    {name:<28}{n:>4}{d1 / n:>10.0%}{d3 / n:>10.0%}{mk:>8}{fp:>9}")
            rows.append(dict(breakdown=label, bin=name, n=n, detected_body_only=d1,
                             detected_body_push=d3, masked=mk, push_only_false_positives=fp))

    table("overlap offset (push onset - body onset)",
          lambda r: bin_index(r["offset_s"], oc.OFFSET_BINS), oc.OFFSET_BIN_NAMES)
    tq_names = [f"|torque| {a:g}-{b:g} N*m" for a, b in zip(oc.TORQUE_BINS[:-1], oc.TORQUE_BINS[1:])]
    table("push magnitude", lambda r: bin_index(abs(r["torque"]), oc.TORQUE_BINS),
          [n.replace("4.0001", "4") for n in tq_names])
    du_names = [f"duration {a * 0.05:g}-{(b - 1) * 0.05:g} s" for a, b in
                zip(oc.DURATION_BINS[:-1], oc.DURATION_BINS[1:])]
    table("push duration", lambda r: bin_index(r["duration"], oc.DURATION_BINS), du_names)
    fb = (1.2, 1.3, 1.4, 1.5, 1.6001)
    table("body-change size", lambda r: bin_index(r["factor"], fb),
          ["x1.2-1.3", "x1.3-1.4", "x1.4-1.5", "x1.5-1.6"])
    with open(os.path.join(OUT, "breakdowns.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # worst-case timing on a finer grid
    edges = np.arange(oc.OFFSET_MIN * 0.05, oc.OFFSET_MAX * 0.05 + oc.OFFSET_FINE_STEP, oc.OFFSET_FINE_STEP)
    fine = []
    for a, b in zip(edges[:-1], edges[1:]):
        ms = [m for m in pr.values() if a <= m["body_push"]["offset_s"] < b]
        if ms:
            fine.append(dict(lo=float(a), hi=float(b), n=len(ms),
                             det1=float(np.mean([m["body_only"]["detected"] for m in ms])),
                             det3=float(np.mean([m["body_push"]["detected"] for m in ms])),
                             masked=sum(m["body_only"]["detected"] and not m["body_push"]["detected"] for m in ms),
                             fp=sum(m["push_only"]["verdict"] != A.WORLD for m in ms)))
    worst = max(fine, key=lambda f: (f["masked"] + f["fp"], -f["det3"]))
    say(f"\n  worst-case timing (0.25 s bins): offset {worst['lo']:+.2f} to {worst['hi']:+.2f} s: "
        f"masked {worst['masked']}, push-only false positives {worst['fp']}, "
        f"detection with push {worst['det3']:.0%} (n={worst['n']})"
        + ("  [no failures in any bin]" if worst["masked"] + worst["fp"] == 0 else ""))
    return fine


def main():
    os.makedirs(OUT, exist_ok=True)
    assert env_setup.verify_physics() == 0.0, "push-capable physics differs from Pendulum-v1"
    model = model_lib.get_model(os.path.join(HERE, run_experiment.config.MODEL_PATH))
    gap = ot.verify_matches_existing(model)
    assert gap == 0.0, f"overlap episode loop differs from run_experiment.run_episode by {gap}"
    say("[overlap] combined loop == existing episode loop for single disturbances (max diff 0)")
    say(f"  {oc.N_TRIPLETS} paired triplets + {oc.N_UNDISTURBED} undisturbed, fresh seeds; "
        f"body x{oc.BODY_FACTOR_MIN}-{oc.BODY_FACTOR_MAX}, push {oc.PUSH_TORQUE_MIN}-{oc.PUSH_TORQUE_MAX} N*m "
        f"for {oc.PUSH_STEPS_MIN * 0.05:g}-{oc.PUSH_STEPS_MAX * 0.05:g} s, "
        f"offset {oc.OFFSET_MIN * 0.05:+g} to {oc.OFFSET_MAX * 0.05:+g} s")
    res = record()
    verdicts(res)

    say("\n== CURRENT rule (3-way: settle <=10 s, else persistence -> ALARM) ==")
    masked, _ = analyse(res, "verdict")
    fine = breakdowns(res)
    paths = {c: {p: sum(r["path"] == p for r in res if r["category"] == c)
                 for p in ("settled", "persistent", "decayed")} for c in CATS}
    say("\n  verdict path per category: " + "; ".join(f"{c} {paths[c]}" for c in CATS))
    for c in CATS:
        d = [r["delay_s"] for r in res if r["category"] == c]
        say(f"    delay {c:<12} median {np.median(d):.2f} s, max {np.max(d):.2f} s")
    if masked:
        say("  masked triplets:")
        pr = pairs(res)
        for t in masked:
            m = pr[t]["body_push"]
            say(f"    {m['param']} x{m['factor']:.2f}, push {m['torque']:+.2f} N*m for {m['duration'] * 0.05:.2f} s, "
                f"offset {m['offset_s']:+.2f} s -> {m['verdict']} via {m['path']}")

    say("\n== PREVIOUS rule (settle <=5 s, undecided -> world), same episodes ==")
    analyse(res, "prev_verdict")

    with open(os.path.join(OUT, "trials.csv"), "w", newline="") as f:
        keys = ["category", "triplet", "seed", "param", "factor", "torque", "duration",
                "body_onset", "push_onset", "offset_s", "onset", "verdict", "path", "delay_s",
                "prev_verdict"]
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        w.writerows([{k: r[k] for k in keys} for r in res])
    figs = [plot_overlap.category_verdicts(res, os.path.join(OUT, "verdicts_by_category.png")),
            plot_overlap.offset_curve(fine, os.path.join(OUT, "detection_vs_offset.png")),
            plot_overlap.magnitude_grid(res, os.path.join(OUT, "masking_by_push.png")),
            plot_overlap.example_triplet(res, os.path.join(OUT, "example_triplet.png"))]
    say("\n  figures: " + ", ".join(os.path.relpath(f, HERE) for f in figs))
    open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
