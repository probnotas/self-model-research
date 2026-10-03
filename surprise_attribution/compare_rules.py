"""Fix the classifier's weak spots, without fooling ourselves.

    python compare_rules.py dev     # record dev episodes, score every candidate, pick one
    python compare_rules.py test    # record FRESH test episodes, evaluate the pick vs the original

Why two stages:
  If we tried several rules and reported whichever scored best on the same
  trials, the winner's score would be inflated by the choosing itself. So:
    dev   candidates are compared on a development set, and one is chosen
          by a criterion written down below BEFORE looking at results;
    test  the chosen rule and the original rule are scored once on new
          seeds that played no part in the choice. Only test numbers are
          the result.

Why traces:
  Each episode is simulated once and its full error / angle / speed series is
  saved. Every rule is then scored on the IDENTICAL episodes, so a difference
  between rules is the rule, not which episodes each happened to see
  (a paired comparison).

Candidates (all use: threshold = mean + 4 std of their own undisturbed scores):
  fixed            the original rule: mean error 1-2 s after onset
  fixed_ratio      the same, divided by the episode's own pre-onset error
  calm             mean error over the first 1 s of SETTLED transitions after
                   1 s; "undecided" (acted on as world) if none by 5 s
  calm_ratio       calm, divided by the pre-onset error
  calm_long        calm, but 2 s of settled transitions (more averaging, slower)
  calm_long_ratio  calm_long, divided by the pre-onset error

  An earlier version of the calm rule (no settle requirement, and a fall-back
  to averaging every step when the pendulum never calmed) was tried on dev and
  made false positives worse; see the README. It was revised on dev only.

Selection criterion (fixed in advance):
  1. zero false positives on the dev main set (the safety-critical error);
  2. among those, the highest balanced boundary coverage
     = (mean detection rate over body cells + mean rejection rate over push cells) / 2;
  3. ties within 0.01 go to the shorter median decision delay.
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

import classifier
import classify_config as cc
import env_setup
import metrics
import model as model_lib
import plot_classifier as pc
import plot_rules as pr
import run_experiment
import trials

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, cc.RESULTS_RULES)
LOG = []
DT = 0.05


def say(msg=""):
    print(msg, flush=True)
    LOG.append(msg)


# ---------------------------------------------------------------------------
# Candidate rules. Each maps one trace to (score, decision_step, used_fallback).
# ---------------------------------------------------------------------------

def _fixed(r):
    return (classifier.late_window_score(r["err"], r["onset"]),
            r["onset"] + cc.LATE_END, False)


def _calm(r, n=None):
    return classifier.calm_gated_score(r["err"], r["theta"], r["thdot"], r["onset"], n)


def _ratio(rule):
    def scored(r):
        s, t, fb = rule(r)
        return s / classifier.pre_onset_level(r["err"], r["onset"]), t, fb
    return scored


RULES = {
    "fixed": _fixed,
    "fixed_ratio": _ratio(_fixed),
    "calm": _calm,
    "calm_ratio": _ratio(_calm),
    "calm_long": lambda r: _calm(r, 2 * cc.CALM_STEPS),
    "calm_long_ratio": _ratio(lambda r: _calm(r, 2 * cc.CALM_STEPS)),
}


# ---------------------------------------------------------------------------
# Recording and caching traces
# ---------------------------------------------------------------------------

def record(stage):
    """Simulate (or load cached) calib / main / boundary traces for a stage."""
    path = os.path.join(OUT, f"{stage}_traces.npz")
    if os.path.exists(path):
        say(f"  loading cached {stage} traces from {os.path.relpath(path, HERE)}")
        return load(path)
    base = cc.SEED_DEV if stage == "dev" else cc.SEED_TEST
    if stage == "dev":
        cc.N_CALIB, cc.N_PER_CLASS, cc.N_BOUNDARY = cc.N_DEV_CALIB, cc.N_DEV_PER_CLASS, cc.N_DEV_BOUNDARY
    specs = (trials.calibration_specs(base) + trials.main_specs(base + 1000)
             + trials.boundary_specs(base + 2000))
    res = trials.run_all(specs, f"{stage}", fn=trials.run_trace_trial)
    save(res, path)
    return res


def save(res, path):
    keys = [k for k in res[0] if k not in ("err", "theta", "thdot")]
    np.savez_compressed(path,
                        err=np.stack([r["err"] for r in res]),
                        theta=np.stack([r["theta"] for r in res]),
                        thdot=np.stack([r["thdot"] for r in res]),
                        specs=np.array(json.dumps([{k: r[k] for k in keys} for r in res])))


def load(path):
    d = np.load(path)
    specs = json.loads(str(d["specs"]))
    for i, s in enumerate(specs):
        s["err"], s["theta"], s["thdot"] = d["err"][i], d["theta"][i], d["thdot"][i]
    return specs


# ---------------------------------------------------------------------------
# Scoring one rule on one stage
# ---------------------------------------------------------------------------

def evaluate(rule_name, res):
    """Threshold from undisturbed trials, then verdicts for every trial."""
    rule = RULES[rule_name]
    scored = []
    for r in res:
        s, t, fb = rule(r)
        scored.append(dict(r, score=s, decision_step=t, fallback=fb,
                           delay_s=(t - r["onset"]) * DT))
    calib = [r for r in scored if r["block"] == "calib"]
    finite = [r["score"] for r in calib if np.isfinite(r["score"])]
    threshold = classifier.calibrate_threshold(finite)
    for r in scored:
        r["prediction"] = classifier.classify(r["score"], threshold)
        r["correct"] = r["prediction"] == r["true_class"]
    main = [r for r in scored if r["block"] == "main"]
    cm = metrics.confusion([r["true_class"] for r in main], [r["prediction"] for r in main])
    body_tab = metrics.boundary_table([r for r in scored if r["block"] == "boundary_body"],
                                      lambda r: (r["param"], r["factor"]), classifier.BODY)
    world_tab = metrics.boundary_table([r for r in scored if r["block"] == "boundary_world"],
                                       lambda r: (abs(r["torque"]), r["duration"]), classifier.WORLD)
    body_cov = float(np.mean([row[3] for row in body_tab]))
    world_cov = float(np.mean([row[3] for row in world_tab]))
    return dict(name=rule_name, threshold=threshold, scored=scored, calib=calib, main=main, cm=cm,
                rates=metrics.rates(cm), body_tab=body_tab, world_tab=world_tab,
                body_cov=body_cov, world_cov=world_cov, balanced=(body_cov + world_cov) / 2,
                calib_alarms=sum(r["prediction"] == classifier.BODY for r in calib),
                median_delay=float(np.median([r["delay_s"] for r in main])),
                max_delay=float(np.max([r["delay_s"] for r in scored if r["block"] != "calib"])),
                fallbacks=sum(r["fallback"] for r in scored if r["block"] != "calib"),
                undecided_body=sum(r["fallback"] for r in scored if r["true_class"] == classifier.BODY),
                undecided_world=sum(r["fallback"] for r in scored if r["true_class"] == classifier.WORLD))


def smallest_detected(body_tab, param):
    rows = [r for r in body_tab if r[0][0] == param]
    best = None
    for k, _, _, rate, _ in reversed(rows):
        if rate >= cc.RELIABLE_RATE:
            best = k[1]
        else:
            break
    return best


def largest_rejected(world_tab, duration):
    best = None
    for k, _, _, rate, _ in [r for r in world_tab if r[0][1] == duration]:
        if rate >= cc.RELIABLE_RATE:
            best = k[0]
        else:
            break
    return best


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from the two discordant counts.

    b = trials the new rule gets right and the old one wrong, c = the reverse.
    Under "the rules are equally good" each discordant trial is a coin flip.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_dev():
    say("[dev] candidate rules on the development set")
    res = record("dev")
    evals = [evaluate(name, res) for name in RULES]
    say(f"\n  {'rule':<17}{'FP':>5}{'FN':>5}{'body cov':>10}{'push cov':>10}"
        f"{'balanced':>10}{'median delay':>14}{'undecided (body/world)':>24}")
    for e in evals:
        say(f"  {e['name']:<17}{e['cm']['world->body']:>5}{e['cm']['body->world']:>5}"
            f"{e['body_cov']:>10.3f}{e['world_cov']:>10.3f}{e['balanced']:>10.3f}"
            f"{e['median_delay']:>12.2f} s{e['undecided_body']:>17}/{e['undecided_world']}")
    # the pre-registered criterion
    safe = [e for e in evals if e["cm"]["world->body"] == 0]
    if not safe:
        say("\n  no candidate had zero false positives on dev; keeping the original rule")
        chosen = "fixed"
    else:
        top = max(e["balanced"] for e in safe)
        tied = [e for e in safe if e["balanced"] >= top - 0.01]
        chosen = min(tied, key=lambda e: e["median_delay"])["name"]
    say(f"\n  chosen by the criterion: {chosen}")
    with open(os.path.join(OUT, "chosen_rule.json"), "w") as f:
        json.dump({"rule": chosen}, f)
    with open(os.path.join(OUT, "dev_summary.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


def stage_test():
    path = os.path.join(OUT, "chosen_rule.json")
    if not os.path.exists(path):
        sys.exit("run `python compare_rules.py dev` first: the rule must be chosen before test")
    chosen = json.load(open(path))["rule"]
    say(f"[test] original rule vs '{chosen}' (chosen on dev), on fresh seeds")
    res = record("test")
    old, new = evaluate("fixed", res), evaluate(chosen, res)

    for e in (old, new):
        r = e["rates"]
        say(f"\n  == {e['name']} ==  threshold {e['threshold']:.4g}")
        say(f"  confusion: body->body {e['cm']['body->body']}, body->world {e['cm']['body->world']}, "
            f"world->body {e['cm']['world->body']}, world->world {e['cm']['world->world']}")
        say(f"  accuracy {r['accuracy'][0]:.1%} [{r['accuracy'][1][0]:.1%}, {r['accuracy'][1][1]:.1%}]")
        say(f"  FALSE-POSITIVE RATE (safety-critical) {r['false_positive_rate'][0]:.1%} "
            f"[{r['false_positive_rate'][1][0]:.1%}, {r['false_positive_rate'][1][1]:.1%}]")
        say(f"  false-negative rate {r['false_negative_rate'][0]:.1%} "
            f"[{r['false_negative_rate'][1][0]:.1%}, {r['false_negative_rate'][1][1]:.1%}]")
        say(f"  undisturbed above threshold (in-sample): {e['calib_alarms']}/{len(e['calib'])}")
        say(f"  decision delay after onset: median {e['median_delay']:.2f} s, max {e['max_delay']:.2f} s")
        say(f"  undecided (no settled evidence, acted on as world): body {e['undecided_body']}, "
            f"world {e['undecided_world']}")
        mb = [r for r in e["main"] if r["true_class"] == classifier.BODY]
        mw = [r for r in e["main"] if r["true_class"] == classifier.WORLD]
        say(f"  margin (decided trials): highest push score "
            f"{max(r['score'] for r in mw if np.isfinite(r['score'])):.4g}, "
            f"lowest body score {min(r['score'] for r in mb if np.isfinite(r['score'])):.4g}, "
            f"threshold {e['threshold']:.4g}")
        for p in cc.BODY_PARAMS:
            s = smallest_detected(e["body_tab"], p)
            say(f"  smallest {'length' if p == 'l' else 'mass'} change reliably detected: "
                + (f"x{s:g}" if s else "none tested"))
        for d in cc.BOUNDARY_DURATIONS:
            l = largest_rejected(e["world_tab"], d)
            say(f"  largest push reliably rejected at {d * DT:g} s: " + (f"{l:g} N*m" if l else "none tested"))
        say(f"  boundary coverage: body {e['body_cov']:.3f}, push {e['world_cov']:.3f}")

    # paired comparison on every disturbed test trial
    o = {(r["block"], r["seed"]): r["correct"] for r in old["scored"] if r["block"] != "calib"}
    n = {(r["block"], r["seed"]): r["correct"] for r in new["scored"] if r["block"] != "calib"}
    b = sum(n[k] and not o[k] for k in o); c = sum(o[k] and not n[k] for k in o)
    say(f"\n  paired, all {len(o)} disturbed test trials: new right & old wrong {b}, "
        f"old right & new wrong {c}, exact McNemar p = {mcnemar_exact(b, c):.3g}")

    # save per-trial verdicts of both rules side by side
    rows = []
    for ro, rn in zip(old["scored"], new["scored"]):
        rows.append({k: ro[k] for k in ("block", "true_class", "param", "factor", "torque",
                                        "duration", "onset", "seed")}
                    | {"old_score": ro["score"], "old_prediction": ro["prediction"],
                       "new_score": rn["score"], "new_prediction": rn["prediction"],
                       "new_delay_s": rn["delay_s"], "new_undecided": rn["fallback"]})
    with open(os.path.join(OUT, "test_trials.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for e in (old, new):
        with open(os.path.join(OUT, f"test_boundary_{e['name']}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["kind", "key1", "key2", "correct", "n", "rate", "ci_low", "ci_high"])
            for k, c_, n_, rate, ci in e["body_tab"]:
                w.writerow(["body", k[0], k[1], c_, n_, rate, ci[0], ci[1]])
            for k, c_, n_, rate, ci in e["world_tab"]:
                w.writerow(["world", k[0], k[1], c_, n_, rate, ci[0], ci[1]])

    # figures
    roc_old = metrics.roc([r["score"] for r in old["main"]], [r["true_class"] for r in old["main"]])
    roc_new = metrics.roc([r["score"] for r in new["main"]], [r["true_class"] for r in new["main"]])
    bnd_true = [r["true_class"] for r in old["scored"] if r["block"].startswith("boundary")]
    roc_old_b = metrics.roc([r["score"] for r in old["scored"] if r["block"].startswith("boundary")], bnd_true)
    roc_new_b = metrics.roc([r["score"] for r in new["scored"] if r["block"].startswith("boundary")], bnd_true)
    say(f"\n  AUC main: old {metrics.auc(roc_old):.4f}, new {metrics.auc(roc_new):.4f}")
    say(f"  AUC boundary trials (the hard ones): old {metrics.auc(roc_old_b):.4f}, "
        f"new {metrics.auc(roc_new_b):.4f}")
    figs = [
        pr.compare_boundary_world(old, new, os.path.join(OUT, "compare_boundary_world.png")),
        pr.compare_boundary_body(old, new, os.path.join(OUT, "compare_boundary_body.png")),
        pr.compare_roc(roc_old_b, roc_new_b, old["name"], new["name"],
                       os.path.join(OUT, "compare_roc_boundary.png")),
        pr.delay_hist(new, os.path.join(OUT, "decision_delay.png")),
        # undecided trials have score -inf and cannot sit on a log axis; they
        # are reported in the summary instead
        pc.plot_scores([r for r in new["calib"] if np.isfinite(r["score"])],
                       [r for r in new["main"] if np.isfinite(r["score"])],
                       new["threshold"], os.path.join(OUT, f"scores_{chosen}.png")),
    ]
    say("\n  figures: " + ", ".join(os.path.relpath(p, HERE) for p in figs))
    with open(os.path.join(OUT, "test_summary.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


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
