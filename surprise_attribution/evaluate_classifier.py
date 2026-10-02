"""Quantified body-vs-world classifier: calibrate, evaluate, sweep, find boundaries.

    python evaluate_classifier.py            # full run (defaults in classify_config.py)
    python evaluate_classifier.py --quick    # tiny smoke test, separate output folder

Pipeline:
  0. check the push-capable physics equals stock Pendulum-v1, load the frozen model
  1. CALIBRATE  run undisturbed episodes, set threshold = mean + k*std of their
                late-window scores (skipped if THRESHOLD is set by hand)
  2. EVALUATE   N_PER_CLASS randomised body-change and world-event trials,
                classify each, print confusion matrix / accuracy / FPR / FNR
  3. ROC        sweep the threshold over the main-evaluation scores
  4. BOUNDARY   fixed-magnitude grids: smallest body change still detected,
                largest push still rejected
  5. SAVE       every trial to CSV, summary tables to CSV, figures to PNG

Every number printed or saved is computed from the episodes run here.
"""
import argparse
import csv
import os
import sys

import numpy as np

import classifier
import classify_config as cc
import env_setup
import metrics
import model as model_lib
import plot_classifier as pc
import run_experiment
import trials

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = []   # everything printed is also written to summary.txt


def say(msg=""):
    print(msg, flush=True)
    LOG.append(msg)


def write_csv(rows, path, fields=None):
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def fmt_rate(name, value, ci, note=""):
    return f"  {name:<22} {value:6.1%}   95% CI [{ci[0]:5.1%}, {ci[1]:5.1%}]  {note}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true",
                    help="tiny run to check the pipeline works (not for reporting)")
    args = ap.parse_args()
    out_dir = os.path.join(HERE, cc.OUT_DIR)
    if args.quick:
        cc.N_CALIB, cc.N_PER_CLASS, cc.N_BOUNDARY = 8, 8, 2
        out_dir += "_quick"
    os.makedirs(out_dir, exist_ok=True)

    # -- 0. sanity checks ---------------------------------------------------
    assert env_setup.verify_physics() == 0.0, "push-capable physics differs from Pendulum-v1"
    model_path = os.path.join(HERE, run_experiment.config.MODEL_PATH)
    model_lib.get_model(model_path)   # trains + caches if missing; workers load it
    say(f"late window: {cc.LATE_START}-{cc.LATE_END} steps "
        f"({cc.LATE_START * 0.05:g}-{cc.LATE_END * 0.05:g} s) after onset")

    # -- 1. calibrate the threshold on undisturbed episodes only ------------
    say("\n[1] calibration (undisturbed episodes only)")
    calib = trials.run_all(trials.calibration_specs(), "undisturbed")
    calib_scores = [r["score"] for r in calib]
    if cc.THRESHOLD is None:
        threshold = classifier.calibrate_threshold(calib_scores)
        say(f"  undisturbed late-window score: mean {np.mean(calib_scores):.4g}, "
            f"std {np.std(calib_scores, ddof=1):.4g}, max {np.max(calib_scores):.4g}")
        say(f"  threshold = mean + {cc.CALIB_SIGMAS:g} x std = {threshold:.4g}")
    else:
        threshold = float(cc.THRESHOLD)
        say(f"  threshold set by hand in classify_config.py: {threshold:.4g}")
    calib_alarms = sum(s > threshold for s in calib_scores)
    say(f"  undisturbed episodes above threshold: {calib_alarms}/{len(calib_scores)}"
        + ("  (in-sample: these set the threshold)" if cc.THRESHOLD is None else ""))
    for r in calib:
        r["prediction"] = classifier.classify(r["score"], threshold)
    write_csv(calib, os.path.join(out_dir, "calibration_trials.csv"))

    # -- 2. main evaluation -------------------------------------------------
    say(f"\n[2] main evaluation ({cc.N_PER_CLASS} body-change + {cc.N_PER_CLASS} world-event trials)")
    say(f"  body change: param in {cc.BODY_PARAMS}, factor U[{cc.BODY_FACTOR_MIN}, {cc.BODY_FACTOR_MAX}]")
    say(f"  world event: |torque| U[{cc.WORLD_TORQUE_MIN}, {cc.WORLD_TORQUE_MAX}] N*m, random sign, "
        f"duration {cc.WORLD_STEPS_MIN}-{cc.WORLD_STEPS_MAX} steps")
    say(f"  onset: U[{cc.ONSET_MIN}, {cc.ONSET_MAX}] steps")
    main_res = trials.run_all(trials.main_specs(), "main")
    for r in main_res:
        r["prediction"] = classifier.classify(r["score"], threshold)
        r["correct"] = r["prediction"] == r["true_class"]
    write_csv(main_res, os.path.join(out_dir, "main_trials.csv"))

    true = [r["true_class"] for r in main_res]; pred = [r["prediction"] for r in main_res]
    cm = metrics.confusion(true, pred)
    rt = metrics.rates(cm)
    say("\n  confusion matrix (rows = truth, columns = prediction)")
    say(f"  {'':<20}{'pred: body change':>20}{'pred: world event':>20}")
    say(f"  {'true: body change':<20}{cm['body->body']:>20}{cm['body->world']:>20}")
    say(f"  {'true: world event':<20}{cm['world->body']:>20}{cm['world->world']:>20}")
    say("")
    say(fmt_rate("accuracy", *rt["accuracy"]))
    say(fmt_rate("FALSE-POSITIVE RATE", *rt["false_positive_rate"],
                 "<- SAFETY-CRITICAL: world event called body change"))
    say(fmt_rate("false-negative rate", *rt["false_negative_rate"],
                 "(body change called world event)"))
    fell = [r for r in main_res if r["fell"]]
    say(f"\n  trials where the pendulum had fallen (|theta| > {cc.FELL_ANGLE} rad) by the end "
        f"of the late window: {len(fell)}/{len(main_res)} "
        f"(body {sum(r['true_class'] == classifier.BODY for r in fell)}, "
        f"world {sum(r['true_class'] == classifier.WORLD for r in fell)})")
    errs = [r for r in main_res if not r["correct"]]
    if errs:
        say("  misclassified trials:")
        for r in errs:
            what = (f"body {r['param']} x{r['factor']:.2f}" if r["true_class"] == classifier.BODY
                    else f"world {r['torque']:+.2f} N*m for {r['duration']} steps")
            say(f"    seed {r['seed']}: {what}, score {r['score']:.4g}, fell={r['fell']}")
    for p in cc.BODY_PARAMS:
        sub = [r for r in main_res if r["true_class"] == classifier.BODY and r["param"] == p]
        if sub:
            say(f"  body trials changing '{p}': {sum(r['correct'] for r in sub)}/{len(sub)} detected")

    # -- 3. threshold sweep / ROC ------------------------------------------
    say("\n[3] threshold sweep (ROC) on the main-evaluation scores")
    scores = [r["score"] for r in main_res]
    roc_rows = metrics.roc(scores, true)
    auc_value = metrics.auc(roc_rows)
    write_csv([{"threshold": th, "false_positive_rate": f, "true_positive_rate": t}
               for th, f, t in roc_rows], os.path.join(out_dir, "roc.csv"))
    say(f"  AUC = {auc_value:.4f}")
    zero_fp = [r for r in roc_rows if r[1] == 0.0]
    best = max(zero_fp, key=lambda r: r[2])
    say(f"  best TPR with zero false positives in this sample: TPR {best[2]:.1%} "
        f"at threshold > {best[0]:.4g}")
    say("  (picking a threshold from this curve and re-scoring the same trials is in-sample;"
        " confirm a chosen value on fresh seeds before reporting it)")
    op_fpr, op_tpr = rt["false_positive_rate"][0], rt["true_positive_rate"][0]

    # -- 4. boundary analysis ----------------------------------------------
    say(f"\n[4] boundary analysis ({cc.N_BOUNDARY} trials per setting, same threshold)")
    bnd = trials.run_all(trials.boundary_specs(), "boundary")
    for r in bnd:
        r["prediction"] = classifier.classify(r["score"], threshold)
        r["correct"] = r["prediction"] == r["true_class"]
    write_csv(bnd, os.path.join(out_dir, "boundary_trials.csv"))
    body_rows = [r for r in bnd if r["block"] == "boundary_body"]
    world_rows = [r for r in bnd if r["block"] == "boundary_world"]

    body_tab = metrics.boundary_table(body_rows, lambda r: (r["param"], r["factor"]), classifier.BODY)
    world_tab = metrics.boundary_table(world_rows, lambda r: (abs(r["torque"]), r["duration"]),
                                       classifier.WORLD)
    write_csv([{"param": k[0], "factor": k[1], "detected": c, "n": n, "rate": rate,
                "ci_low": ci[0], "ci_high": ci[1]} for k, c, n, rate, ci in body_tab],
              os.path.join(out_dir, "boundary_body.csv"))
    write_csv([{"torque": k[0], "duration_steps": k[1], "rejected": c, "n": n, "rate": rate,
                "ci_low": ci[0], "ci_high": ci[1]} for k, c, n, rate, ci in world_tab],
              os.path.join(out_dir, "boundary_world.csv"))

    # Smallest body change still detected: the smallest factor such that it
    # AND every larger factor tested are detected at >= RELIABLE_RATE. Requiring
    # all larger ones too stops one lucky small setting from counting.
    say(f"  'reliably' = at least {cc.RELIABLE_RATE:.0%} of trials in a setting are correct")
    for p in cc.BODY_PARAMS:
        rows = [r for r in body_tab if r[0][0] == p]
        say(f"  body '{p}': " + ", ".join(f"x{k[1]:g} {rate:.0%}" for k, _, _, rate, _ in rows))
        smallest = None
        for k, _, _, rate, _ in reversed(rows):
            if rate >= cc.RELIABLE_RATE:
                smallest = k[1]
            else:
                break
        say(f"    -> smallest {'length' if p == 'l' else 'mass'} change reliably detected: "
            + (f"x{smallest:g}" if smallest else "none in the tested range"))

    # Largest push still rejected, per duration: the largest torque such that it
    # AND every smaller torque tested are rejected at >= RELIABLE_RATE.
    for d in cc.BOUNDARY_DURATIONS:
        rows = [r for r in world_tab if r[0][1] == d]
        say(f"  push {d} steps ({d * 0.05:g} s): "
            + ", ".join(f"{k[0]:g}N*m {rate:.0%}" for k, _, _, rate, _ in rows))
        largest = None
        for k, _, _, rate, _ in rows:
            if rate >= cc.RELIABLE_RATE:
                largest = k[0]
            else:
                break
        say(f"    -> largest push reliably rejected at this duration: "
            + (f"{largest:g} N*m" if largest else "none in the tested range"))
    bfell = [r for r in bnd if r["fell"]]
    say(f"  boundary trials where the pendulum had fallen by the end of the window: "
        f"{len(bfell)}/{len(bnd)}")

    # -- 5. figures + summary ----------------------------------------------
    say("\n[5] figures")
    for p in (pc.plot_scores(calib, main_res, threshold, os.path.join(out_dir, "scores.png")),
              pc.plot_confusion(cm, os.path.join(out_dir, "confusion.png")),
              pc.plot_roc(roc_rows, auc_value, op_fpr, op_tpr, os.path.join(out_dir, "roc.png")),
              pc.plot_boundary_body(body_tab, os.path.join(out_dir, "boundary_body.png")),
              pc.plot_boundary_world(world_tab, os.path.join(out_dir, "boundary_world.png"))):
        say(f"  -> {os.path.relpath(p, HERE)}")
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")
    say(f"  -> {os.path.relpath(os.path.join(out_dir, 'summary.txt'), HERE)}")


if __name__ == "__main__":
    sys.exit(main())
