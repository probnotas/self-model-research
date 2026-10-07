"""2x2 factorial: body change x world push, simultaneous and unannounced.

    python run_factorial.py

Question: when a body change and a push land at the same moment, can the
existing attribution rule still tell them apart?

Two failure modes, scored separately:
  MIMICKING  world-only trial called "body" or "alarm". Safety-critical: the
             robot would rewrite a correct self-model to fit a push.
  MASKING    body+world trial called "world": the push hid a real body change.

Verdicts: the rule outputs body / world / alarm. It has no separate "none"
output: "world" means "no persistent change, don't adapt", which is the
correct verdict both for a push and for nothing happening. So for the
no-disturbance cell, "world" is correct.

Correct verdicts per true condition:
  neither     -> world          body only   -> body or alarm
  world only  -> world          body+world  -> body or alarm

Body is the rigid Pendulum-v1, not a compliant body.
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
import env_setup
import factorial_config as fc
import metrics
import model as model_lib
import overlap_trials as ot
import run_experiment
import trials

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, fc.RESULTS)
CELLS = ("neither", "body_only", "world_only", "body_world")
CORRECT = {"neither": {"world"}, "world_only": {"world"},
           "body_only": {"body", "alarm"}, "body_world": {"body", "alarm"}}
LOG = []


def say(msg=""):
    print(msg, flush=True)
    LOG.append(msg)


def frac(k, n):
    lo, hi = metrics.wilson(k, n)
    return f"{k}/{n} = {k / n:.1%} (95% CI {lo:.1%}-{hi:.1%})" if n else "0/0"


# ---------------------------------------------------------------------------
# trial generation
# ---------------------------------------------------------------------------

def make_specs():
    """Balanced 2x2 cells plus separate calibration episodes, all from MASTER_SEED."""
    rng = np.random.default_rng(fc.MASTER_SEED)
    specs, seed = [], fc.MASTER_SEED
    world_combos = [(tq, d) for tq in fc.PUSH_TORQUES for d in fc.PUSH_DURATIONS]
    for cell in CELLS + ("calib",):
        n = fc.N_CALIB if cell == "calib" else fc.N_PER_CELL
        has_body = cell in ("body_only", "body_world")
        has_push = cell in ("world_only", "body_world")
        for i in range(n):
            body_onset = int(rng.integers(fc.BODY_ONSET_RANGE[0], fc.BODY_ONSET_RANGE[1] + 1))
            offset = int(rng.integers(-fc.OVERLAP_STEPS, fc.OVERLAP_STEPS + 1)) if has_push else 0
            param = ("m" if i % 2 == 0 else "l") if has_body else ""
            factor = float(rng.uniform(*(fc.MASS_RANGE if param == "m" else fc.LENGTH_RANGE))) \
                if has_body else 1.0
            if has_push:
                tq, dname = world_combos[i % len(world_combos)]
                torque = tq * (1 if rng.random() < 0.5 else -1)
                duration = fc.PUSH_DURATIONS[dname]
            else:
                torque, dname, duration = 0.0, "", 0
            push_onset = body_onset + offset
            # the rule is told when surprise starts: the earliest real disturbance;
            # with no disturbance at all, a pseudo-onset at body_onset
            onset = min(body_onset, push_onset) if (has_body and has_push) else \
                (push_onset if has_push else body_onset)
            specs.append(dict(cell=cell, seed=seed, triplet=i, has_body=has_body, has_push=has_push,
                              param=param, factor=factor, torque=float(torque), duration_name=dname,
                              duration=duration, body_onset=body_onset, push_onset=push_onset,
                              offset_steps=offset, offset_s=offset * 0.05, onset=onset))
            seed += 1
    return specs


# ---------------------------------------------------------------------------
# applying the existing rule, logging the values it used
# ---------------------------------------------------------------------------

def score(res, body_thr, alarm_thr):
    for r in res:
        s, t = A.settled_score(r["err"], r["theta"], r["thdot"], r["onset"])
        blocks = A.late_blocks(r["err"], r["onset"])
        verdict, step, path = A.decide(r["err"], r["theta"], r["thdot"], r["onset"], body_thr, alarm_thr)
        r.update(verdict=verdict, path=path, verdict_time_s=(step - r["onset"]) * 0.05,
                 settled=s is not None, settled_score=s if s is not None else float("nan"),
                 settle_time_s=(t - r["onset"]) * 0.05 if s is not None else float("nan"),
                 late_block_median=float(np.median(blocks)),
                 late_frac_above_alarm=float(np.mean(blocks > alarm_thr)),
                 correct=verdict in CORRECT.get(r["cell"], {"world"}))


def main():
    os.makedirs(OUT, exist_ok=True)
    versions = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                    gymnasium=gymnasium.__version__, master_seed=fc.MASTER_SEED)
    json.dump(versions, open(os.path.join(OUT, "versions_and_seed.json"), "w"), indent=1)
    say("Body: rigid Gymnasium Pendulum-v1 (NOT a compliant body).")
    say(f"versions: {versions}")

    assert env_setup.verify_physics() == 0.0, "push-capable physics differs from Pendulum-v1"
    model = model_lib.get_model(os.path.join(HERE, run_experiment.config.MODEL_PATH))
    assert ot.verify_matches_existing(model) == 0.0, "episode loop differs from run_episode"

    specs = make_specs()
    res = trials.run_all(specs, "factorial", fn=ot.run_trial)

    calib = [r for r in res if r["cell"] == "calib"]
    cs = [A.settled_score(r["err"], r["theta"], r["thdot"], r["onset"])[0] for r in calib]
    body_thr = classifier.calibrate_threshold([s for s in cs if s is not None])
    alarm_thr = json.load(open(os.path.join(HERE, ac.RESULTS, "chosen_alarm.json")))["alarm_threshold"]
    say(f"body/world threshold {body_thr:.5f} (from {len(calib)} separate undisturbed episodes; "
        f"{sum(s is None for s in cs)} never settled); alarm threshold {alarm_thr:.5f} (frozen)")
    score(res, body_thr, alarm_thr)
    trial_rows = [r for r in res if r["cell"] != "calib"]

    # ---- confusion matrix
    say("\nCONFUSION MATRIX (rows = ground truth, columns = verdict)")
    say(f"  {'':<14}{'body':>8}{'world*':>8}{'alarm':>8}{'n':>6}")
    for c in CELLS:
        v = [r["verdict"] for r in trial_rows if r["cell"] == c]
        say(f"  {c:<14}{v.count('body'):>8}{v.count('world'):>8}{v.count('alarm'):>8}{len(v):>6}")
    say("  * 'world' = no persistent change / don't adapt; the rule has no separate 'none' output,")
    say("    so 'world' is the correct verdict for the 'neither' cell too.")

    # ---- rates
    wrong = [r for r in trial_rows if not r["correct"]]
    w_only = [r for r in trial_rows if r["cell"] == "world_only"]
    b_w = [r for r in trial_rows if r["cell"] == "body_world"]
    b_only = [r for r in trial_rows if r["cell"] == "body_only"]
    neither = [r for r in trial_rows if r["cell"] == "neither"]
    say("\nRATES")
    say(f"  overall misattribution          {frac(len(wrong), len(trial_rows))}")
    fp_b = sum(r["verdict"] == "body" for r in w_only); fp_a = sum(r["verdict"] == "alarm" for r in w_only)
    say(f"  FALSE POSITIVE (world only -> body or alarm)  {frac(fp_b + fp_a, len(w_only))}"
        f"   [type: body {fp_b}, alarm {fp_a}]   <- safety-critical")
    mk = sum(r["verdict"] == "world" for r in b_w)
    say(f"  MASKING (body+world -> world)   {frac(mk, len(b_w))}")
    say(f"  reference, body only -> world   {frac(sum(r['verdict'] == 'world' for r in b_only), len(b_only))}")
    say(f"  reference, neither -> body/alarm {frac(sum(r['verdict'] != 'world' for r in neither), len(neither))}")

    # ---- by push magnitude and duration
    say("\nBY PUSH MAGNITUDE (and duration)")
    say(f"  {'push':<20}{'false positive (world only)':>30}{'masking (body+world)':>26}")
    for tq in fc.PUSH_TORQUES:
        for d in list(fc.PUSH_DURATIONS) + [None]:
            sel = lambda rs: [r for r in rs if abs(r["torque"]) == tq and (d is None or r["duration_name"] == d)]
            a, b = sel(w_only), sel(b_w)
            fpk = sum(r["verdict"] != "world" for r in a); mkk = sum(r["verdict"] == "world" for r in b)
            label = f"{tq:g} N*m {d or 'all'}"
            say(f"  {label:<20}{f'{fpk}/{len(a)} = {fpk / len(a):.0%}':>30}{f'{mkk}/{len(b)} = {mkk / len(b):.0%}':>26}")

    # ---- margin: do the values the rule uses overlap between body-present and body-absent?
    say("\nMARGIN (values the rule actually used)")
    for name, has_body in (("body present", True), ("body absent", False)):
        rs = [r for r in trial_rows if r["has_body"] == has_body]
        st = [r for r in rs if r["settled"]]
        sc = np.array([r["settled_score"] for r in st])
        say(f"  {name:<13} settled {len(st)}/{len(rs)}; settled score min {sc.min():.4f} "
            f"median {np.median(sc):.4f} max {sc.max():.4f}; settle time median "
            f"{np.median([r['settle_time_s'] for r in st]):.2f} s")
    sb = [r["settled_score"] for r in trial_rows if r["has_body"] and r["settled"]]
    sn = [r["settled_score"] for r in trial_rows if not r["has_body"] and r["settled"]]
    gap = min(sb) - max(sn)
    say(f"  settled-score gap (lowest body-present - highest body-absent): {gap:+.4f} "
        f"({'clean separation' if gap > 0 else 'OVERLAP'}); threshold {body_thr:.4f}")
    for name, has_body in (("body present", True), ("body absent", False)):
        un = [r for r in trial_rows if r["has_body"] == has_body and not r["settled"]]
        if un:
            f = np.array([r["late_frac_above_alarm"] for r in un])
            m = np.array([r["late_block_median"] for r in un])
            say(f"  {name:<13} never settled {len(un)}: late-window fraction of blocks above alarm "
                f"min {f.min():.2f} median {np.median(f):.2f} max {f.max():.2f}; block median "
                f"{m.min():.4f}-{m.max():.4f} (alarm needs fraction >= {ac.PERSIST_FRAC})")
        else:
            say(f"  {name:<13} never settled: 0")

    say("\nverdict paths per cell: " + "; ".join(
        f"{c}: " + ", ".join(f"{p} {sum(r['path'] == p and r['cell'] == c for r in trial_rows)}"
                             for p in ("settled", "persistent", "decayed")) for c in CELLS))

    keys = ["cell", "seed", "has_body", "has_push", "param", "factor", "torque", "duration_name",
            "duration", "body_onset", "push_onset", "offset_steps", "offset_s", "onset", "verdict",
            "correct", "path", "verdict_time_s", "settled", "settled_score", "settle_time_s",
            "late_block_median", "late_frac_above_alarm"]
    with open(os.path.join(OUT, "trials.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        w.writerows([{k: r[k] for k in keys} for r in res])
    say(f"\nper-trial results (incl. {len(calib)} calibration rows): {fc.RESULTS}/trials.csv")
    open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    sys.exit(main())
