"""Trials for the 3-way classifier: five categories, full 10 s traces.

Each trial is a spec dict (as in trials.py) plus:
    category   calib / body / world / severe / bigpush (how it was generated)
    truth      none / body / world / severe          (what the right answer is)

body and severe differ only in the size of the change (moderate x1.2-1.5 vs
severe x1.5-1.7); world and bigpush are both world events (true class world),
kept as separate categories so the big-push failure mode is reported on its own.

trials.py is reused, not changed: its _init_worker loads the frozen model into
each worker process and spec_to_condition turns a spec into a schedule.
"""
import numpy as np

import alarm_config as ac
import classify_config as cc
import run_experiment
import trials

STEPS = cc.ONSET_MAX + ac.EXT_MAX_LAG + ac.EPISODE_MARGIN


def _onset(rng):
    return int(rng.integers(cc.ONSET_MIN, cc.ONSET_MAX + 1))


def _body(rng, seed, lo, hi, category):
    return dict(block=category, category=category,
                truth="severe" if category == "severe" else "body",
                true_class="body", param=str(rng.choice(cc.BODY_PARAMS)),
                factor=float(rng.uniform(lo, hi)), torque=0.0, duration=0,
                onset=_onset(rng), seed=int(seed))


def _push(rng, seed, tq_lo, tq_hi, d_lo, d_hi, category):
    sign = 1.0 if rng.random() < 0.5 else -1.0
    return dict(block=category, category=category, truth="world", true_class="world",
                param="", factor=1.0, torque=float(sign * rng.uniform(tq_lo, tq_hi)),
                duration=int(rng.integers(d_lo, d_hi + 1)), onset=_onset(rng), seed=int(seed))


def _none(rng, seed):
    return dict(block="calib", category="calib", truth="none", true_class="none",
                param="", factor=1.0, torque=0.0, duration=0, onset=_onset(rng), seed=int(seed))


def make_specs(counts, base):
    """All categories, each from its own seed sub-block (base + 1000 * k)."""
    specs = []
    for k, (cat, n) in enumerate(counts.items()):
        rng = np.random.default_rng(base + 1000 * k)
        for i in range(n):
            seed = base + 1000 * k + i
            if cat == "calib":
                specs.append(_none(rng, seed))
            elif cat == "body":
                specs.append(_body(rng, seed, ac.BODY_MIN, ac.BODY_MAX, "body"))
            elif cat == "severe":
                specs.append(_body(rng, seed, ac.SEVERE_MIN, ac.SEVERE_MAX, "severe"))
            elif cat == "world":
                specs.append(_push(rng, seed, cc.WORLD_TORQUE_MIN, cc.WORLD_TORQUE_MAX,
                                   cc.WORLD_STEPS_MIN, cc.WORLD_STEPS_MAX, "world"))
            elif cat == "bigpush":
                specs.append(_push(rng, seed, ac.BIG_TQ_MIN, ac.BIG_TQ_MAX,
                                   ac.BIG_STEPS_MIN, ac.BIG_STEPS_MAX, "bigpush"))
    return specs


def run_long_trial(spec):
    """One episode, long enough for the full 10 s window; keep the traces.

    Runs inside a worker process, where trials._init_worker has loaded the
    frozen model into trials._MODEL.
    """
    rows = run_experiment.run_episode(trials._MODEL, trials.spec_to_condition(spec),
                                      seed=spec["seed"], steps=STEPS, onset=spec["onset"])
    out = dict(spec)
    out["err"] = np.array([r["surprise"] for r in rows], dtype=np.float32)
    out["theta"] = np.array([r["theta"] for r in rows], dtype=np.float32)
    out["thdot"] = np.array([r["theta_dot"] for r in rows], dtype=np.float32)
    return out


def run(specs, label):
    return trials.run_all(specs, label, fn=run_long_trial)
