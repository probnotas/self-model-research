"""Does a short-history forward model put the surprise signal in the right order?

    python run_history.py                    # every config; resumes from cached episodes/models
    python run_history.py --only soft_low    # record episodes for just these configs
    python run_history.py --summary          # score configs whose episodes exist

Paired comparison on the SAME trials as results_compliant/ (same configs,
factorial, MASTER_SEED 90000, start states, and the same single-state model
driving the controller). The rule reads either the single-state model's
one-step error (old) or a K-history model's (new, K = 2, 4, 8).
See history_config.py for what is held fixed.

Episodes are re-recorded (results_compliant/ did not log actions, which a
history model needs) with the identical loop, and every trial's single-state
error trace is checked bit-for-bit against the traces regenerated for
results_calibrated/, and its settled score and verdict against the logged
results_compliant/ CSV.

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

import alarm_config as ac
import classifier
import compliant_config as kc
import compliant_env as ce
import compliant_trials as ct
import config
import factorial_config as fc
import history_config as hc
import history_model as hm
import model as model_lib
import overlap_trials as ot
import planner
import plot_history
import run_factorial as rf
from run_calibrated import apply_rule, check_reproduction, metrics_for
import alarm_classifier as A

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, hc.RESULTS)
EP_DIR = os.path.join(OUT, hc.EPISODES)
OLD = os.path.join(HERE, kc.RESULTS)
CALIB_TRACES = os.path.join(HERE, "results_calibrated", "traces")
LOG = []


def say(msg=""):
    print(msg, flush=True)
    LOG.append(msg)


# ---------------------------------------------------------------------------
# 1. record the same episodes, now with observations and actions
# ---------------------------------------------------------------------------

def record_episode(m, cfg, spec, steps):
    """compliant_trials.run_episode, line for line, also returning obs and actions."""
    env = ce.make(cfg, max_steps=steps, noise_seed=spec["seed"] + 1_000_000)
    env.reset(seed=spec["seed"])
    if config.START_MODE == "upright":
        rng = np.random.default_rng(spec["seed"]); r = config.START_RANGE
        env.unwrapped.state = np.array([rng.uniform(-r, r), rng.uniform(-r, r)])
    obs = env.observe()
    np.random.seed(spec["seed"])
    O, U, err, th, thd = [obs], [], [], [], []
    for t in range(steps):
        if spec["has_body"] and t == spec["body_onset"]:
            env.change_body({spec["param"]: spec["factor"]})
        if spec["has_push"]:
            if t == spec["push_onset"]:
                env.set_external_torque(spec["torque"])
            if t == spec["push_onset"] + spec["duration"]:
                env.set_external_torque(0.0)
        action = planner.choose_action(m, obs)
        with torch.no_grad():
            pred = m.predict_next(torch.from_numpy(obs.astype(np.float32))[None],
                                  torch.from_numpy(action)[None])[0].numpy()
        nxt, *_ = env.step(action)
        err.append(float(np.linalg.norm(nxt - pred)))
        th.append(float(np.arctan2(nxt[1], nxt[0])))
        thd.append(float(nxt[2]))
        O.append(nxt); U.append(action)
        obs = nxt
    env.close()
    return (np.array(O, np.float32), np.array(U, np.float32), np.array(err, np.float32),
            np.array(th, np.float32), np.array(thd, np.float32))


_M = {}


def _worker(spec):
    if spec["model_path"] not in _M:
        _M[spec["model_path"]] = model_lib.load_model(spec["model_path"])
    out = dict(spec)
    out["obs"], out["act"], out["err"], out["theta"], out["thdot"] = record_episode(
        _M[spec["model_path"]], spec["cfg"], spec, spec["steps"])
    return out


def single_model_path(name, cfg):
    return (os.path.join(HERE, "forward_model.pt") if name == "rigid_none"
            else ct.get_model(name, cfg)[1])                 # cached, not retrained


def episodes(name, cfg, regenerate):
    path = os.path.join(EP_DIR, f"{name}.npz")
    specs = rf.make_specs()
    if not os.path.exists(path):
        if not regenerate:
            return None
        mp = single_model_path(name, cfg)
        for s in specs:
            s.update(cfg=cfg, model_path=mp, steps=ot.STEPS, config=name)
        import multiprocessing as mpc
        res = []
        with mpc.get_context("fork").Pool(kc.N_WORKERS, initializer=ct._init_worker) as pool:
            for i, r in enumerate(pool.imap(_worker, specs, chunksize=2), 1):
                res.append(r)
                if i % 100 == 0 or i == len(specs):
                    print(f"    {name} {i}/{len(specs)}", flush=True)
        os.makedirs(EP_DIR, exist_ok=True)
        np.savez_compressed(path, seed=[r["seed"] for r in res], **{
            k: np.stack([r[k] for r in res]) for k in ("obs", "act", "err", "theta", "thdot")})
    d = np.load(path)
    assert list(d["seed"]) == [s["seed"] for s in specs]
    for i, s in enumerate(specs):
        s.update(config=name, obs=d["obs"][i], act=d["act"][i], err=d["err"][i],
                 theta=d["theta"][i], thdot=d["thdot"][i])
    return specs


def trace_mismatches(name, res):
    """Compare single-state error traces with results_calibrated's regenerated traces."""
    p = os.path.join(CALIB_TRACES, f"{name}.npz")
    if not os.path.exists(p):
        return None
    ref = np.load(p)["err"]
    return int(sum(not np.array_equal(r["err"], ref[i]) for i, r in enumerate(res)))


# ---------------------------------------------------------------------------
# 2. score every signal with the unchanged rule and its own calibration
# ---------------------------------------------------------------------------

def body_threshold(res, sig_key):
    cs = [A.settled_score(r[sig_key], r["theta"], r["thdot"], r["onset"])[0]
          for r in res if r["cell"] == "calib"]
    finite = [s for s in cs if s is not None]
    return (classifier.calibrate_threshold(finite) if len(finite) >= 2 else float("nan")), len(finite)


def run_config(name, cfg, res, alarm_thr):
    single = model_lib.load_model(single_model_path(name, cfg))
    hist, train_mse = {}, {}
    for k in hc.K_SWEEP:
        hist[k], train_mse[k] = hm.get(name, cfg, k)
    held = hm.heldout(single, hist, cfg)
    for r in res:
        for k, m in hist.items():
            r[f"err_k{k}"] = hm.episode_errors(m, r["obs"], r["act"])
    tags = {"old": "err", **{f"k{k}": f"err_k{k}" for k in hc.K_SWEEP}}
    thr = {}
    for tag, key in tags.items():
        thr[tag], n_settled = body_threshold(res, key)
        for r in res:
            settled, st = apply_rule(r, r[key], thr[tag], alarm_thr, tag)
            if tag == "old":
                r["settled"], r["settle_time_s"] = settled, st
                r["raw_settled_score"] = r["old_settled_score"]
    bad_score, bad_verdict = check_reproduction(name, res)
    rows = [r for r in res if r["cell"] != "calib"]
    logged_held = json.load(open(os.path.join(OLD, f"{name}.json")))["heldout_error"]
    out = dict(config=name, stiffness=cfg["stiffness"], noise=list(cfg["noise"]),
               heldout=held, heldout_single_logged=logged_held, train_mse=train_mse,
               body_threshold=thr, alarm_threshold=alarm_thr,
               trace_mismatches=trace_mismatches(name, res),
               repro_settled_score_mismatches=bad_score, repro_verdict_mismatches=bad_verdict,
               **{tag: metrics_for(rows, tag) for tag in tags})
    keys = ["config", "cell", "seed", "has_body", "has_push", "param", "factor", "torque",
            "duration_name", "offset_s", "onset", "settled", "settle_time_s"]
    for tag in tags:
        keys += [f"{tag}_settled_score", f"{tag}_verdict", f"{tag}_path",
                 f"{tag}_late_block_median", f"{tag}_late_frac_above_alarm"]
    keys += ["logged_settled_score", "logged_verdict"]
    ren = lambda k: k.replace("old_", "single_", 1) if k.startswith("old_") else k
    with open(os.path.join(OUT, f"trials_{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[ren(k) for k in keys]); w.writeheader()
        w.writerows([{ren(k): r[k] for k in keys} for r in res])
    json.dump(out, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
    return out


# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default="", help="comma-separated configs to record episodes for")
    ap.add_argument("--summary", action="store_true", help="never record; score cached episodes only")
    args = ap.parse_args()
    only = set(filter(None, args.only.split(",")))
    os.makedirs(OUT, exist_ok=True)
    versions = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                    gymnasium=gymnasium.__version__, master_seed=fc.MASTER_SEED,
                    model_train_seed=config.SEED, k_sweep=list(hc.K_SWEEP), k_main=hc.K_MAIN,
                    train_threads=hc.TRAIN_THREADS, reused_from=kc.RESULTS,
                    controller="single-state model (forward_model.pt / compliant_models/*.pt), unchanged")
    json.dump(versions, open(os.path.join(OUT, "versions_and_seed.json"), "w"), indent=1)
    alarm_thr = json.load(open(os.path.join(HERE, ac.RESULTS, "chosen_alarm.json")))["alarm_threshold"]
    say("SIMULATED compliant body (series-elastic + hysteresis + damping + sensor noise): "
        "a model of compliance, not real hardware.")
    say(f"versions/seed: {versions}")

    results = []
    for s, k in kc.STIFFNESS.items():
        for n, nz in kc.NOISE.items():
            name, cfg = f"{s}_{n}", dict(stiffness=k, noise=nz)
            cached = os.path.exists(os.path.join(OUT, f"{name}.json"))
            res = episodes(name, cfg, regenerate=not args.summary and (not only or name in only))
            if res is None:
                continue
            r = run_config(name, cfg, res, alarm_thr)
            results.append(r)
            h = r["heldout"]
            say(f"\n== {name} == held-out error: single {h['single']:.4f} (logged, all targets "
                f"{r['heldout_single_logged']:.4f}) | " + " | ".join(f"k{k} {h[f'k{k}']:.4f}" for k in hc.K_SWEEP))
            say(f"  reproduction: error traces {r['trace_mismatches']} mismatches; logged CSV "
                f"{r['repro_settled_score_mismatches']} settled-score, {r['repro_verdict_mismatches']} "
                f"verdict mismatches of 530")
            for tag in ["old"] + [f"k{k}" for k in hc.K_SWEEP]:
                m = r[tag]
                say(f"  {'single' if tag == 'old' else tag:<7} thr {r['body_threshold'][tag]:.4f} | "
                    f"FP {m['false_positive']['k']:>3}/120 (body {m['fp_body']}, alarm {m['fp_alarm']}) | "
                    f"mask {m['masking']['k']:>3} | miss(no push) {m['miss_body_only']['k']:>3} | "
                    f"flag(neither) {m['false_alarm_neither']['k']:>3} | misattr {m['misattribution']['rate']:.1%} | "
                    f"gap {m['gap']:+.4f} | AUC {m['settled_auc']:.3f}")

    km = f"k{hc.K_MAIN}"
    say(f"\nSUMMARY: single-state (old) vs {hc.K_MAIN}-step history (new), same trials; counts of 120")
    say(f"  {'config':<15}{'held single':>12}{'held k4':>9}{'AUC old':>9}{'k2':>7}{'k4':>7}{'k8':>7}"
        f"{'FP old':>8}{'k4':>5}{'mask old':>10}{'k4':>5}{'miss old':>10}{'k4':>5}{'gap old':>10}{'gap k4':>9}")
    for r in results:
        o, n = r["old"], r[km]
        say(f"  {r['config']:<15}{r['heldout']['single']:>12.4f}{r['heldout'][km]:>9.4f}"
            f"{o['settled_auc']:>9.3f}" + "".join(f"{r[f'k{k}']['settled_auc']:>7.3f}" for k in hc.K_SWEEP) +
            f"{o['false_positive']['k']:>8}{n['false_positive']['k']:>5}{o['masking']['k']:>10}{n['masking']['k']:>5}"
            f"{o['miss_body_only']['k']:>10}{n['miss_body_only']['k']:>5}{o['gap']:>+10.4f}{n['gap']:>+9.4f}")
    json.dump(results, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    fig = plot_history.sweep(results, os.path.join(OUT, "sweep_history.png"))
    say(f"\nfigure: {os.path.relpath(fig, HERE)}")
    open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    sys.exit(main())
