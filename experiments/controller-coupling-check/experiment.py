"""Does the main experiment's controller, reacting to a heavier body, cause the motion drop?

    python experiments/controller-coupling-check/experiment.py

Reuses, unchanged: compliant_env (soft, k = 4, no noise), the cached learned forward
model compliant_models/soft_none.pt, the CEM planner (planner.choose_action) and the
main experiment's episode loop run_history.record_episode. Paired healthy / mass x1.5
runs on 30 fresh seeds. Writes episodes.npz, results.json, three plots and RESULTS.md.
SIMULATION ONLY.
"""
import json
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SA = os.path.join(HERE, "..", "..", "surprise_attribution")
sys.path.insert(0, SA)
os.chdir(SA)                                    # some modules resolve paths relative to their folder
import compliant_config as kc  # noqa: E402
import compliant_trials as ct  # noqa: E402
import factorial_config as fc  # noqa: E402
import overlap_trials as ot  # noqa: E402
import run_history as rh  # noqa: E402

NAME = "soft_none"
CFG = dict(stiffness=kc.STIFFNESS["soft"], noise=kc.NOISE["none"])
N, SEED0, FACTOR = 30, 640000, 1.5
W0, W1 = 20, 100                                 # run_twosided.py window: onset+20 .. onset+100
MAIN_RATIO = 0.0150 / 0.0501                     # main experiment, soft body (findings_arc.md, Exp 6)
CONDS = ("healthy", "mass x1.5")


def specs():
    rng = np.random.default_rng(SEED0)
    model_path = ct.get_model(NAME, CFG)[1]      # cached file; asserts it exists rather than retraining
    assert os.path.exists(model_path)
    out = []
    for i in range(N):
        onset = int(rng.integers(fc.BODY_ONSET_RANGE[0], fc.BODY_ONSET_RANGE[1] + 1))
        for cond in CONDS:
            out.append(dict(cond=cond, seed=SEED0 + i, has_body=cond != "healthy", has_push=False,
                            param="m", factor=FACTOR, body_onset=onset, push_onset=-1, duration=0,
                            onset=onset, steps=ot.STEPS, cfg=CFG, model_path=model_path))
    return out


def boot_ratio(x, y, n=10000, seed=0):
    """Geometric mean of paired ratios x/y with a 95% bootstrap CI."""
    lr = np.log(np.asarray(x) / np.asarray(y))
    rng = np.random.default_rng(seed)
    bs = np.exp(lr[rng.integers(0, len(lr), (n, len(lr)))].mean(1))
    return dict(point=float(np.exp(lr.mean())), ci95=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))])


def main():
    sp = specs()
    with mp.get_context("fork").Pool(4, initializer=ct._init_worker) as pool:
        res = []
        for i, r in enumerate(pool.imap(rh._worker, sp, chunksize=1), 1):
            res.append(r)
            if i % 10 == 0:
                print(f"  {i}/{len(sp)} episodes", flush=True)
    by = {c: sorted([r for r in res if r["cond"] == c], key=lambda r: r["seed"]) for c in CONDS}

    # per-timestep logs (obs has steps+1 rows; everything else has steps)
    log = {c: dict(theta=np.stack([r["theta"] for r in by[c]]), thdot=np.stack([r["thdot"] for r in by[c]]),
                   torque=np.stack([r["act"][:, 0] for r in by[c]]), err=np.stack([r["err"] for r in by[c]]),
                   motion=np.stack([np.linalg.norm(r["obs"][1:] - r["obs"][:-1], axis=1) for r in by[c]]),
                   onset=np.array([r["onset"] for r in by[c]]))
           for c in CONDS}
    np.savez_compressed(os.path.join(HERE, "episodes.npz"), seeds=np.array([r["seed"] for r in by["healthy"]]),
                        **{f"{c.split()[0]}_{k}": v for c in CONDS for k, v in log[c].items()})

    pre_diff = max(float(np.abs(log["healthy"][k][i, :o] - log["mass x1.5"][k][i, :o]).max())
                   for k in ("theta", "torque", "err") for i, o in enumerate(log["healthy"]["onset"]))

    def win(c, key, absval=False):
        X = np.abs(log[c][key]) if absval else log[c][key]
        return np.array([X[i, o + W0:o + W1].mean() for i, o in enumerate(log[c]["onset"])])

    def pre(c, key, absval=False):
        X = np.abs(log[c][key]) if absval else log[c][key]
        return np.array([X[i, o - 40:o].mean() for i, o in enumerate(log[c]["onset"])])

    metrics = {}
    for name, key, ab in (("motion", "motion", False), ("torque_abs", "torque", True), ("pred_error", "err", False)):
        h, m = win("healthy", key, ab), win("mass x1.5", key, ab)
        metrics[name] = dict(healthy=dict(mean=float(h.mean()), std=float(h.std(ddof=1)), median=float(np.median(h))),
                             changed=dict(mean=float(m.mean()), std=float(m.std(ddof=1)), median=float(np.median(m))),
                             ratio_of_means=float(m.mean() / h.mean()), paired_ratio=boot_ratio(m, h),
                             pre_onset_mean=float(pre("healthy", key, ab).mean()))
    sat = {c: float((np.abs(np.concatenate([log[c]["torque"][i, o + W0:o + W1] for i, o in
                                            enumerate(log[c]["onset"])])) >= 1.99).mean()) for c in CONDS}
    hold = {c: float(np.median(win(c, "theta"))) for c in CONDS}   # mean signed angle in window: offset from upright?
    upright = {c: float(np.mean([np.abs(log[c]["theta"][i, o + W0:o + W1]).max() < np.pi / 2
                                 for i, o in enumerate(log[c]["onset"])])) for c in CONDS}

    mot, tq = metrics["motion"]["paired_ratio"], metrics["torque_abs"]["paired_ratio"]
    motion_drops = mot["ci95"][1] < 1
    if not motion_drops:
        outcome = "none: motion did not drop under these conditions"
    elif tq["ci95"][1] < 1 and tq["point"] <= 0.90:
        outcome = "A"
    elif tq["ci95"][0] > 1 and tq["point"] >= 1.10:
        outcome = "C"
    else:
        outcome = "B"
    summary = dict(config=NAME, n_seeds=N, seeds=[SEED0, SEED0 + N - 1], factor=FACTOR, window=[W0, W1],
                   pre_onset_identical_max_abs_diff=pre_diff, metrics=metrics, torque_saturation_fraction=sat,
                   median_window_mean_angle=hold, fraction_upright_in_window=upright,
                   main_experiment_motion_ratio=MAIN_RATIO, outcome=outcome)
    json.dump(summary, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    plots(log)
    write_results(summary)
    print(json.dumps(summary, indent=1))


def plots(log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"healthy": "#1b9e77", "mass x1.5": "#d95f02"}
    lag = np.arange(-40, W1 + 20)
    for key, ylab, fname, ab, title in (("motion", "motion ‖s(t+1) − s(t)‖", "motion_vs_time.png", False, "motion"),
                                        ("torque", "controller torque |u| (N·m)", "torque_vs_time.png", True, "controller torque"),
                                        ("err", "prediction error ‖s(t+1) − ŝ(t+1)‖", "prediction_error_vs_time.png", False,
                                         "prediction error")):
        fig, ax = plt.subplots(figsize=(8, 4))
        for c in CONDS:
            X = np.abs(log[c][key]) if ab else log[c][key]
            A = np.stack([X[i, o + lag] for i, o in enumerate(log[c]["onset"])])
            t = lag * 0.05
            ax.plot(t, A.mean(0), color=col[c], label=f"{c} (mean of {len(A)} seeds)")
            ax.fill_between(t, A.mean(0) - A.std(0), A.mean(0) + A.std(0), color=col[c], alpha=0.2)
        ax.axvline(0, color="red", lw=1.5, label="body change (mass ×1.5)")
        ax.axvspan(W0 * 0.05, W1 * 0.05, color="grey", alpha=0.1, label="scoring window")
        ax.set_xlabel("time since onset (s)"); ax.set_ylabel(ylab); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_title(f"Soft compliant pendulum, main-experiment controller: {title}")
        fig.tight_layout(); fig.savefig(os.path.join(HERE, fname), dpi=130); plt.close(fig)


def write_results(s):
    M = s["metrics"]
    f = lambda d: f"{d['mean']:.4f} ± {d['std']:.4f}"
    pr = lambda d: f"×{d['point']:.2f} (95% CI {d['ci95'][0]:.2f}–{d['ci95'][1]:.2f})"
    meaning = {"A": "controller output DROPS on the heavier body → coupling effect confirmed",
               "B": "controller output about the SAME, motion still drops → something else, investigate further",
               "C": "controller output RISES on the heavier body but motion still drops → compensating but failing, "
                    "a different mechanism"}
    L = ["# Results: controller coupling check", "",
         f"*Simulation only. Soft compliant pendulum ({s['config']}), the main experiment's CEM planner on the learned "
         f"model `soft_none.pt`, {s['n_seeds']} paired seeds ({s['seeds'][0]}–{s['seeds'][1]}), mass ×{s['factor']} at "
         f"the onset. Window: onset+{s['window'][0]} to onset+{s['window'][1]} steps (1–5 s after). All numbers come "
         f"from `results.json`.*", "",
         f"## Outcome: **{s['outcome']}**" + (f": {meaning[s['outcome']]}" if s['outcome'] in meaning else ""), "",
         "| metric (window mean per seed) | healthy (mean ± SD) | mass ×1.5 (mean ± SD) | ratio of means | paired ratio |",
         "|---|---|---|---|---|"]
    for k, lab in (("motion", "motion ‖Δs‖"), ("torque_abs", "controller torque magnitude (N·m)"),
                   ("pred_error", "prediction error ‖s − ŝ‖")):
        L.append(f"| {lab} | {f(M[k]['healthy'])} | {f(M[k]['changed'])} | ×{M[k]['ratio_of_means']:.2f} | {pr(M[k]['paired_ratio'])} |")
    L += ["",
          f"- **Main experiment's motion drop:** ×{s['main_experiment_motion_ratio']:.2f} (median 0.0501 → 0.0150).",
          f"- **Motion medians here:** {M['motion']['healthy']['median']:.4f} → {M['motion']['changed']['median']:.4f}.",
          f"- **Torque at the ±2 N·m limit in the window:** healthy {s['torque_saturation_fraction']['healthy']:.1%}, "
          f"mass ×1.5 {s['torque_saturation_fraction']['mass x1.5']:.1%} of steps.",
          f"- **Still upright (|θ| < 90°) throughout the window:** healthy {s['fraction_upright_in_window']['healthy']:.0%}, "
          f"mass ×1.5 {s['fraction_upright_in_window']['mass x1.5']:.0%} of seeds.",
          f"- **Check:** healthy and changed runs are identical before the onset (max difference "
          f"{s['pre_onset_identical_max_abs_diff']:.1e}).", "",
          "Decision rule (fixed in README.md before running): torque paired-ratio CI entirely below 1 and ≤ 0.90 → A; "
          "entirely above 1 and ≥ 1.10 → C; otherwise B. This applies only if the motion paired-ratio CI is entirely below 1.", "",
          "Plots: `motion_vs_time.png`, `torque_vs_time.png`, `prediction_error_vs_time.png` (mean ± SD across seeds, "
          "aligned at the onset).", ""]
    open(os.path.join(HERE, "RESULTS.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    main()
