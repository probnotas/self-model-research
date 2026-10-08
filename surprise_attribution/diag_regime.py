"""Diagnostic: is the soft-body model worse than "nothing changes" everywhere, or only when quiet?

    python diag_regime.py

Reuses recorded data only (results_history/episodes/, the single-state model's
one-step errors logged there). No model is run or trained; nothing is simulated.

For every transition t (state s_t -> s_{t+1}, both as observed):
    model error  = recorded one-step error ||s_{t+1} - f(s_t, a_t)||
    hold error   = ||s_{t+1} - s_t||            ("nothing changes" predictor)

HEALTHY DYNAMICS ONLY. Two sources, both with the body unchanged:
  * healthy episodes: the 50 calibration + 120 "neither" trials, every step;
  * push-only trials, only the steps AFTER the push has switched off (the push
    is the only thing the model does not know about; once it is off, the
    dynamics are the healthy ones again). The push steps themselves are excluded.
  The healthy episodes alone contain no high-motion steps (all start within
  0.2 rad of upright and balance), so the second source is where high motion
  comes from. The random-action data the models were trained on is not on disk.

REGIME SPLIT, fixed before looking at errors: |theta_dot| of the observed state
the prediction starts from, at OMEGA_CALM = 0.5 rad/s - the repo's existing
definition of "slow" (classify_config.py), chosen long before this check.
  low motion  |theta_dot| <  0.5     high motion  |theta_dot| >= 0.5
Finer |theta_dot| bins inside each regime are also reported.

SIMULATED compliant body, not hardware.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import classify_config as cc
import run_factorial as rf
from plot_classifier import INK, MUTED, _save, _style

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_regime")
CONFIGS = ("rigid_none", "soft_none", "soft_low")
BINS = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, np.inf)
THR = cc.OMEGA_CALM


def steps(name):
    """Pooled healthy-dynamics transitions: model error, hold error, |thdot| at start, source."""
    specs = rf.make_specs()
    d = np.load(os.path.join(HERE, "results_history", "episodes", f"{name}.npz"))
    obs, err = d["obs"], d["err"]
    hold = np.linalg.norm(obs[:, 1:] - obs[:, :-1], axis=2)
    w0 = np.abs(obs[:, :-1, 2])
    M, H, W, S = [], [], [], []
    for i, s in enumerate(specs):
        if s["cell"] in ("calib", "neither"):
            t = np.arange(err.shape[1]); src = "healthy episode"
        elif s["cell"] == "world_only":
            t = np.arange(s["push_onset"] + s["duration"], err.shape[1]); src = "after push"
        else:
            continue
        M.append(err[i, t]); H.append(hold[i, t]); W.append(w0[i, t]); S += [src] * len(t)
    return (np.concatenate(M).astype(np.float64), np.concatenate(H).astype(np.float64),
            np.concatenate(W).astype(np.float64), np.array(S))


def summarise(m, h, sel):
    if sel.sum() == 0:
        return dict(n=0)
    return dict(n=int(sel.sum()), model_mean=float(m[sel].mean()), hold_mean=float(h[sel].mean()),
                ratio=float(m[sel].mean() / h[sel].mean()),
                frac_model_better=float((m[sel] < h[sel]).mean()))


def main():
    os.makedirs(OUT, exist_ok=True)
    res, log = {}, []
    say = lambda x="": (print(x, flush=True), log.append(x))
    say("SIMULATED compliant body, not hardware. Recorded data only (results_history/episodes/).")
    say(f"Regime split: |theta_dot| at the start of the transition, threshold {THR} rad/s (classify_config.OMEGA_CALM).")
    for name in CONFIGS:
        m, h, w, src = steps(name)
        r = dict(low=summarise(m, h, w < THR), high=summarise(m, h, w >= THR),
                 low_healthy_only=summarise(m, h, (w < THR) & (src == "healthy episode")),
                 low_after_push=summarise(m, h, (w < THR) & (src == "after push")),
                 bins={f"{a:g}-{b:g}": summarise(m, h, (w >= a) & (w < b)) for a, b in zip(BINS[:-1], BINS[1:])})
        res[name] = r
        say(f"\n== {name} ==")
        for k in ("low", "high", "low_healthy_only", "low_after_push"):
            v = r[k]
            say(f"  {k:<17} n={v['n']:>7}" + ("" if not v["n"] else
                f"  model {v['model_mean']:.4f}  nothing-changes {v['hold_mean']:.4f}  ratio {v['ratio']:.2f}"
                f"  model better on {v['frac_model_better']:.0%} of steps"))
        say("  by |theta_dot| bin (rad/s):")
        for b, v in r["bins"].items():
            if v["n"]:
                say(f"    {b:>8}  n={v['n']:>7}  model {v['model_mean']:.4f}  hold {v['hold_mean']:.4f}  ratio {v['ratio']:.2f}")
    json.dump(res, open(os.path.join(OUT, "regime_split.json"), "w"), indent=1)

    # representative episodes: one healthy episode, and one strong push (4 N*m, 0.5 s) with the push window marked
    specs = rf.make_specs()
    calib = next(i for i, s in enumerate(specs) if s["cell"] == "calib")
    push = next(i for i, s in enumerate(specs) if s["cell"] == "world_only" and abs(s["torque"]) == 4.0
                and s["duration_name"] == "medium")
    fig, axes = plt.subplots(2, len(CONFIGS), figsize=(16, 7.5), sharex=True)
    for j, name in enumerate(CONFIGS):
        d = np.load(os.path.join(HERE, "results_history", "episodes", f"{name}.npz"))
        for row, i in enumerate((calib, push)):
            ax = axes[row, j]; obs = d["obs"][i]; e = d["err"][i]
            hold = np.linalg.norm(obs[1:] - obs[:-1], axis=1); w = np.abs(obs[:-1, 2])
            t = np.arange(len(e)) * 0.05
            hi = w >= THR
            ax.fill_between(t, 0, 1, where=hi, transform=ax.get_xaxis_transform(), color="#f2c14e", alpha=0.25, lw=0,
                            label=f"high motion (|θ̇| ≥ {THR})")
            if row == 1:
                s = specs[i]
                ax.axvspan(s["push_onset"] * 0.05, (s["push_onset"] + s["duration"]) * 0.05, color="#c0392b", alpha=0.25,
                           lw=0, label="push on (excluded)")
            ax.plot(t, e, color="#2a78d6", lw=1.1, label="model one-step error")
            ax.plot(t, hold, color="#333333", lw=1.1, ls="--", label="'nothing changes' error")
            ax.set_yscale("log")
            ax.set_title(f"{name}: " + ("healthy episode (calibration #1)" if row == 0 else
                         f"push only, {specs[i]['torque']:+g} N·m for 0.5 s"), loc="left", fontsize=9.5, color=INK)
            if row == 1:
                ax.set_xlabel("time (s)")
            _style(ax)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[1, 0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Model one-step error vs 'nothing changes', by motion regime. SIMULATED compliant body, not hardware",
                 x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    say(f"\nfigure: {os.path.relpath(_save(fig, os.path.join(OUT, 'regime_episodes.png')), HERE)}")
    open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(log) + "\n")


if __name__ == "__main__":
    sys.exit(main())
