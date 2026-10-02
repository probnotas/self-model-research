"""Figures for the classifier evaluation.

  scores.png          every trial's late-window score, by class, with the threshold
  confusion.png       the 2x2 confusion matrix at the operating threshold
  roc.png             FPR vs TPR over all thresholds, operating point marked
  boundary_body.png   detection rate vs body-change size, per parameter
  boundary_world.png  rejection rate over the push torque x duration grid

Colours follow the earlier figure: baseline/undisturbed blue, body change
orange, world event green. Identity never relies on colour alone: every series
is also labelled in text.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import classify_config as cc
from classifier import BODY, WORLD

C_NONE, C_BODY, C_WORLD = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"
PARAM_NAME = {"l": "length", "m": "mass"}


def _style(ax):
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def _save(fig, path):
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


def plot_scores(calib, main, threshold, path):
    """Strip plot: one dot per trial, log x-axis, threshold as a vertical line.

    This is the most direct picture of the classifier: everything right of the
    line is called "body change", everything left "world event".
    """
    rows = [("undisturbed\n(calibration)", [r["score"] for r in calib], C_NONE),
            ("world event", [r["score"] for r in main if r["true_class"] == WORLD], C_WORLD),
            ("body change", [r["score"] for r in main if r["true_class"] == BODY], C_BODY)]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, (label, s, c) in enumerate(rows):
        y = i + rng.uniform(-0.18, 0.18, size=len(s))
        ax.scatter(s, y, s=22, color=c, alpha=0.75, edgecolors="white", linewidths=0.6)
    ax.axvline(threshold, color=INK, ls="--", lw=1.4)
    ax.text(threshold, len(rows) - 0.45, f"  threshold = {threshold:.4g}", color=INK,
            fontsize=9, va="bottom")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xscale("log")
    ax.set_xlabel(f"late-window score: mean one-step prediction error, "
                  f"{cc.LATE_START * 0.05:g}–{cc.LATE_END * 0.05:g} s after onset (log)")
    ax.set_title("Late-window score per trial  ·  right of the line = called \"body change\"",
                 loc="left", fontsize=11, color=INK)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    _style(ax)
    return _save(fig, path)


def plot_confusion(cm, path):
    m = np.array([[cm["body->body"], cm["body->world"]],
                  [cm["world->body"], cm["world->world"]]])
    labels = [["true positive", "false negative"], ["FALSE POSITIVE\n(safety-critical)", "true negative"]]
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.imshow(m, cmap="Blues", vmin=0, vmax=max(1, m.max()))
    for i in range(2):
        for j in range(2):
            dark = m[i, j] > 0.6 * m.max()
            ax.text(j, i, f"{m[i, j]}\n{labels[i][j]}", ha="center", va="center",
                    fontsize=10, color="white" if dark else INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['predicted\n"body change"', 'predicted\n"world event"'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true\nbody change", "true\nworld event"])
    ax.set_title("Confusion matrix (main evaluation)", loc="left", fontsize=11, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    return _save(fig, path)


def plot_roc(roc_rows, auc_value, op_fpr, op_tpr, path):
    f = [r[1] for r in roc_rows]; t = [r[2] for r in roc_rows]
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1, label="chance")
    ax.step(f, t, where="post", color=C_BODY, lw=2, label=f"classifier (AUC = {auc_value:.3f})")
    ax.plot([op_fpr], [op_tpr], "o", ms=9, color=INK, mec="white", mew=1.5,
            label=f"operating point (FPR {op_fpr:.2f}, TPR {op_tpr:.2f})")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("false-positive rate  (world event called \"body change\")  ← safety-critical")
    ax.set_ylabel("true-positive rate  (body change detected)")
    ax.set_title("ROC: every threshold on the late-window score", loc="left", fontsize=11, color=INK)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    _style(ax)
    return _save(fig, path)


def plot_boundary_body(table, path):
    """Detection rate vs multiplier, one line per parameter, Wilson 95% bars."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for param, color, ls in (("l", C_BODY, "-"), ("m", "#4a3aa7", "--")):
        rows = [r for r in table if r[0][0] == param]
        if not rows:
            continue
        x = [r[0][1] for r in rows]; y = [r[3] for r in rows]
        lo = [r[3] - r[4][0] for r in rows]; hi = [r[4][1] - r[3] for r in rows]
        ax.errorbar(x, y, yerr=[lo, hi], color=color, ls=ls, marker="o", ms=6, lw=2,
                    capsize=3, label=f"{PARAM_NAME[param]} × factor")
    ax.axhline(cc.RELIABLE_RATE, color=MUTED, ls=":", lw=1.2)
    ax.text(ax.get_xlim()[0], cc.RELIABLE_RATE + 0.015, f" reliable = {cc.RELIABLE_RATE:.0%}",
            color=MUTED, fontsize=9)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("body-change multiplier")
    ax.set_ylabel("fraction called \"body change\" (detected)")
    ax.set_title(f"Body-change boundary  ·  {cc.N_BOUNDARY} trials per point, 95% Wilson bars",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax)
    return _save(fig, path)


def plot_boundary_world(table, path):
    """Heatmap: rejection rate for each (torque, duration) cell, value printed."""
    torques = sorted({r[0][0] for r in table}); durs = sorted({r[0][1] for r in table})
    m = np.full((len(durs), len(torques)), np.nan)
    for (tq, d), k, n, rate, _ in table:
        m[durs.index(d), torques.index(tq)] = rate
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1, origin="lower", aspect="auto")
    for i in range(len(durs)):
        for j in range(len(torques)):
            v = m[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.6 else INK)
    ax.set_xticks(range(len(torques))); ax.set_xticklabels([f"{t:g}" for t in torques])
    ax.set_yticks(range(len(durs))); ax.set_yticklabels([f"{d} steps ({d * 0.05:g} s)" for d in durs])
    ax.set_xlabel("push torque magnitude (N·m)   [controller limit: 2 N·m]")
    ax.set_ylabel("push duration")
    ax.set_title(f"World-event boundary: fraction correctly called \"world event\"  ·  "
                 f"{cc.N_BOUNDARY} trials per cell", loc="left", fontsize=11, color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04); cb.set_label("rejection rate")
    return _save(fig, path)
