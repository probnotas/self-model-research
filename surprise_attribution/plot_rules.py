"""Figures comparing the original rule with the improved one (test set).

  compare_boundary_world.png  push rejection grid, original vs improved, same colour scale
  compare_boundary_body.png   body detection vs size, original (dashed) vs improved (solid)
  compare_roc_boundary.png    ROC over the hard boundary trials, both rules
  decision_delay.png          how long the improved rule waits before deciding
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import classify_config as cc
from classifier import BODY, WORLD
from plot_classifier import C_BODY, C_WORLD, INK, MUTED, _save, _style

C_OLD, C_NEW = "#8a8a8a", "#2a78d6"
LABEL = {"fixed": "original: fixed 1–2 s window",
         "fixed_ratio": "fixed window ÷ own baseline",
         "calm": "improved: wait until calm",
         "calm_ratio": "wait until calm ÷ own baseline",
         "calm_long": "wait until calm, 2 s average",
         "calm_long_ratio": "wait until calm, 2 s average ÷ own baseline"}


def _grid(tab):
    torques = sorted({r[0][0] for r in tab}); durs = sorted({r[0][1] for r in tab})
    m = np.full((len(durs), len(torques)), np.nan)
    for (tq, d), k, n, rate, _ in tab:
        m[durs.index(d), torques.index(tq)] = rate
    return torques, durs, m


def compare_boundary_world(old, new, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, e in zip(axes, (old, new)):
        torques, durs, m = _grid(e["world_tab"])
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1, origin="lower", aspect="auto")
        for i in range(len(durs)):
            for j in range(len(torques)):
                ax.text(j, i, f"{m[i, j]:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if m[i, j] > 0.6 else INK)
        ax.set_xticks(range(len(torques))); ax.set_xticklabels([f"{t:g}" for t in torques])
        ax.set_yticks(range(len(durs))); ax.set_yticklabels([f"{d * 0.05:g} s" for d in durs])
        ax.set_xlabel("push torque (N·m)")
        ax.set_title(LABEL.get(e["name"], e["name"]), loc="left", fontsize=11, color=INK)
    axes[0].set_ylabel("push duration")
    cb = fig.colorbar(im, ax=axes, fraction=0.025); cb.set_label("fraction correctly called \"world event\"")
    fig.suptitle(f"Push rejection, test set ({cc.N_BOUNDARY} trials per cell)", x=0.06, ha="left",
                 fontsize=12, color=INK)
    return _save(fig, path)


def compare_boundary_body(old, new, path):
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for e, ls, alpha in ((old, "--", 0.55), (new, "-", 1.0)):
        for param, color in (("l", C_BODY), ("m", "#4a3aa7")):
            rows = [r for r in e["body_tab"] if r[0][0] == param]
            ax.plot([r[0][1] for r in rows], [r[3] for r in rows], ls=ls, marker="o", ms=5,
                    lw=2, color=color, alpha=alpha,
                    label=f"{'length' if param == 'l' else 'mass'} · {LABEL.get(e['name'], e['name'])}")
    ax.axhline(cc.RELIABLE_RATE, color=MUTED, ls=":", lw=1.2)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("body-change multiplier")
    ax.set_ylabel("fraction called \"body change\" (detected)")
    ax.set_title(f"Body-change detection, test set ({cc.N_BOUNDARY} trials per point)",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _style(ax)
    return _save(fig, path)


def compare_roc(roc_old, roc_new, name_old, name_new, path):
    from metrics import auc
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1, label="chance")
    for rows, name, color, ls in ((roc_old, name_old, C_OLD, "--"), (roc_new, name_new, C_NEW, "-")):
        ax.step([r[1] for r in rows], [r[2] for r in rows], where="post", color=color, ls=ls, lw=2,
                label=f"{LABEL.get(name, name)} (AUC {auc(rows):.3f})")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("false-positive rate (push called \"body change\")  ← safety-critical")
    ax.set_ylabel("true-positive rate (body change detected)")
    ax.set_title("ROC on the hard boundary trials, test set", loc="left", fontsize=11, color=INK)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    _style(ax)
    return _save(fig, path)


def delay_hist(e, path):
    d_body = [r["delay_s"] for r in e["scored"] if r["block"] != "calib" and r["true_class"] == BODY]
    d_world = [r["delay_s"] for r in e["scored"] if r["block"] != "calib" and r["true_class"] == WORLD]
    bins = np.arange(cc.MIN_LAG * 0.05 + cc.CALM_STEPS * 0.05 - 0.05, cc.MAX_LAG * 0.05 + 0.15, 0.1)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.hist([d_body, d_world], bins=bins, color=[C_BODY, C_WORLD], label=["body change", "world event"],
            stacked=False, edgecolor="white")
    ax.set_xlabel("time from onset to verdict (s)")
    ax.set_ylabel("trials")
    ax.set_title(f"How long \"{LABEL.get(e['name'], e['name'])}\" waits, test set (all disturbed trials)",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    return _save(fig, path)
