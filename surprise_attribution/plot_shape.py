"""Old settle rule vs shape rule, same trials, per stiffness and noise.

  AUC: old (settled trials only, dashed) vs shape log-ratio (all trials, solid)
  AUC of the shape feature vs window length W, every config
  detection of body changes: old (dashed), shape (solid), shape+elevation (dotted)
  false positive (push alone -> body): same line styles
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import shape_config as sc
from plot_classifier import INK, MUTED, _save, _style
from plot_compliant import COLORS, NOISES

NOISE_LS = {"none": "-", "low": "--", "medium": ":"}


def sweep(results, path):
    W = sc.W_MAIN
    by = {}
    for r in results:
        s, n = r["config"].rsplit("_", 1)
        by.setdefault(s, {})[n] = r
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    x = np.arange(len(NOISES))
    series = {
        "auc": [("old", lambda r: r["old"]["settled_auc"], "--", "o"),
                ("shape", lambda r: r["shape"][W]["auc_all"], "-", "s")],
        "detection": [("old", lambda r: r["old"]["detection"]["rate"], "--", "o"),
                      ("shape", lambda r: r["shape"][W]["detection"]["rate"], "-", "s"),
                      ("shape+elev", lambda r: r["shape_elev"][W]["detection"]["rate"], ":", "^")],
        "false_positive": [("old", lambda r: r["old"]["false_positive"]["rate"], "--", "o"),
                           ("shape", lambda r: r["shape"][W]["false_positive"]["rate"], "-", "s"),
                           ("shape+elev", lambda r: r["shape_elev"][W]["false_positive"]["rate"], ":", "^")],
    }
    titles = {"auc": f"AUC: old settle rule (dashed, settled trials only) vs shape log-ratio (solid, all trials), W={W}",
              "detection": "body changes detected (of 240): old -- , shape —, shape+elevation ···",
              "false_positive": "false positive: push alone → body: old -- , shape —, shape+elevation ···"}
    for ax, key in ((axes[0, 0], "auc"), (axes[1, 0], "detection"), (axes[1, 1], "false_positive")):
        for s, rs in by.items():
            have = [i for i, n in enumerate(NOISES) if n in rs]
            for j, (lab, f, ls, mk) in enumerate(series[key]):
                ax.plot(x[have] + (j - 1) * 0.04, [f(rs[NOISES[i]]) for i in have], color=COLORS[s], ls=ls,
                        marker=mk, ms=5, lw=1.8, alpha=0.55 if lab == "old" else 1.0, label=f"{s} {lab}")
        ax.set_title(titles[key], loc="left", fontsize=9, color=INK)
        ax.set_ylim(-0.03, 1.03)
        if key == "auc":
            ax.axhline(0.5, color=MUTED, ls=":", lw=1)
        ax.set_xticks(x); ax.set_xticklabels([f"noise: {n}" for n in NOISES])
        _style(ax)
    axes[0, 0].legend(frameon=False, fontsize=7.5, ncol=2, loc="lower left")

    ax = axes[0, 1]
    for r in results:
        s, n = r["config"].rsplit("_", 1)
        ax.plot(sc.W_SWEEP, [r["shape"][w]["auc_all"] for w in sc.W_SWEEP], color=COLORS[s], ls=NOISE_LS[n],
                marker="o", ms=4, lw=1.6, label=f"{s}/{n}")
    ax.axhline(0.5, color=MUTED, ls=":", lw=1)
    ax.set_xscale("log", base=2); ax.set_xticks(sc.W_SWEEP)
    ax.set_xticklabels([f"{w}\n({w * 0.05:g} s)" for w in sc.W_SWEEP])
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("shape-feature AUC (all trials) vs window length W", loc="left", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower right")
    _style(ax)
    fig.suptitle("Shape-based attribution (no settle gate) vs settle rule, same trials. SIMULATED compliant body, not hardware",
                 x=0.02, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, path)
