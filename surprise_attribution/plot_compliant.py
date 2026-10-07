"""Sweep figure: where does the rule break as stiffness drops and noise rises?

Four panels sharing the noise axis, one line per stiffness:
  false-positive rate (world only -> body/alarm)   masking rate (body+world -> world)
  body-only miss rate (no push at all)              settled-score separation (AUC)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_classifier import INK, MUTED, _save, _style

COLORS = {"rigid": "#2a78d6", "stiff": "#eb6834", "medium": "#1baf7a", "soft": "#4a3aa7"}
STYLE = {"rigid": "-", "stiff": "--", "medium": "-.", "soft": ":"}
NOISES = ("none", "low", "medium")


def sweep(results, path):
    by = {}
    for r in results:
        s, n = r["config"].rsplit("_", 1)
        by.setdefault(s, {})[n] = r
    panels = (("false_positive", "false positive: push alone → body/alarm  (safety-critical)"),
              ("masking", "masking: body+push → world"),
              ("miss_body_only", "body change missed with NO push"),
              ("settled_auc", "settled-score separation, AUC (1 = clean, 0.5 = none)"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    x = np.arange(len(NOISES))
    for ax, (key, title) in zip(axes.flat, panels):
        for s, rs in by.items():
            y = [rs[n][key] if key == "settled_auc" else rs[n][key]["rate"] for n in NOISES]
            ax.plot(x, y, color=COLORS[s], ls=STYLE[s], marker="o", ms=6, lw=2,
                    label=f"{s}" + (" (rigid Pendulum-v1)" if s == "rigid" else f" (k={rs['none']['stiffness']:g})"))
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.set_ylim((0.45, 1.02) if key == "settled_auc" else (-0.03, 1.03))
        if key == "settled_auc":
            ax.axhline(0.5, color=MUTED, ls=":", lw=1)
        ax.set_xticks(x); ax.set_xticklabels([f"noise: {n}" for n in NOISES])
        _style(ax)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Attribution rule on a SIMULATED compliant body (series-elastic + hysteresis + sensor noise)",
                 x=0.02, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, path)
