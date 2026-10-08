"""Old (raw error) vs new (per-body z-score) on the same trials, per stiffness and noise.

Four panels sharing the noise axis; colour = stiffness, dashed = old, solid = new:
  settled-score AUC       false positive (world only -> body/alarm)
  masking (body+world -> world)       body change missed with NO push
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_classifier import INK, MUTED, _save, _style
from plot_compliant import COLORS, NOISES


def sweep(results, path):
    by = {}
    for r in results:
        s, n = r["config"].rsplit("_", 1)
        by.setdefault(s, {})[n] = r
    panels = (("settled_auc", "settled-score AUC (1 = clean, 0.5 = none, <0.5 inverted)"),
              ("false_positive", "false positive: push alone → body/alarm"),
              ("masking", "masking: body+push → world"),
              ("miss_body_only", "body change missed with NO push"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    x = np.arange(len(NOISES))
    for ax, (key, title) in zip(axes.flat, panels):
        for s, rs in by.items():
            have = [i for i, n in enumerate(NOISES) if n in rs]
            for tag, ls, mk in (("old", "--", "o"), ("new", "-", "s")):
                y = [rs[NOISES[i]][tag][key] if key == "settled_auc" else rs[NOISES[i]][tag][key]["rate"]
                     for i in have]
                ax.plot(x[have] + (0.04 if tag == "new" else -0.04), y, color=COLORS[s], ls=ls,
                        marker=mk, ms=5, lw=1.8, alpha=1.0 if tag == "new" else 0.6,
                        label=f"{s} {'new (z vs own baseline)' if tag == 'new' else 'old (raw)'}")
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.set_ylim(-0.03, 1.03)
        if key == "settled_auc":
            ax.axhline(0.5, color=MUTED, ls=":", lw=1)
            ax.text(0.02, 0.04, "old and new AUC are identical by construction:\n"
                    "a per-body z-score is monotone within a body",
                    transform=ax.transAxes, fontsize=8.5, color=MUTED)
        ax.set_xticks(x); ax.set_xticklabels([f"noise: {n}" for n in NOISES])
        _style(ax)
    axes[0, 1].legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    fig.suptitle("Per-body calibration vs raw error, same trials. SIMULATED compliant body, not hardware",
                 x=0.02, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, path)
