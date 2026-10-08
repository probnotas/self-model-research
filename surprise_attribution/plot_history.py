"""Single-state vs history-input model as the surprise signal, same trials.

  settled-score AUC vs noise (dashed = single-state, solid = K_MAIN history)
  settled-score AUC vs history length K (K=1 is the single-state model), every config
  false positive (world only -> body/alarm)      masking (body+world -> world)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import history_config as hc
from plot_classifier import INK, MUTED, _save, _style
from plot_compliant import COLORS, NOISES

NOISE_LS = {"none": "-", "low": "--", "medium": ":"}


def sweep(results, path):
    km = f"k{hc.K_MAIN}"
    by = {}
    for r in results:
        s, n = r["config"].rsplit("_", 1)
        by.setdefault(s, {})[n] = r
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    x = np.arange(len(NOISES))
    panels = ((axes[0, 0], "settled_auc", f"settled-score AUC: single-state (dashed) vs {hc.K_MAIN}-step history (solid)"),
              (axes[1, 0], "false_positive", "false positive: push alone → body/alarm"),
              (axes[1, 1], "masking", "masking: body+push → world"))
    for ax, key, title in panels:
        for s, rs in by.items():
            have = [i for i, n in enumerate(NOISES) if n in rs]
            for tag, ls, mk, dx in (("old", "--", "o", -0.04), (km, "-", "s", 0.04)):
                y = [rs[NOISES[i]][tag][key] if key == "settled_auc" else rs[NOISES[i]][tag][key]["rate"]
                     for i in have]
                ax.plot(x[have] + dx, y, color=COLORS[s], ls=ls, marker=mk, ms=5, lw=1.8,
                        alpha=1.0 if tag == km else 0.55,
                        label=f"{s} {'single' if tag == 'old' else f'k={hc.K_MAIN}'}")
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_ylim(-0.03, 1.03)
        if key == "settled_auc":
            ax.axhline(0.5, color=MUTED, ls=":", lw=1)
        ax.set_xticks(x); ax.set_xticklabels([f"noise: {n}" for n in NOISES])
        _style(ax)
    axes[1, 0].legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")

    ax = axes[0, 1]
    ks = [1] + list(hc.K_SWEEP)
    for r in results:
        s, n = r["config"].rsplit("_", 1)
        y = [r["old"]["settled_auc"]] + [r[f"k{k}"]["settled_auc"] for k in hc.K_SWEEP]
        ax.plot(ks, y, color=COLORS[s], ls=NOISE_LS[n], marker="o", ms=4, lw=1.6, label=f"{s}/{n}")
    ax.axhline(0.5, color=MUTED, ls=":", lw=1)
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels(["1\n(single)"] + [str(k) for k in hc.K_SWEEP])
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("settled-score AUC vs history length K (line style = noise)", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower right")
    _style(ax)
    fig.suptitle("History-input surprise signal vs single-state, same trials. SIMULATED compliant body, not hardware",
                 x=0.02, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, path)
