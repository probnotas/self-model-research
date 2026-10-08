"""AUC vs rollout horizon H, every config (soft highlighted), plus the divergence ratio.

  AUC body-only vs undisturbed vs H      AUC body-only vs push-only vs H
  healthy rollout error / 'nothing changes' predictor vs H (>1 = diverged)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rollout_config as rc
from plot_classifier import INK, MUTED, _save, _style
from plot_compliant import COLORS

NOISE_LS = {"none": "-", "low": "--", "medium": ":"}


def sweep(results, path):
    H = list(rc.H_SWEEP)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    panels = ((axes[0], lambda r, h: r["by_H"][h]["auc_body_vs_neither"], "AUC body-only vs UNDISTURBED (main test)"),
              (axes[1], lambda r, h: r["by_H"][h]["auc_body_vs_push"], "AUC body-only vs push-only (must stay > 0.5)"),
              (axes[2], lambda r, h: r["by_H"][h]["healthy_median"] / r["by_H"][h]["healthy_hold_median"],
               "healthy rollout error ÷ 'nothing changes' error (> 1: diverged)"))
    for ax, f, title in panels:
        for r in results:
            s, n = r["config"].rsplit("_", 1)
            soft = s == "soft"
            ax.plot(H, [f(r, h) for h in H], color=COLORS[s], ls=NOISE_LS[n], marker="o", ms=4 if soft else 3,
                    lw=2.6 if soft else 1.2, alpha=1.0 if soft else 0.55, label=f"{s}/{n}")
        ax.axhline(1.0 if "diverged" in title else 0.5, color=MUTED, ls=":", lw=1)
        ax.set_xscale("log"); ax.set_xticks(H); ax.set_xticklabels([f"{h}\n({h * 0.05:g}s)" for h in H], fontsize=8)
        if "diverged" not in title:
            ax.set_ylim(-0.03, 1.03)
        else:
            ax.set_yscale("log")
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel("rollout horizon H (steps); H=1 is one-step error", fontsize=8.5)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
    fig.suptitle("Free-running rollout error vs horizon, same trials (thick = soft body). SIMULATED compliant body, not hardware",
                 x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, path)
