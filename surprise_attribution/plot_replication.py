"""Figure for the replication: motion z-score per fresh trial, by condition and body.

Reads results_replication/trials_<config>.csv (fresh seed 777000). One panel per
body; each dot is one trial; the black bar is the median; dashed lines are the
pre-registered thresholds z = +/-2. SIMULATED compliant body, not hardware.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = [("neither", "undisturbed", "#2a78d6"), ("body_only", "body change", "#eb6834"),
         ("world_only", "push", "#1baf7a"), ("body_world", "body change\n+ push", "#eda100")]
PANELS = [("soft_none", "soft joint, no noise"), ("soft_low", "soft joint, low noise"),
          ("rigid_none", "rigid (control)"), ("medium_none", "medium stiffness (control)")]


def main():
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
    rng = np.random.default_rng(0)
    for ax, (cfg, title) in zip(axes, PANELS):
        rows = list(csv.DictReader(open(os.path.join(HERE, "results_replication", f"trials_{cfg}.csv"))))
        for j, (cell, label, col) in enumerate(CELLS):
            z = np.array([float(r["motion_z"]) for r in rows if r["cell"] == cell])
            ax.scatter(j + rng.uniform(-0.22, 0.22, len(z)), z, s=9, color=col, alpha=0.55, lw=0)
            ax.plot([j - 0.3, j + 0.3], [np.median(z)] * 2, color="#222222", lw=2.2)
        ax.axhline(2, color="#888888", ls="--", lw=1); ax.axhline(-2, color="#888888", ls="--", lw=1)
        ax.axhline(0, color="#cccccc", lw=0.8)
        ax.set_xticks(range(4)); ax.set_xticklabels([c[1] for c in CELLS], fontsize=8.5)
        ax.set_title(title, loc="left", fontsize=10.5, color="#222222")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#eeeeee", lw=0.8); ax.set_axisbelow(True)
    axes[0].set_ylabel("change in motion vs healthy baseline (z)", fontsize=9.5)
    fig.suptitle("Fresh trials (seed 777000): after a body change the soft joint moves LESS; after a push it moves MORE. "
                 "Dashed: z = ±2. Simulated, not hardware.", x=0.01, ha="left", fontsize=10.5, color="#222222")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(HERE, "results_replication", "motion_z_by_condition.png")
    fig.savefig(out, dpi=160, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
