"""Plot the three surprise signals on one figure.

    python plot.py      # re-plot from the CSVs in results/ without re-running

Two stacked panels, same time axis:
  top     the raw one-step prediction error, every step
  bottom  the same error smoothed with a trailing mean (SMOOTH_WINDOW steps).
          Trailing, not centred, so the smoothed curve never "knows" about a
          step before it happens - it is what an online detector would see.

Both panels use a LOG y-axis. The baseline error is small and the disturbed
errors can be 10-100x larger; on a linear axis the baseline would be a flat
line at zero and you could not judge whether the world-event error really
returns to the baseline level. On a log axis "back to baseline" is visible.

Annotations:
  dashed vertical line    the onset step (disturbance begins)
  orange band             the steps the external push is active
  grey band (bottom)      the "late" window used by the persistence feature
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")   # write files; no display needed
import matplotlib.pyplot as plt
import numpy as np

import config

HERE = os.path.dirname(os.path.abspath(__file__))

# Fixed colour per condition (colourblind-checked categorical palette, slots
# 1-3) plus a distinct line style, so identity never relies on colour alone.
STYLE = {
    "baseline":    dict(color="#2a78d6", ls="-",  label="Baseline (healthy, undisturbed)"),
    "body_change": dict(color="#eb6834", ls="-",  label="Body change (permanent)"),
    "world_event": dict(color="#1baf7a", ls="--", label="World event (brief push)"),
}
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"


def trailing_mean(x, w):
    x = np.asarray(x, dtype=float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def _describe_body():
    names = {"l": "length", "m": "mass", "g": "gravity"}
    return ", ".join(f"{names[k]} x{v:g}" for k, v in config.BODY_CHANGE.items())


def plot_surprise(series, path):
    """series: {condition_name: 1-D array of per-step errors}. Returns path."""
    dt = 0.05
    onset = config.ONSET_STEP
    push_end = onset + config.WORLD_DURATION
    late0 = onset + config.LATE_LAG
    late1 = late0 + config.LATE_WINDOW

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                             gridspec_kw=dict(hspace=0.12))
    panels = [(axes[0], "One-step prediction error (raw)", lambda e: e),
              (axes[1], f"Trailing mean over {config.SMOOTH_WINDOW} steps "
                        f"({config.SMOOTH_WINDOW * dt:g} s)",
               lambda e: trailing_mean(e, config.SMOOTH_WINDOW))]

    for ax, title, transform in panels:
        for name, err in series.items():
            st = STYLE.get(name, dict(color=INK, ls="-", label=name))
            t = np.arange(len(err)) * dt
            ax.plot(t, transform(err), color=st["color"], ls=st["ls"], lw=1.6,
                    label=st["label"])
        ax.set_yscale("log")
        ax.axvline(onset * dt, color=MUTED, ls=":", lw=1.2)
        ax.axvspan(onset * dt, push_end * dt, color="#1baf7a", alpha=0.12, lw=0)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_ylabel("‖actual − predicted‖", color=INK)
        ax.grid(True, which="major", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED)

    # annotate onset / push / late window once, on the relevant panel
    top, bot = axes
    ymax = top.get_ylim()[1]
    top.text(onset * dt, ymax, "onset  ", va="top", ha="right", color=MUTED, fontsize=9)
    top.text(push_end * dt, ymax, f"  push active ({config.WORLD_DURATION} steps)",
             va="top", ha="left", color=MUTED, fontsize=9, transform=top.transData)
    bot.axvspan(late0 * dt, late1 * dt, color=MUTED, alpha=0.10, lw=0)
    bot.text((late0 + late1) / 2 * dt, bot.get_ylim()[1], "late window\n(persistence test)",
             va="top", ha="center", color=MUTED, fontsize=9)

    # direct labels at the right end of the smoothed panel + a legend on top
    for name, err in series.items():
        st = STYLE.get(name, dict(color=INK))
        y = trailing_mean(err, config.SMOOTH_WINDOW)[-1]
        bot.annotate(name.replace("_", " "), xy=(len(err) * dt, y), xytext=(4, 0),
                     textcoords="offset points", va="center", fontsize=9, color=INK)
    top.legend(loc="upper left", bbox_to_anchor=(0, -0.02, 1, 1), frameon=False,
               fontsize=9, labelcolor=INK)

    bot.set_xlabel("time (s)   [1 step = 0.05 s]", color=INK)
    fig.suptitle("Surprise signatures: body change vs world event",
                 x=0.07, ha="left", fontsize=13, color=INK)
    fig.text(0.07, 0.925,
             f"Frozen forward model, CEM-MPC balancing Pendulum-v1. At t = {onset * dt:g} s: "
             f"body change = {_describe_body()} (permanent); "
             f"world event = {config.WORLD_TORQUE:g} N·m external torque for "
             f"{config.WORLD_DURATION * dt:g} s.",
             fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.07, right=0.9, top=0.88, bottom=0.08)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def load_series(out_dir):
    """Read results/<condition>.csv back into {name: error array}."""
    series = {}
    for name in STYLE:
        p = os.path.join(out_dir, f"{name}.csv")
        if os.path.exists(p):
            with open(p) as f:
                series[name] = np.array([float(r["surprise"]) for r in csv.DictReader(f)])
    return series


if __name__ == "__main__":
    out_dir = os.path.join(HERE, config.OUT_DIR)
    s = load_series(out_dir)
    if not s:
        raise SystemExit(f"no CSVs in {out_dir}; run run_experiment.py first")
    print(plot_surprise(s, os.path.join(out_dir, "surprise_signatures.png")))
