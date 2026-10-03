"""Figures for the 3-way classifier.

  dev_alarm_sweep.png        dev: false alarms and severe misses vs alarm threshold
  confusion_3x3.png          test: previous vs new rule, 3x3 confusion matrices
  outcomes_by_category.png   test: verdict mix per trial category, previous vs new
  error_over_time.png        test: why time works. 1 s block error over 10 s for
                             severe damage vs big pushes, with the alarm threshold
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import alarm_classifier as A
import alarm_config as ac
from plot_classifier import C_BODY, C_WORLD, INK, MUTED, _save, _style

C_ALARM = "#e34948"   # red: reserved here for the alarm verdict
C_SEVERE = "#4a3aa7"
TRUTHS = ("body", "world", "severe")
PREDS = (A.BODY, A.WORLD, A.ALARM)


def dev_sweep(rows, chosen, path):
    t = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(t, [r["false_alarms"] for r in rows], color=C_WORLD, lw=2, marker="o", ms=3,
            label="false alarms (world events + undisturbed)")
    ax.plot(t, [r["severe_missed"] for r in rows], color=C_SEVERE, lw=2, ls="--", marker="o", ms=3,
            label="severe damage missed (called world)")
    ax.plot(t, [r["severe_alarm"] for r in rows], color=C_ALARM, lw=1.5, ls=":",
            label="severe damage -> alarm")
    ax.axvline(chosen, color=INK, lw=1.2, ls="--")
    ax.text(chosen, ax.get_ylim()[1] * 0.95, f"  chosen {chosen:.3g}", color=INK, fontsize=9, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("alarm threshold on 1 s block-mean error (log)")
    ax.set_ylabel("dev trials")
    ax.set_title("Dev: choosing the alarm threshold", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    return _save(fig, path)


def confusions(m_prev, m_new, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    vmax = max(max(m_prev.values()), max(m_new.values()))
    for ax, m, title in ((axes[0], m_prev, "previous: settle ≤5 s, undecided → world"),
                         (axes[1], m_new, "new: settle ≤10 s, else persistence → ALARM")):
        a = np.array([[m[(t, p)] for p in PREDS] for t in TRUTHS])
        ax.imshow(a, cmap="Blues", vmin=0, vmax=vmax)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, a[i, j], ha="center", va="center", fontsize=11,
                        color="white" if a[i, j] > 0.6 * vmax else INK)
        ax.set_xticks(range(3)); ax.set_xticklabels([f"verdict:\n{p}" for p in PREDS])
        ax.set_yticks(range(3)); ax.set_yticklabels([f"true {t}" for t in TRUTHS])
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Test set, same episodes for both rules", x=0.04, ha="left", fontsize=12, color=INK)
    return _save(fig, path)


def category_outcomes(res, prev, new, path):
    cats = ("body", "world", "severe", "bigpush", "calib")
    names = {"body": "moderate body\n×1.2–1.5", "world": "ordinary push", "severe": "severe body\n×1.5–1.7",
             "bigpush": "big long push\n2–4 N·m, 0.5–1 s", "calib": "undisturbed"}
    colors = {A.BODY: C_BODY, A.WORLD: C_WORLD, A.ALARM: C_ALARM}
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(cats)); w = 0.38
    for off, preds, label in ((-w / 2 - 0.01, prev, "previous"), (w / 2 + 0.01, new, "new")):
        bottom = np.zeros(len(cats))
        for p in PREDS:
            vals = []
            for c in cats:
                v = [q for r, (q, _, _) in zip(res, preds) if r["category"] == c]
                vals.append(v.count(p) / len(v))
            ax.bar(x + off, vals, w, bottom=bottom, color=colors[p], edgecolor="white", lw=1.5,
                   label=f"verdict: {p}" if label == "new" else None)
            bottom += vals
        for xi in x + off:
            ax.text(xi, 1.02, label, ha="center", fontsize=8.5, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels([names[c] for c in cats])
    ax.set_ylim(0, 1.1); ax.set_ylabel("fraction of trials")
    ax.set_title("Verdicts by trial category, test set (left bar previous, right bar new)",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _style(ax)
    return _save(fig, path)


def error_over_time(res, alarm_thr, path):
    """1 s block-mean error after onset, median and 10-90% band per category."""
    blocks = ac.EXT_MAX_LAG // ac.ALARM_BLOCK
    t = (np.arange(blocks) + 0.5) * ac.ALARM_BLOCK * 0.05
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for cat, color, label in (("severe", C_SEVERE, "severe body change ×1.5–1.7"),
                              ("bigpush", C_WORLD, "big long push 2–4 N·m, 0.5–1 s"),
                              ("body", C_BODY, "moderate body change ×1.2–1.5"),
                              ("calib", "#2a78d6", "undisturbed")):
        e = np.array([[r["err"][r["onset"] + k * ac.ALARM_BLOCK: r["onset"] + (k + 1) * ac.ALARM_BLOCK].mean()
                       for k in range(blocks)] for r in res if r["category"] == cat])
        ax.fill_between(t, np.percentile(e, 10, axis=0), np.percentile(e, 90, axis=0), color=color, alpha=0.15, lw=0)
        ax.plot(t, np.median(e, axis=0), color=color, lw=2, label=label)
    ax.axhline(alarm_thr, color=C_ALARM, ls="--", lw=1.4)
    ax.text(t[-1], alarm_thr, f"alarm threshold {alarm_thr:.3g} ", color=C_ALARM, fontsize=9,
            ha="right", va="bottom")
    ax.axvspan(ac.ALARM_FROM * 0.05, ac.EXT_MAX_LAG * 0.05, color=MUTED, alpha=0.08, lw=0)
    ax.text(ac.ALARM_FROM * 0.05, ax.get_ylim()[1] if False else 0.6, "  persistence window", color=MUTED,
            fontsize=9, va="top")
    ax.set_yscale("log"); ax.set_ylim(0.005, 0.8)
    ax.set_xlabel("time after onset (s)")
    ax.set_ylabel("1 s block-mean prediction error (log)")
    ax.set_title("Why time separates them: median and 10–90% band, test set", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style(ax)
    return _save(fig, path)
