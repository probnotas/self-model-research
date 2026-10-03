"""Figures for the simultaneous-disturbance test.

  verdicts_by_category.png  verdict mix for each of the four categories (current rule)
  detection_vs_offset.png   body detection with vs without the push, and push-only
                            false positives, across the overlap offset
  masking_by_push.png       body+push detection rate over push torque x duration
  example_triplet.png       one triplet's error traces: body only / push only / both
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import alarm_classifier as A
import overlap_config as oc
from plot_classifier import C_BODY, C_WORLD, INK, MUTED, _save, _style

C_ALARM, C_BOTH, C_NONE = "#e34948", "#4a3aa7", "#2a78d6"
CATS = ("body_only", "push_only", "body_push", "undisturbed")
NAMES = {"body_only": "(1) body only", "push_only": "(2) push only",
         "body_push": "(3) body + push", "undisturbed": "(4) undisturbed"}


def category_verdicts(res, path):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(CATS)); bottom = np.zeros(len(CATS))
    for v, c in ((A.BODY, C_BODY), (A.WORLD, C_WORLD), (A.ALARM, C_ALARM)):
        vals = np.array([np.mean([r["verdict"] == v for r in res if r["category"] == cat]) for cat in CATS])
        ax.bar(x, vals, 0.6, bottom=bottom, color=c, edgecolor="white", lw=1.5, label=f"verdict: {v}")
        for xi, b, h in zip(x, bottom, vals):
            if h >= 0.04:
                ax.text(xi, b + h / 2, f"{h:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if v != A.WORLD else INK)
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels([NAMES[c] for c in CATS])
    ax.set_ylabel("fraction of trials"); ax.set_ylim(0, 1.02)
    ax.set_title("Verdicts per category, current rule, unchanged", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _style(ax)
    return _save(fig, path)


def offset_curve(fine, path):
    mid = [(f["lo"] + f["hi"]) / 2 for f in fine]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(mid, [f["det1"] for f in fine], color=C_BODY, ls="--", marker="o", ms=5, lw=2,
            label="body detected, without push (1)")
    ax.plot(mid, [f["det3"] for f in fine], color=C_BOTH, marker="o", ms=5, lw=2,
            label="body detected, with push (3)")
    ax.plot(mid, [f["fp"] / f["n"] for f in fine], color=C_ALARM, marker="s", ms=5, lw=1.5,
            label="push alone called body/alarm (2)")
    ax.axvspan(-0.25, 0.25, color=MUTED, alpha=0.12, lw=0)
    ax.text(0, 0.5, "push on\nbody onset", ha="center", fontsize=9, color=MUTED)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("overlap offset: push onset − body-change onset (s)   [negative = push first]")
    ax.set_ylabel("fraction of triplets in bin")
    ax.set_title("Does the timing of an overlapping push matter? (0.25 s bins, test set)",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _style(ax)
    return _save(fig, path)


def magnitude_grid(res, path):
    b3 = [r for r in res if r["category"] == "body_push"]
    tq, du = oc.TORQUE_BINS, oc.DURATION_BINS
    m = np.full((len(du) - 1, len(tq) - 1), np.nan); lab = [["" for _ in tq[:-1]] for _ in du[:-1]]
    for i in range(len(du) - 1):
        for j in range(len(tq) - 1):
            cell = [r for r in b3 if du[i] <= r["duration"] < du[i + 1] and tq[j] <= abs(r["torque"]) < tq[j + 1]]
            if cell:
                k = sum(r["detected"] for r in cell)
                m[i, j] = k / len(cell); lab[i][j] = f"{k}/{len(cell)}"
    fig, ax = plt.subplots(figsize=(8, 4.4))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1, origin="lower", aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, lab[i][j], ha="center", va="center", fontsize=9,
                    color="white" if (not np.isnan(m[i, j]) and m[i, j] > 0.6) else INK)
    ax.set_xticks(range(len(tq) - 1)); ax.set_xticklabels([f"{a:g}–{min(b, 4):g}" for a, b in zip(tq[:-1], tq[1:])])
    ax.set_yticks(range(len(du) - 1))
    ax.set_yticklabels([f"{a * 0.05:g}–{(b - 1) * 0.05:g} s" for a, b in zip(du[:-1], du[1:])])
    ax.set_xlabel("overlapping push |torque| (N·m)"); ax.set_ylabel("push duration")
    ax.set_title("Body change detected despite an overlapping push (3): detected/n",
                 loc="left", fontsize=11, color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04); cb.set_label("detection rate")
    return _save(fig, path)


def example_triplet(res, path):
    """A hard triplet: a masked one if any exist, else the biggest push near onset."""
    by = {}
    for r in res:
        if r["triplet"] >= 0:
            by.setdefault(r["triplet"], {})[r["category"]] = r
    masked = [t for t, m in by.items() if m["body_only"]["detected"] and not m["body_push"]["detected"]]
    if masked:
        tid, why = masked[0], "a MASKED case"
    else:
        near = [t for t, m in by.items() if abs(m["body_push"]["offset_s"]) <= 0.25]
        tid = max(near, key=lambda t: abs(by[t]["body_push"]["torque"]) * by[t]["body_push"]["duration"])
        why = "largest push landing on the body-change onset"
    m = by[tid]; r3 = m["body_push"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    t = np.arange(len(r3["err"])) * 0.05
    for cat, c, ls in (("push_only", C_WORLD, "--"), ("body_only", C_BODY, "-"), ("body_push", C_BOTH, "-")):
        r = m[cat]
        k = max(1, 10)
        sm = np.convolve(r["err"], np.ones(k) / k, mode="full")[:len(r["err"])]
        ax.plot(t, sm, color=c, ls=ls, lw=1.8, label=f"{NAMES[cat]} → {r['verdict']} ({r['delay_s']:.1f} s)")
    ax.axvline(r3["body_onset"] * 0.05, color=INK, ls=":", lw=1.2)
    ax.axvspan(r3["push_onset"] * 0.05, (r3["push_onset"] + r3["duration"]) * 0.05, color=C_WORLD, alpha=0.15, lw=0)
    ax.set_yscale("log"); ax.set_xlabel("time (s)"); ax.set_ylabel("prediction error, 0.5 s trailing mean (log)")
    ax.set_title(f"Example triplet ({why}): {r3['param']} ×{r3['factor']:.2f}, push {r3['torque']:+.1f} N·m "
                 f"for {r3['duration'] * 0.05:.2f} s, offset {r3['offset_s']:+.2f} s",
                 loc="left", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style(ax)
    return _save(fig, path)
