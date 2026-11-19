"""Score main_runs.json against PREREG.md and write results.json, success_rates.png and RESULTS_numbers.md.

    python experiments/rung-8-layer/analyze.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CONDS = ("A_policy", "B_filter", "C_adapt", "D_oracle")
PERTS = ("healthy", "payload_add", "spring_soft", "spring_stiff", "damping_high", "motor_weak", "gravity_tilt", "sensor_delay")


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [100 * max(0, c - h), 100 * min(1, c + h)]


def main():
    D = json.load(open(os.path.join(HERE, "main_runs.json")))
    runs = D["runs"]
    cell = lambda c, p: [r for r in runs if r["cond"] == c and r["pert"] == p]
    rate = {c: {p: 100 * np.mean([r["success"] for r in cell(c, p)]) for p in PERTS} for c in CONDS}
    n = {c: {p: len(cell(c, p)) for p in PERTS} for c in CONDS}
    ci = {c: {p: wilson(sum(r["success"] for r in cell(c, p)), n[c][p]) for p in PERTS} for c in CONDS}
    fell = {c: {p: int(sum(r["fell"] for r in cell(c, p))) for p in PERTS} for c in CONDS}
    logs = {p: dict(B_reject_rate_mean=float(np.mean([r["filter_reject_rate"] for r in cell("B_filter", p)])),
                    C_adapted_fraction_mean=float(np.mean([r["adapt_fraction"] for r in cell("C_adapt", p)])),
                    C_mean_scale=float(np.nanmean([r["adapt_mean_scale"] if r["adapt_mean_scale"] is not None else np.nan
                                                   for r in cell("C_adapt", p)]))
                    if any(r["adapt_mean_scale"] is not None for r in cell("C_adapt", p)) else None,
                    mean_abs_u_post={c: float(np.mean([r["mean_abs_u_post"] for r in cell(c, p)])) for c in CONDS})
            for p in PERTS}
    A, B, C, Dd = (rate[c] for c in CONDS)
    claims = {
        "1_policy_degrades_>=30pp": dict(value=A["healthy"] - A["payload_add"], passed=bool(A["healthy"] - A["payload_add"] >= 30)),
        "2_oracle_degrades_<=10pp": dict(value=Dd["healthy"] - Dd["payload_add"], passed=bool(Dd["healthy"] - Dd["payload_add"] <= 10)),
        "3_selfmodel_beats_policy_>=15pp": dict(value=max(B["payload_add"], C["payload_add"]) - A["payload_add"],
                                                 passed=bool(max(B["payload_add"], C["payload_add"]) - A["payload_add"] >= 15)),
        "4_adapt_beats_filter_somewhere": dict(value=[p for p in PERTS[1:] if C[p] > B[p]],
                                                passed=bool(any(C[p] > B[p] for p in PERTS[1:]))),
    }
    out = dict(threshold_B=D["threshold_B"], success_rate_pct=rate, wilson95=ci, n=n, fell=fell, logs=logs, claims=claims)
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float)

    L = ["| perturbation | A policy | B filter | C adapt | D oracle |", "|---|---|---|---|---|"]
    for p in PERTS:
        L.append(f"| {p} | " + " | ".join(f"{rate[c][p]:.0f}% ({ci[c][p][0]:.0f}–{ci[c][p][1]:.0f})" for c in CONDS) + " |")
    L += ["", "| perturbation | falls A / B / C / D | B reject rate | C steps adapted | C mean scale |", "|---|---|---|---|---|"]
    for p in PERTS:
        lg = logs[p]
        scale = "n/a" if lg["C_mean_scale"] is None else f"{lg['C_mean_scale']:.2f}"
        L.append(f"| {p} | {fell['A_policy'][p]} / {fell['B_filter'][p]} / {fell['C_adapt'][p]} / {fell['D_oracle'][p]} | "
                 f"{lg['B_reject_rate_mean']:.1%} | {lg['C_adapted_fraction_mean']:.1%} | {scale} |")
    L += [""] + [f"- Claim {k}: value {v['value']}, **{'PASS' if v['passed'] else 'FAIL'}**" for k, v in claims.items()]
    open(os.path.join(HERE, "RESULTS_numbers.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(PERTS)); wdt = 0.2
    cols = {"A_policy": "#7f7f7f", "B_filter": "#1b9e77", "C_adapt": "#d95f02", "D_oracle": "#7570b3"}
    for i, c in enumerate(CONDS):
        ax.bar(x + (i - 1.5) * wdt, [rate[c][p] for p in PERTS], wdt, label=c, color=cols[c],
               yerr=np.array([[rate[c][p] - ci[c][p][0], ci[c][p][1] - rate[c][p]] for p in PERTS]).T, capsize=2)
    ax.set_xticks(x); ax.set_xticklabels(PERTS, rotation=25, ha="right"); ax.set_ylabel("success rate (%), Wilson 95%")
    ax.set_ylim(0, 105); ax.legend(ncol=4, fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Rung 8, in simulation: balance success under body changes (30 seeds per bar)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "success_rates.png"), dpi=130)


if __name__ == "__main__":
    main()
