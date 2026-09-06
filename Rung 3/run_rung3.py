"""Rung 3 — ensembles, uncertainty-aware planning, and GrBAL-style meta-learning.

Rung 2 found that a single stale forward model plans worse on a damaged body
than a model-free reflex policy, because model error compounds over the 20-step
planning horizon. This runs the two standard fixes and separates their
contributions.

  Stage 1  ensemble of 5 (bootstrap + different inits); CEM scores the task cost
           against the ensemble MEAN and penalises member DISAGREEMENT with
           weight lambda. lambda=0 is the control that isolates "ensemble alone"
           from "uncertainty penalty".

  Stage 2  meta-train across a distribution of bodies (mass, length, damping,
           torque scale) with a MAML inner/outer loop, so that adapting on K
           recent transitions is maximally informative; then adapt online on a
           sliding window before every decision.

Experimental controls worth knowing about:
  * Conditions (b), (c), (d) share ONE fixed post-damage dataset, so the only
    thing differing between them is the model and the planner.
  * lambda and K are selected on a VALIDATION set of start states and reported
    on a disjoint TEST set, so the headline numbers are not selected on
    themselves.
  * The meta-training ranges are deliberately narrower than every test body, so
    all evaluated damages are extrapolations rather than interpolations.
"""
import json
import pathlib
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "Rung 1"))
import sample_effeciency as se  # noqa: E402

import bodies, models, planners  # noqa: E402

torch.set_num_threads(2)

# ---------------------------------------------------------------- config
PRE_SAMPLES   = 2000          # healthy-body data for the non-meta models
E             = 5
LAMBDAS       = [0.0, 0.1, 1.0, 10.0]
K_VALUES      = [10, 25]
ADAPT_POINTS  = [5, 10, 25, 50, 100]
MAIN_FT       = 100           # post-damage samples for conditions (b)/(c)/(d)
FT_EPOCHS     = 1000
N_TASKS       = 200
TASK_LEN      = 300
META_ITERS    = 15000
META_ALPHA    = 0.01
SEED          = 0
HELDOUT       = 1000

TEST_SEEDS = se.get_eval_seeds()                                   # the 15 used in Rungs 1-2
VAL_SEEDS  = np.random.default_rng(9999).integers(0, 2**31 - 1, 15)  # disjoint, for selection


def banner(t):
    print("\n" + "=" * 76 + f"\n{t}\n" + "=" * 76, flush=True)


def ev(params, lam=0.0, body="mass", adaptive=False, K=10, seeds=None):
    """Evaluate one controller configuration and return (mean, std, runs)."""
    seeds = TEST_SEEDS if seeds is None else seeds
    ctrl = (planners.adaptive_controller(params, lam=lam, K=K, alpha=META_ALPHA)
            if adaptive else planners.static_controller(params, lam=lam))
    return planners.evaluate(bodies.body_fn(body), seeds, ctrl)


def main():
    t0 = time.time()
    R = {}

    banner("RUNG 3 — SETUP")
    w = bodies.verify_equivalence()
    assert w == 0.0, f"ParamPendulum diverges from Pendulum-v1 (max {w:.2e})"
    print(f"body equivalence vs Pendulum-v1: exact (max diff {w:.1e})")
    print(f"meta-training ranges: {bodies.META_RANGES}")
    print(f"test bodies: {bodies.BODIES}")
    print("all test damages lie OUTSIDE the meta-training ranges (extrapolation)")

    # ---------------- data ----------------
    healthy = bodies.body_fn("healthy")
    damaged = bodies.body_fn("mass")
    Sh, Ah, S2h = bodies.collect(healthy, PRE_SAMPLES, seed=SEED)
    hoS, hoA, hoS2 = bodies.collect(damaged, HELDOUT, seed=SEED + 4242)
    postS, postA, postS2 = bodies.collect(damaged, max(ADAPT_POINTS), seed=SEED + 500)
    print(f"\nhealthy pool {PRE_SAMPLES} · post-damage stream {max(ADAPT_POINTS)} "
          f"· damaged held-out {HELDOUT}")

    # ---------------- stage 0: single + ensemble on healthy data ----------------
    banner("STAGE 0 — baseline models on the healthy body")
    p_single = models.train_ensemble(Sh, Ah, S2h, E=1, epochs=1000, seed=SEED,
                                     bootstrap=False)
    p_ens = models.train_ensemble(Sh, Ah, S2h, E=E, epochs=1000, seed=SEED,
                                  bootstrap=True)
    print(f"single  : damaged-body held-out MSE {models.prediction_mse(p_single, hoS, hoA, hoS2):.6f}")
    print(f"ensemble: damaged-body held-out MSE {models.prediction_mse(p_ens, hoS, hoA, hoS2):.6f}")

    def finetune(params, n, Emembers):
        return models.train_ensemble(postS[:n], postA[:n], postS2[:n], E=Emembers,
                                     epochs=FT_EPOCHS, seed=SEED, bootstrap=Emembers > 1,
                                     params=params)

    p_single_ft = finetune(p_single, MAIN_FT, 1)
    p_ens_ft = finetune(p_ens, MAIN_FT, E)
    print(f"\nafter fine-tuning on {MAIN_FT} post-damage samples:")
    print(f"single  : {models.prediction_mse(p_single_ft, hoS, hoA, hoS2):.6f}")
    print(f"ensemble: {models.prediction_mse(p_ens_ft, hoS, hoA, hoS2):.6f}")

    # ---------------- stage 2 training: meta-learning ----------------
    banner("STAGE 2 — meta-training across a distribution of bodies")
    rng = np.random.default_rng(SEED)
    tasks = [bodies.collect(lambda p=bodies.sample_task(rng): bodies.make_body(p),
                            TASK_LEN, seed=SEED + 1000 + i) for i in range(N_TASKS)]
    print(f"{N_TASKS} tasks x {TASK_LEN} transitions collected")

    meta = {}
    for K in K_VALUES:
        t = time.time()
        pk, _ = models.meta_train(tasks, E=E, K=K, iters=META_ITERS,
                                  alpha=META_ALPHA, seed=SEED, second_order=True)
        meta[K] = pk
        print(f"K={K:>2}: meta-trained E={E}, {META_ITERS} iters, second-order "
              f"[{time.time()-t:.0f}s]")

    def member0(p):
        return [t[:1].clone() for t in p]

    # ---------------- selection on VALIDATION seeds ----------------
    banner("SELECTION (validation start states — disjoint from the reported set)")

    print("\nlambda sweep, fine-tuned ensemble:")
    lam_val = {}
    for lam in LAMBDAS:
        m, s, _ = ev(p_ens_ft, lam=lam, seeds=VAL_SEEDS)
        lam_val[lam] = m
        print(f"   lambda={lam:<5} -> {m:.3f} +/- {s:.3f}")
    best_lam = min(lam_val, key=lam_val.get)
    print(f"   selected lambda = {best_lam}")

    print("\nK sweep, meta-learned ensemble (lambda=0):")
    k_val = {}
    for K in K_VALUES:
        m, s, _ = ev(meta[K], lam=0.0, adaptive=True, K=K, seeds=VAL_SEEDS)
        k_val[K] = m
        print(f"   K={K:<3} -> {m:.3f} +/- {s:.3f}")
    best_K = min(k_val, key=k_val.get)
    print(f"   selected K = {best_K}")

    print("\nlambda sweep, meta-learned ensemble (K={}):".format(best_K))
    lam_meta_val = {}
    for lam in LAMBDAS:
        m, s, _ = ev(meta[best_K], lam=lam, adaptive=True, K=best_K, seeds=VAL_SEEDS)
        lam_meta_val[lam] = m
        print(f"   lambda={lam:<5} -> {m:.3f} +/- {s:.3f}")
    best_lam_meta = min(lam_meta_val, key=lam_meta_val.get)
    print(f"   selected lambda = {best_lam_meta}")

    R["selection"] = {"lambda_ensemble": lam_val, "K": k_val,
                      "lambda_meta": lam_meta_val, "best_lambda": best_lam,
                      "best_K": best_K, "best_lambda_meta": best_lam_meta}

    # ---------------- main table on TEST seeds ----------------
    banner("MAIN RESULTS — damaged body (mass x1.5), 15 evaluation runs")
    conds = {}
    conds["a"] = ("single model + CEM (stale, no adaptation)", ev(p_single))
    conds["b"] = (f"single + naive fine-tune ({MAIN_FT}) + CEM", ev(p_single_ft))
    conds["c"] = (f"ensemble + CEM, lambda=0", ev(p_ens_ft, lam=0.0))
    conds["d"] = (f"ensemble + uncertainty CEM, lambda={best_lam}",
                  ev(p_ens_ft, lam=best_lam))
    conds["e"] = (f"meta + online adapt (K={best_K}) + CEM",
                  ev(member0(meta[best_K]), lam=0.0, adaptive=True, K=best_K))
    conds["f"] = (f"meta + online adapt + uncertainty CEM, lambda={best_lam_meta}",
                  ev(meta[best_K], lam=best_lam_meta, adaptive=True, K=best_K))

    # (g) model-free reflex policy, from the Rung 2 run (same eval seeds, same body)
    sac_path = pathlib.Path(__file__).resolve().parent.parent / "Rung 2" / "results_sac_adapt.json"
    if sac_path.exists():
        sac = json.load(open(sac_path))
        conds["g"] = ("model-free SAC pre-trained on healthy (from Rung 2)",
                      (sac["pre_damaged_stale"], float("nan"), None))
    print(f"\n{'':>3} {'condition':<52} {'cost':>8} {'std':>8}")
    print("-" * 76)
    for k in "abcdefg":
        if k not in conds:
            continue
        name, (m, s, _) = conds[k]
        ss = "     n/a" if np.isnan(s) else f"{s:>8.3f}"
        print(f"({k}) {name:<52} {m:>8.3f} {ss}")
    print("-" * 76)
    R["main"] = {k: {"name": v[0], "mean": v[1][0], "std": v[1][1]}
                 for k, v in conds.items()}

    base = conds["b"][1][0]
    print(f"\nrelative to (b), the baseline to beat ({base:.3f}):")
    for k in "acdef":
        if k in conds:
            m = conds[k][1][0]
            print(f"   ({k}) {m:>7.3f}   {100*(base-m)/base:+6.1f}%")

    # ---------------- adaptation curves ----------------
    banner("ADAPTATION CURVES — control cost vs post-damage transitions")
    curves = {"b": [], "e": [], "f": []}
    for n in ADAPT_POINTS:
        pb = finetune(p_single, n, 1)
        curves["b"].append(ev(pb)[:2])
        curves["e"].append(ev(member0(meta[best_K]), lam=0.0, adaptive=True,
                              K=min(best_K, n))[:2])
        curves["f"].append(ev(meta[best_K], lam=best_lam_meta, adaptive=True,
                              K=min(best_K, n))[:2])
        print(f"   n={n:>4}   (b) {curves['b'][-1][0]:.3f}   "
              f"(e) {curves['e'][-1][0]:.3f}   (f) {curves['f'][-1][0]:.3f}", flush=True)

    print(f"\n{'n':>6} {'(b) naive FT':>16} {'(e) meta':>16} {'(f) meta+unc':>16}")
    print("-" * 60)
    for i, n in enumerate(ADAPT_POINTS):
        print(f"{n:>6} {curves['b'][i][0]:>10.3f} +/-{curves['b'][i][1]:<4.2f} "
              f"{curves['e'][i][0]:>10.3f} +/-{curves['e'][i][1]:<4.2f} "
              f"{curves['f'][i][0]:>10.3f} +/-{curves['f'][i][1]:<4.2f}")
    print("-" * 60)
    print("note: (e)/(f) adapt online on a sliding window, so n is the window")
    print("      budget, not a training set — they never fine-tune offline.")
    R["adaptation"] = {k: [[float(a), float(b)] for a, b in v] for k, v in curves.items()}

    # ---------------- generalization ----------------
    banner("GENERALIZATION — damage types beyond the meta-training ranges")
    gen = {}
    for body in ("length", "damping"):
        pb = models.train_ensemble(
            *bodies.collect(bodies.body_fn(body), MAIN_FT, seed=SEED + 700),
            E=1, epochs=FT_EPOCHS, seed=SEED, bootstrap=False, params=p_single)
        gen[body] = {
            "b": ev(pb, body=body)[:2],
            "f": ev(meta[best_K], lam=best_lam_meta, adaptive=True,
                    K=best_K, body=body)[:2],
            "a": ev(p_single, body=body)[:2],
        }
        print(f"   {body}: stale {gen[body]['a'][0]:.3f}   "
              f"naive FT {gen[body]['b'][0]:.3f}   meta+unc {gen[body]['f'][0]:.3f}",
              flush=True)

    print(f"\n{'body':>10} {'(a) stale':>14} {'(b) naive FT':>16} {'(f) meta+unc':>16}")
    print("-" * 60)
    for body, d in gen.items():
        print(f"{body:>10} {d['a'][0]:>9.3f}+/-{d['a'][1]:<4.2f} "
              f"{d['b'][0]:>10.3f}+/-{d['b'][1]:<4.2f} "
              f"{d['f'][0]:>10.3f}+/-{d['f'][1]:<4.2f}")
    print("-" * 60)
    R["generalization"] = {b: {k: [float(x[0]), float(x[1])] for k, x in d.items()}
                           for b, d in gen.items()}

    # ---------------- plots ----------------
    lbl = {k: f"({k})" for k in "abcdefg"}
    ks = [k for k in "abcdefg" if k in conds]
    vals = [conds[k][1][0] for k in ks]
    errs = [0 if np.isnan(conds[k][1][1]) else conds[k][1][1] / np.sqrt(15) * 1.96
            for k in ks]
    plt.figure(figsize=(9, 5))
    cols = ["#CE7A1B" if k in "ab" else "#3E7BC8" if k in "cd"
            else "#38A08A" if k in "ef" else "#7C7C7C" for k in ks]
    plt.bar([lbl[k] for k in ks], vals, yerr=errs, capsize=4, color=cols)
    for i, (k, v) in enumerate(zip(ks, vals)):
        plt.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=9)
    plt.axhline(base, color="#CE7A1B", ls=":", lw=1.4, label=f"(b) baseline {base:.2f}")
    plt.ylabel("control cost (lower = better)")
    plt.title("Rung 3 — damaged body (mass x1.5), all conditions")
    plt.legend(fontsize=8); plt.grid(axis="y", alpha=.3); plt.tight_layout()
    plt.savefig("rung3_conditions.png", dpi=150)

    plt.figure(figsize=(7.6, 5))
    for k, c, nm in (("b", "#CE7A1B", "(b) naive fine-tuning"),
                     ("e", "#3E7BC8", "(e) meta + online adapt"),
                     ("f", "#38A08A", "(f) meta + online + uncertainty")):
        m = [x[0] for x in curves[k]]
        s = [x[1] / np.sqrt(15) * 1.96 for x in curves[k]]
        plt.errorbar(ADAPT_POINTS, m, yerr=s, marker="o", capsize=4, color=c, label=nm)
    plt.xscale("log"); plt.xlabel("post-damage transitions")
    plt.ylabel("control cost (lower = better)")
    plt.title("Adaptation: naive fine-tuning vs meta-learned adaptation")
    plt.grid(True, which="both", alpha=.3); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig("rung3_adaptation.png", dpi=150)

    plt.figure(figsize=(7.2, 5))
    xs = np.arange(len(gen)); wdt = 0.26
    for i, (k, c, nm) in enumerate((("a", "#7C7C7C", "(a) stale"),
                                    ("b", "#CE7A1B", "(b) naive FT"),
                                    ("f", "#38A08A", "(f) meta + uncertainty"))):
        plt.bar(xs + (i - 1) * wdt, [gen[b][k][0] for b in gen], wdt,
                yerr=[gen[b][k][1] / np.sqrt(15) * 1.96 for b in gen],
                capsize=3, color=c, label=nm)
    plt.xticks(xs, [f"{b} (+50%)" if b == "length" else f"{b} 0.5" for b in gen])
    plt.ylabel("control cost (lower = better)")
    plt.title("Generalization to damages outside the meta-training range")
    plt.grid(axis="y", alpha=.3); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig("rung3_generalization.png", dpi=150)
    print("\nsaved rung3_conditions.png, rung3_adaptation.png, rung3_generalization.png")

    R["config"] = {"pre_samples": PRE_SAMPLES, "E": E, "lambdas": LAMBDAS,
                   "K_values": K_VALUES, "main_ft": MAIN_FT, "n_tasks": N_TASKS,
                   "task_len": TASK_LEN, "meta_iters": META_ITERS,
                   "meta_alpha": META_ALPHA, "meta_ranges": bodies.META_RANGES}
    with open("results_rung3.json", "w") as f:
        json.dump(R, f, indent=2, default=float)
    print("saved results_rung3.json")
    print(f"\nTotal runtime: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
