"""Drive the ACTUAL compliant pendulum open-loop with a sine torque, healthy vs changed body.

    python experiments/resonance-check-in-env/experiment.py

Reuses surprise_attribution/compliant_env.py unchanged: the same CompliantPendulum
class, the same "soft" stiffness (compliant_config.STIFFNESS["soft"] = 4 N*m/rad),
hysteresis, spring damper, motor inertia/friction, joint damping, Pendulum-v1's
m, l, g, dt, torque limit and speed limit. The body change uses the same hook as
the main experiments (env.change_body). Only the controller is replaced: instead of
the planner, the motor gets u(t) = A sin(omega t), A = max torque.

Sensor noise only touches observations, never the physics, so amplitudes are
measured on the true link angle (soft/"none" config).

Writes results.json, amplitude_vs_freq.png, ratio_vs_freq.png and RESULTS.md here.
SIMULATION ONLY.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SA = os.path.join(HERE, "..", "..", "surprise_attribution")
sys.path.insert(0, SA)
import compliant_config as kc  # noqa: E402
import compliant_env as ce  # noqa: E402

CFG = dict(stiffness=kc.STIFFNESS["soft"], noise=kc.NOISE["none"])
OMEGAS = [0.1, 0.3, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 8, 10]
SETTLE_S = 30.0                  # transient time (joint damping 0.3/s plus motor and spring damping)
WINDOW_S = 10.0                  # measurement window; stretched to one full period at slow drives
BODIES = {"healthy": {}, "mass x1.5": {"m": 1.5}, "length x1.35 (extra)": {"l": 1.35}}
START_SEEDS = range(5)           # robustness: random Pendulum-v1 start states
FINE = [round(x, 2) for x in np.arange(2.0, 6.001, 0.05)]   # fine grid around the peak (healthy and mass only)
TARGET = 0.0150 / 0.0501         # main experiment, soft body: median motion after / before a body change


def drive(omega, body, start_seed=None):
    """One open-loop run. start_seed None = hanging at rest (theta = pi). Returns amplitudes."""
    env = ce.make(CFG, max_steps=10 ** 6)
    env.reset(seed=0 if start_seed is None else start_seed)
    base = env.unwrapped
    if start_seed is None:
        base.state = np.array([np.pi, 0.0])
    if body:
        env.change_body(body)                    # same hook the main experiment uses
    A, dt = base.max_torque, base.dt
    window = max(WINDOW_S, 2 * np.pi / omega)
    n_settle, n_win = round(SETTLE_S / dt), round(window / dt)
    th, w = [], []
    for i in range(n_settle + n_win):
        env.step(np.array([A * np.sin(omega * i * dt)], dtype=np.float32))
        if i >= n_settle:
            th.append(base.state[0]); w.append(base.state[1])
    env.close()
    th, w = np.array(th), np.array(w)            # th is unwrapped in Pendulum-v1's internal state
    return dict(amp=float((th.max() - th.min()) / 2),          # peak-to-peak / 2 of the arm angle (rad)
                vel_amp=float((w.max() - w.min()) / 2),         # same for angular velocity (rad/s)
                center=float(np.mean(th) % (2 * np.pi)),        # pi = swinging about the bottom
                spins=bool(th.max() - th.min() > 2 * np.pi))     # went over the top: amplitude not meaningful


def main():
    res = {b: {} for b in BODIES}
    for b, body in BODIES.items():
        for om in OMEGAS:
            r = drive(om, body)
            amps = [drive(om, body, s)["amp"] for s in START_SEEDS]
            r["amp_random_starts"] = [min(amps), max(amps)]
            res[b][om] = r
            print(f"{b:22s} omega {om:5.1f}  amp {r['amp']:.4f} rad  vel amp {r['vel_amp']:.3f}  "
                  f"random starts {min(amps):.4f}-{max(amps):.4f}{'  SPINS' if r['spins'] else ''}", flush=True)

    H, M = res["healthy"], res["mass x1.5"]
    peak = {b: max(OMEGAS, key=lambda o: res[b][o]["amp"]) for b in BODIES}
    ratio = {b: {o: res[b][o]["amp"] / H[o]["amp"] for o in OMEGAS} for b in BODIES if b != "healthy"}
    valid = [o for o in OMEGAS if not (H[o]["spins"] or M[o]["spins"])]
    rmin_o = min(valid, key=lambda o: ratio["mass x1.5"][o])
    summary = dict(
        config=dict(stiffness=CFG["stiffness"], drive_amplitude=2.0, settle_s=SETTLE_S, window_s=WINDOW_S,
                    amplitude="peak-to-peak/2 of the true link angle over the window"),
        peak_omega=peak,
        peak_shift_mass=peak["mass x1.5"] / peak["healthy"],
        peak_shift_pure_spring_prediction=float(np.sqrt(1 / 1.5)),
        ratio_at_healthy_peak=ratio["mass x1.5"][peak["healthy"]],
        min_ratio_mass=dict(omega=rmin_o, ratio=ratio["mass x1.5"][rmin_o]),
        min_ratio_length_extra=min((ratio["length x1.35 (extra)"][o], o) for o in OMEGAS
                                   if not res["length x1.35 (extra)"][o]["spins"] and not H[o]["spins"]),
        spinning=[(b, o) for b in BODIES for o in OMEGAS if res[b][o]["spins"]],
        main_experiment_ratio=TARGET)
    fine = {b: [drive(o, BODIES[b])["amp"] for o in FINE] for b in ("healthy", "mass x1.5")}
    fr = [m / h for m, h in zip(fine["mass x1.5"], fine["healthy"])]
    summary["fine_grid"] = dict(omegas=[FINE[0], FINE[-1], 0.05],
                                peak_healthy=FINE[int(np.argmax(fine["healthy"]))],
                                peak_mass=FINE[int(np.argmax(fine["mass x1.5"]))],
                                min_ratio=dict(omega=FINE[int(np.argmin(fr))], ratio=float(min(fr))),
                                max_ratio=dict(omega=FINE[int(np.argmax(fr))], ratio=float(max(fr))))
    summary["swing_center_rad"] = dict(min=min(res[b][o]["center"] for b in BODIES for o in OMEGAS),
                                       max=max(res[b][o]["center"] for b in BODIES for o in OMEGAS))
    json.dump(dict(summary=summary, fine=dict(omegas=FINE, **fine), runs={b: {str(o): v for o, v in d.items()} for b, d in res.items()}),
              open(os.path.join(HERE, "results.json"), "w"), indent=1)
    plots(res, ratio)
    print(json.dumps(summary, indent=1))


def plots(res, ratio):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {"healthy": dict(color="#1b9e77", marker="o"), "mass x1.5": dict(color="#d95f02", marker="s"),
             "length x1.35 (extra)": dict(color="#7570b3", marker="^", ls="--", alpha=0.7)}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for b, d in res.items():
        ax.plot(OMEGAS, [d[o]["amp"] for o in OMEGAS], label=b, **style[b])
        sp = [o for o in OMEGAS if d[o]["spins"]]
        ax.scatter(sp, [d[o]["amp"] for o in sp], s=140, facecolors="none", edgecolors="red")
    ax.set_xlabel("drive frequency ω (rad/s)"); ax.set_ylabel("steady-state arm amplitude (rad)")
    ax.set_yscale("log"); ax.grid(alpha=0.3)
    ax.set_title("Soft compliant pendulum (k = 4), open-loop sine drive, A = 2 N·m\n(red ring = arm spun over the top)")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(HERE, "amplitude_vs_freq.png"), dpi=130)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for b, r in ratio.items():
        ax.plot(OMEGAS, [r[o] for o in OMEGAS], label=f"{b} / healthy", **style[b])
    ax.axhline(TARGET, color="k", ls=":", label=f"main experiment ×{TARGET:.2f}")
    ax.axhline(1, color="grey", lw=0.8)
    ax.set_xlabel("drive frequency ω (rad/s)"); ax.set_ylabel("amplitude ratio (changed / healthy)")
    ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("Where does the body change shrink the swing most?")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "ratio_vs_freq.png"), dpi=130)


if __name__ == "__main__":
    main()
