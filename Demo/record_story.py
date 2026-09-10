"""One continuous run: healthy -> damaged -> detected -> fallen -> repaired -> upright.

Three phases on a single timeline, one pendulum, no cuts:

  A  0-59     healthy body, healthy model            balances
  B  60-119   actuator drops to 50% mid-run          error spikes, alarm fires, it falls
  C  120-239  stood back up, model retrained on the  balances again
              first 25 transitions of phase B

The repair data is not collected separately -- it is literally the 25 moves the
pendulum made while falling over in phase B. The damage persists through phase C;
only the model changes.
"""
import copy, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Rung 1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Rung 3"))
import sample_effeciency as se
import bodies

PH_A = int(os.environ.get("PH_A", 60))
PH_B = int(os.environ.get("PH_B", 160))
STEPS = int(os.environ.get("STEPS", 260))
DAMAGE_SCALE = 0.5
REPAIR_N = int(os.environ.get("REPAIR_N", 100))
RETRY = int(os.environ.get("RETRY", 50))   # stand it back up every RETRY steps in phase B
START, WINDOW, SIGMA = 0.2, 10, 3.0
N_EP = 10


def finetune(model, S, A, S2, epochs=1000, lr=1e-3):
    m = copy.deepcopy(model)
    X = torch.tensor(np.concatenate([S, A], axis=1), dtype=torch.float32)
    Y = torch.tensor(S2 - S, dtype=torch.float32)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lf = torch.nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); lf(m(X), Y).backward(); opt.step()
    return m


def stand_up(env, rng):
    env.unwrapped.state = np.array([rng.uniform(-START, START), rng.uniform(-START, START)])
    return env.unwrapped._get_obs()


def story(model, seed):
    env = bodies.make_body({}, max_episode_steps=STEPS + 10)
    rng = np.random.default_rng(int(seed))
    env.reset(seed=int(seed))
    obs = stand_up(env, rng)
    np.random.seed(int(seed) % (2**31 - 1))

    th, td, tq, cs, er = [], [], [], [], []
    repair, active = [], model
    for t in range(STEPS):
        if t == PH_A:
            env.unwrapped.torque_scale = DAMAGE_SCALE          # the damage
        if PH_A < t < PH_B and (t - PH_A) % RETRY == 0:
            obs = stand_up(env, rng)                           # it tries again
        if t == PH_B:
            use = repair[:REPAIR_N]
            S, A, S2 = (np.array([r[i] for r in use]) for i in range(3))
            active = finetune(model, S, A, S2)                 # repair, from its own failure
            obs = stand_up(env, rng)                           # stand it back up, still damaged

        a = se.choose_action(active, obs)
        pred = se.predict(active, obs, a)
        th.append(float(np.arctan2(obs[1], obs[0]))); td.append(float(obs[2]))
        tq.append(float(a[0]))
        nxt, _, _, _, _ = env.step(a)
        er.append(float(np.mean((pred - nxt) ** 2)))
        cs.append(float(se.cost(nxt)))
        if PH_A <= t < PH_B:
            repair.append((obs, np.asarray(a, dtype=np.float64), nxt))
        obs = nxt
    env.close()
    return dict(theta=th, thdot=td, torque=tq, cost=cs, err=er)


def rolling(x, w):
    x = np.asarray(x, float)
    return np.array([x[max(0, i - w + 1):i + 1].mean() for i in range(len(x))])


if __name__ == "__main__":
    print("training the healthy model...", flush=True)
    S, A, S2 = se.collect_data(2000, seed=se.MASTER_SEED)
    model = se.train_model(S, A, S2, seed=0)
    seeds = se.get_eval_seeds(N_EP)

    eps = []
    for i, sd in enumerate(seeds):
        eps.append(story(model, sd))
        print(f"  run {i + 1}/{N_EP} done", flush=True)

    R = np.array([rolling(e["err"], WINDOW) for e in eps])
    ang = np.array([np.abs(e["theta"]) for e in eps])
    upA = (ang[:, PH_A - 20:PH_A].mean(axis=1) < 0.35)
    upC = (ang[:, -30:].mean(axis=1) < 0.35)
    # Phase B is a sequence of attempts, each ended by a stand-up, so "did it
    # fall" has to be scored at the END of each attempt -- not over a fixed
    # window, which lands just after a stand-up and reads as upright.
    ends = [PH_A + (k + 1) * RETRY - 1 for k in range((PH_B - PH_A) // RETRY)]
    fell = ang[:, ends] > 0.35                       # (runs, attempts)
    n_att = len(ends)

    lat, det, fa = [], 0, 0
    for i in range(N_EP):                                   # leave-one-out threshold
        pre = np.delete(R, i, axis=0)[:, WINDOW:PH_A]
        th_ = pre.mean() + SIGMA * pre.std()
        fa += int((R[i, WINDOW:PH_A] > th_).sum())
        w = np.where(R[i, PH_A:PH_B] > th_)[0]
        if len(w):
            det += 1; lat.append(int(w[0]) + 1)
    pre_all = R[:, WINDOW:PH_A]
    thresh = float(pre_all.mean() + SIGMA * pre_all.std())

    robust = {}
    for name, rule in [("mean + 3\u03c3", lambda p: p.mean() + 3 * p.std()),
                       ("mean + 4\u03c3", lambda p: p.mean() + 4 * p.std()),
                       ("max healthy value", lambda p: p.max())]:
        la, dt_, fa_ = [], 0, 0
        for i in range(N_EP):
            t_ = rule(np.delete(R, i, axis=0)[:, WINDOW:PH_A])
            fa_ += int((R[i, WINDOW:PH_A] > t_).sum())
            w = np.where(R[i, PH_A:PH_B] > t_)[0]
            if len(w):
                dt_ += 1; la.append(int(w[0]) + 1)
        robust[name] = dict(detected=dt_, median=round(float(np.median(la)) * 0.05, 2), false=fa_)
        print(f"  {name:<18} {dt_}/{N_EP}  median {np.median(la)*0.05:.2f}s  false {fa_}")

    print(f"\nupright at end of phase A (healthy):  {upA.sum()}/{N_EP}")
    print(f"attempts that ended fallen (damaged): {fell.sum()}/{N_EP * n_att}"
          f"  (all {n_att} failed in {int((fell.all(axis=1)).sum())}/{N_EP} runs)")
    print(f"upright at end of phase C (repaired): {upC.sum()}/{N_EP}")
    print(f"detected {det}/{N_EP}, median {np.median(lat)} steps "
          f"({np.median(lat) * 0.05:.2f}s), false alarms {fa}/{N_EP * (PH_A - WINDOW)}")

    ok = upA & upC & fell.all(axis=1)          # healthy holds, every attempt fails, repair holds
    pick = int(np.argmax(ok)) if ok.any() else int(np.argmax(upC))
    print(f"showing run {pick} (seed {seeds[pick]})")

    ep = eps[pick]
    sig = lambda a: [float(f"{v:.4g}") for v in a]
    rnd = lambda a, n: [round(float(v), n) for v in a]
    json.dump(dict(
        steps=STEPS, dt=0.05, ph_a=PH_A, ph_b=PH_B, window=WINDOW, repair_n=REPAIR_N,
        damage_pct=int(DAMAGE_SCALE * 100), n_ep=N_EP, seed=int(seeds[pick]), pick=pick,
        theta=rnd(ep["theta"], 4), thdot=rnd(ep["thdot"], 3), torque=rnd(ep["torque"], 3),
        cost=rnd(ep["cost"], 3), roll=sig(R[pick]),
        band_lo=sig(R.min(axis=0)), band_hi=sig(R.max(axis=0)),
        threshold=float(f"{thresh:.4g}"),
        retry=RETRY, attempts=(PH_B - PH_A) // RETRY,
        cross=int(np.where(R[pick, PH_A:PH_B] > thresh)[0][0]) + 1,
        detected=det, detect_median=float(np.median(lat)), false_alarms=fa,
        healthy_steps=int(pre_all.size),
        up_a=int(upA.sum()), up_c=int(upC.sum()), n_att=n_att,
        fell_all=int(fell.all(axis=1).sum()), fell_att=int(fell.sum()), robust=robust,
    ), open(os.environ.get("OUT", "story_trace.json"), "w"), separators=(",", ":"))
    print("wrote", os.environ.get("OUT", "story_trace.json"))
