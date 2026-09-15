"""Rung 4 - can the self-model tell "my body changed" from "something pushed me"?

Both events look identical at the instant they happen: the model's prediction
stops matching reality. The claim under test is that they differ in SHAPE, not
size - a body change is permanent, so the error stays elevated; a world event is
transient, so the error spikes and returns.

The design point that makes this non-trivial: the world push is CALIBRATED so its
peak error matches the body change's. Without that, "body vs world" collapses into
"big vs small" and the classifier learns nothing about self-models.

  python3 rung4.py calibrate   # size the push to match the body change
  python3 rung4.py run         # the experiment
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Rung 1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Rung 3"))
import sample_effeciency as se
import world

TRAIN = 2000
STEPS, ONSET = 240, 80
START = 0.2
WINDOW, SIGMA = 10, 3.0
PUSH_STEPS = 5
N_SEEDS = 20

# Doubling the mass and halving the actuator are BIT-IDENTICAL for this body:
# both reduce the torque term 3/(m l^2) * (u * torque_scale) to 1.5u and leave
# the gravity term untouched. They are one perturbation, not two, so only one
# appears here - and no self-model can ever separate them (see identifiability.py).
BODY = {
    "body: mass doubled":  dict(kind="body", params=dict(m=2.0)),
    "body: length +50%":   dict(kind="body", params=dict(l=1.5)),
    "body: damping added": dict(kind="body", params=dict(damping=0.5)),
}
MATCHED_PUSH = 0.75      # error scales ~quadratically with torque; 0.75 matches the body peak


def stand_up(env, rng):
    env.unwrapped.state = np.array([rng.uniform(-START, START), rng.uniform(-START, START)])
    return env.unwrapped._get_obs()


def episode(model, cond, seed, steps=STEPS):
    env = world.make({}, max_episode_steps=steps + 10)
    rng = np.random.default_rng(int(seed))
    env.reset(seed=int(seed)); obs = stand_up(env, rng)
    np.random.seed(int(seed) % (2**31 - 1))
    err, res, th = [], [], []
    for t in range(steps):
        if t == ONSET:
            if cond["kind"] == "body":
                for k, v in cond["params"].items():
                    setattr(env.unwrapped, k, v)
            elif cond["kind"] == "world":
                env.unwrapped.ext_torque = cond["torque"]
        if cond["kind"] == "world" and t == ONSET + cond["steps"]:
            env.unwrapped.ext_torque = 0.0

        a = se.choose_action(model, obs)
        pred = se.predict(model, obs, a)
        nxt, _, _, _, _ = env.step(a)
        err.append(float(np.mean((pred - nxt) ** 2)))
        res.append((pred - nxt).tolist())
        th.append(float(np.arctan2(obs[1], obs[0])))
        obs = nxt
    env.close()
    return dict(err=err, res=res, theta=th)


def rolling(x, w=WINDOW):
    x = np.asarray(x, float)
    return np.array([x[max(0, i - w + 1):i + 1].mean() for i in range(len(x))])


def peak(e):
    return float(rolling(e)[ONSET:ONSET + 15].max())


def train():
    S, A, S2 = se.collect_data(TRAIN, seed=se.MASTER_SEED)
    return se.train_model(S, A, S2, seed=0)


def calibrate(model, seeds):
    """Pick the push torque whose peak error matches the body changes'."""
    target = {}
    for name, c in BODY.items():
        target[name] = float(np.mean([peak(episode(model, c, s)["err"]) for s in seeds]))
        print(f"  {name:<22} peak error {target[name]:.2e}", flush=True)
    goal = float(np.mean(list(target.values())))
    print(f"  target peak to match: {goal:.2e}\n", flush=True)

    best, rows = None, []
    for tq in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        c = dict(kind="world", torque=tq, steps=PUSH_STEPS)
        eps = [episode(model, c, s) for s in seeds]
        pk = float(np.mean([peak(e["err"]) for e in eps]))
        fell = float(np.mean([abs(np.array(e["theta"])[-40:]).mean() > 0.35 for e in eps]))
        rows.append((tq, pk, fell))
        print(f"  push {tq:>4}  peak {pk:.2e}  ratio {pk / goal:>5.2f}x  "
              f"knocked over {fell * 100:>3.0f}%", flush=True)
        if fell == 0.0 and (best is None or abs(np.log(pk / goal)) < abs(np.log(rows[best][1] / goal))):
            best = len(rows) - 1
    if best is None:
        best = int(np.argmin([abs(np.log(r[1] / goal)) for r in rows]))
    print(f"\n  chosen push torque: {rows[best][0]} "
          f"(peak {rows[best][1]:.2e} vs body {goal:.2e})")
    return rows[best][0], goal, rows


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    model = train()
    seeds = se.get_eval_seeds(N_SEEDS)

    if mode == "calibrate":
        print("calibrating the world push against the body changes (5 seeds)\n")
        tq, goal, rows = calibrate(model, seeds[:5])
        json.dump(dict(push=tq, body_peak=goal, rows=rows),
                  open("calibration.json", "w"), indent=1)
        raise SystemExit

    cal = json.load(open("calibration.json"))
    conds = {"baseline (nothing happens)": dict(kind="none")}
    conds.update(BODY)
    conds["world: push (peak-matched)"] = dict(kind="world", torque=MATCHED_PUSH, steps=PUSH_STEPS)
    conds["world: push (bigger spike)"] = dict(kind="world", torque=cal["push"], steps=PUSH_STEPS)
    conds["world: long push (10 steps)"] = dict(kind="world", torque=MATCHED_PUSH, steps=PUSH_STEPS * 2)

    out = {}
    for name, c in conds.items():
        out[name] = [episode(model, c, s) for s in seeds]
        e = np.array([r["err"] for r in out[name]])
        print(f"{name:<28} peak {np.mean([peak(r['err']) for r in out[name]]):.2e}  "
              f"late(+60) {rolling(e.mean(axis=0))[ONSET + 60]:.2e}", flush=True)

    json.dump(dict(steps=STEPS, onset=ONSET, window=WINDOW, sigma=SIGMA,
                   n_seeds=N_SEEDS, push=cal["push"], push_steps=PUSH_STEPS,
                   conditions={k: [{"err": r["err"], "res": r["res"], "theta": r["theta"]}
                                   for r in v] for k, v in out.items()}),
              open("results_rung4.json", "w"), separators=(",", ":"))
    print("\nwrote results_rung4.json")
