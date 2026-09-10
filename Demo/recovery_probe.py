"""Does fine-tuning restore balance after the motor is weakened?

Damage: torque_scale 0.5 (the actuator delivers half the commanded torque).
Chosen because it is the damage that actually breaks balance-hold -- mass +50%,
the Rung 2 damage, leaves 90% of runs still upright, which is why it never made
a legible demo.

Two data sources for the repair, because it matters which one is realistic:
  random    random torque on the damaged body (motor babbling)
  onpolicy  the stale controller trying to balance and failing -- the data a
            robot would actually generate while discovering it is broken
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Rung 1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Rung 3"))
import sample_effeciency as se
import bodies
from balance_probe import balance_eval, N_EP, START

DAMAGE = dict(torque_scale=0.5)
SIZES = [10, 25, 50, 100, 200, 500]
FT_EPOCHS = 1000


def collect_damaged(model, n, seed, mode):
    """n transitions from the damaged body, either random or under the stale planner."""
    env = bodies.make_body(DAMAGE, max_episode_steps=200)
    env.action_space.seed(seed)
    rng = np.random.default_rng(seed)
    np.random.seed(seed % (2**31 - 1))
    S, A, S2 = [], [], []
    env.reset(seed=seed)
    env.unwrapped.state = np.array([rng.uniform(-START, START), rng.uniform(-START, START)])
    obs = env.unwrapped._get_obs()
    t = 0
    while len(S) < n:
        a = env.action_space.sample() if mode == "random" else se.choose_action(model, obs)
        o2, _, _, _, _ = env.step(a)
        S.append(obs); A.append(np.asarray(a, dtype=np.float64)); S2.append(o2)
        obs = o2; t += 1
        if t % 200 == 0:                        # restart near upright, as a robot retrying would
            env.reset()
            env.unwrapped.state = np.array([rng.uniform(-START, START), rng.uniform(-START, START)])
            obs = env.unwrapped._get_obs()
    env.close()
    return np.array(S), np.array(A), np.array(S2)


def finetune(model, data, epochs=FT_EPOCHS, lr=1e-3):
    import copy, torch
    m = copy.deepcopy(model)
    S, A, S2 = data
    X = torch.tensor(np.concatenate([S, A], axis=1), dtype=torch.float32)
    Y = torch.tensor(S2 - S, dtype=torch.float32)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lf = torch.nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); lf(m(X), Y).backward(); opt.step()
    return m


if __name__ == "__main__":
    S, A, S2 = se.collect_data(2000, seed=se.MASTER_SEED)
    healthy = se.train_model(S, A, S2, seed=0)
    seeds = se.get_eval_seeds(N_EP)

    out = {}
    c, h = balance_eval(healthy, {}, seeds);            out["healthy_body"] = (c, h)
    print(f"{'healthy body, healthy model':>34} {c:>9.3f} {h*100:>6.0f}%", flush=True)
    c, h = balance_eval(healthy, DAMAGE, seeds);        out["stale"] = (c, h)
    print(f"{'motor 50%, STALE model':>34} {c:>9.3f} {h*100:>6.0f}%", flush=True)

    for mode in ("onpolicy", "random"):
        out[mode] = {}
        for n in SIZES:
            data = collect_damaged(healthy, n, 4242, mode)
            m = finetune(healthy, data)
            c, h = balance_eval(m, DAMAGE, seeds)
            out[mode][n] = (c, h)
            print(f"{f'motor 50%, +{n} {mode} samples':>34} {c:>9.3f} {h*100:>6.0f}%", flush=True)
    json.dump(out, open("results_balance_recovery.json", "w"), indent=1)
