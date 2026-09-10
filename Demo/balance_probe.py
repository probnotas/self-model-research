"""Can the model-based controller HOLD the pendulum upright, and does damage break it?

Swing-up from a hanging start is beyond this planner (0-25% success at every
horizon and budget tried -- see planner_sweep.py). Balance-hold is a different,
easier task: start near upright and stay there. If the healthy model holds and a
damaged body breaks it, that is the recovery story, visibly.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Rung 1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Rung 3"))
import sample_effeciency as se
import bodies

N_EP, STEPS = 10, 200
START = 0.2          # start within +-0.2 rad of upright, +-0.2 rad/s


def balance_eval(model, params, seeds, steps=STEPS):
    env = bodies.make_body(params, max_episode_steps=steps + 10)
    costs, held = [], 0
    for sd in seeds:
        rng = np.random.default_rng(int(sd))
        env.reset(seed=int(sd))
        env.unwrapped.state = np.array([rng.uniform(-START, START),
                                        rng.uniform(-START, START)])
        obs = env.unwrapped._get_obs()
        np.random.seed(int(sd) % (2**31 - 1))
        c, ang = [], []
        for _ in range(steps):
            obs, _, _, _, _ = env.step(se.choose_action(model, obs))
            c.append(se.cost(obs))
            ang.append(abs(np.arctan2(obs[1], obs[0])))
        costs.append(np.mean(c))
        held += int(np.mean(ang[-50:]) < 0.35)      # still upright at the end
    env.close()
    return float(np.mean(costs)), held / len(seeds)


if __name__ == "__main__":
    S, A, S2 = se.collect_data(2000, seed=se.MASTER_SEED)
    model = se.train_model(S, A, S2, seed=0)
    seeds = se.get_eval_seeds(N_EP)

    print(f"{'body':>22} {'mean cost':>10} {'held upright':>13}")
    for name, p in [("healthy (stock)", {}),
                    ("mass +50%", dict(m=1.5)),
                    ("mass x2", dict(m=2.0)),
                    ("motor at 50%", dict(torque_scale=0.5)),
                    ("motor at 35%", dict(torque_scale=0.35)),
                    ("length +50%", dict(l=1.5))]:
        m, h = balance_eval(model, p, seeds)
        print(f"{name:>22} {m:>10.3f} {h * 100:>12.0f}%", flush=True)
