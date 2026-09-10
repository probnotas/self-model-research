"""Why does the planner only swing up 3 times in 10?

Mean cost is the wrong score for this: outcomes are bimodal (about 0.01 when it
balances, about 4.00 when it hangs), so a mean of 3.7 describes no actual run.
This scores SUCCESS RATE -- the fraction of episodes still upright at the end --
across planning horizons and CEM budgets, on the healthy body.
"""
import os, sys, time
import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Rung 1"))
import sample_effeciency as se

N_EP, STEPS = 8, 200


def score(model, horizon, iters, cands, elites, seeds):
    se.HORIZON, se.CEM_ITERS = horizon, iters
    se.CEM_CANDIDATES, se.CEM_ELITES = cands, elites
    env = gym.make("Pendulum-v1", max_episode_steps=STEPS + 10)
    costs, ups = [], 0
    for sd in seeds:
        obs, _ = env.reset(seed=int(sd))
        np.random.seed(int(sd) % (2**31 - 1))
        c = []
        for _ in range(STEPS):
            obs, _, term, trunc, _ = env.step(se.choose_action(model, obs))
            c.append(se.cost(obs))
            if term or trunc:
                obs, _ = env.reset()
        costs.append(np.mean(c))
        ups += int(np.mean(c[-50:]) < 1.0)          # still upright at the end
    env.close()
    return np.mean(costs), ups / len(seeds)


if __name__ == "__main__":
    S, A, S2 = se.collect_data(2000, seed=se.MASTER_SEED)
    model = se.train_model(S, A, S2, seed=0)
    seeds = se.get_eval_seeds(N_EP)
    print(f"{'horizon':>7} {'iters':>6} {'cands':>6} {'mean cost':>10} {'upright':>9} {'sec':>6}")
    for h, it, cd, el in [(20, 5, 100, 10), (40, 5, 100, 10), (40, 8, 200, 20),
                          (60, 8, 200, 20), (80, 8, 300, 30), (100, 8, 300, 30)]:
        t = time.time()
        m, u = score(model, h, it, cd, el, seeds)
        print(f"{h:>7} {it:>6} {cd:>6} {m:>10.3f} {u * 100:>8.0f}% {time.time() - t:>6.0f}", flush=True)
