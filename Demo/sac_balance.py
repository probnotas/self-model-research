"""The control that decides whether the recovery claim has a headline.

Rung 2's lesson was that a pre-trained model-free policy barely degrades under
damage, which killed the "1000x" claim. So before claiming anything about
sample efficiency here, check: does SAC actually FAIL on the weakened motor?
Only if it does is there a recovery to compare against.
"""
import json
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Rung 1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Rung 3"))
import sample_effeciency as se
import bodies

DAMAGE = dict(torque_scale=0.5)
START, N_EP, STEPS = 0.2, 10, 200
PRETRAIN = 20000
CKPTS = [25, 50, 100, 200, 500, 1000, 2000, 5000, 10000]


def sac_balance(agent, params, seeds):
    env = bodies.make_body(params, max_episode_steps=STEPS + 10)
    costs, held = [], 0
    for sd in seeds:
        rng = np.random.default_rng(int(sd))
        env.reset(seed=int(sd))
        env.unwrapped.state = np.array([rng.uniform(-START, START), rng.uniform(-START, START)])
        obs = env.unwrapped._get_obs()
        c, ang = [], []
        for _ in range(STEPS):
            a, _ = agent.predict(obs, deterministic=True)
            obs, _, _, _, _ = env.step(a)
            c.append(se.cost(obs)); ang.append(abs(np.arctan2(obs[1], obs[0])))
        costs.append(np.mean(c)); held += int(np.mean(ang[-50:]) < 0.35)
    env.close()
    return float(np.mean(costs)), held / len(seeds)


if __name__ == "__main__":
    seeds = se.get_eval_seeds(N_EP)
    out = {}
    print(f"training SAC {PRETRAIN} steps on the healthy body...", flush=True)
    agent = SAC("MlpPolicy", gym.make("Pendulum-v1"), verbose=0, seed=0)
    agent.learn(total_timesteps=PRETRAIN, progress_bar=False)
    agent.save("/tmp/sac_healthy"); agent.save_replay_buffer("/tmp/sac_healthy_buf")

    for lab, p in [("healthy body", {}), ("motor 50%, no adaptation", DAMAGE)]:
        c, h = sac_balance(agent, p, seeds); out[lab] = (c, h)
        print(f"{lab:>34} {c:>9.3f} {h*100:>6.0f}%", flush=True)

    out["adapt"] = {}
    for n in CKPTS:
        ag = SAC.load("/tmp/sac_healthy", env=bodies.make_body(DAMAGE))
        ag.load_replay_buffer("/tmp/sac_healthy_buf")     # save() omits the buffer
        ag.learn(total_timesteps=n, reset_num_timesteps=False, progress_bar=False)
        c, h = sac_balance(ag, DAMAGE, seeds); out["adapt"][n] = (c, h)
        print(f"{f'motor 50%, +{n} SAC steps':>34} {c:>9.3f} {h*100:>6.0f}%", flush=True)
    json.dump(out, open("results_sac_balance.json", "w"), indent=1)
