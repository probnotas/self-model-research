"""Is the self-repairing model actually better than RL? Paired test, same seeds.

Every condition is scored on the identical task, the identical damaged body, and
the identical ten start states, so the comparison is paired and a signed-rank
test is meaningful.

  MB-stale      model-based planner, model never updated
  MB-repaired   the demo: fine-tuned on 100 transitions from its own failures
  SAC-0         SAC pre-trained on the healthy body, NO adaptation
  SAC-100       SAC given the SAME 100 environment steps of adaptation
"""
import json, os, sys
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Rung 1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Rung 3"))
import sample_effeciency as se
import bodies
from record_story import finetune, stand_up, START, PH_A, PH_B, RETRY, REPAIR_N

DAMAGE = dict(torque_scale=0.5)


def wilcoxon_exact(a, b):
    """Exact two-sided Wilcoxon signed-rank p. n=10, so enumerate all 2^n sign
    assignments rather than lean on a normal approximation."""
    from itertools import product
    d = [x - y for x, y in zip(a, b) if x != y]
    n = len(d)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    rank = [0.0] * n
    i = 0
    while i < n:                                   # average ranks within ties
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            rank[order[k]] = (i + j) / 2 + 1
        i = j + 1
    obs = sum(r for r, v in zip(rank, d) if v > 0)
    total = sum(rank)
    hits = sum(1 for signs in product([0, 1], repeat=n)
               if abs(sum(r for r, sg in zip(rank, signs) if sg) - total / 2) >= abs(obs - total / 2))
    return hits / 2 ** n


def binom_exact(k, n, p=0.5):
    """Exact two-sided binomial p."""
    from math import comb
    if n == 0:
        return 1.0
    pm = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return min(1.0, sum(v for i, v in enumerate(pm) if v <= pm[k] + 1e-12))
STEPS, N_EP = 200, 10
HELD = 0.35


def score(act_fn, seed, params=DAMAGE):
    env = bodies.make_body(params, max_episode_steps=STEPS + 10)
    rng = np.random.default_rng(int(seed) + 991)      # start states, shared by all conditions
    env.reset(seed=int(seed)); obs = stand_up(env, rng)
    np.random.seed(int(seed) % (2**31 - 1))
    c, ang = [], []
    for _ in range(STEPS):
        obs, _, _, _, _ = env.step(act_fn(obs))
        c.append(se.cost(obs)); ang.append(abs(np.arctan2(obs[1], obs[0])))
    env.close()
    return float(np.mean(c)), int(np.mean(ang[-50:]) < HELD)


def repair(model, seed):
    """Exactly the demo's protocol: two failed attempts on the damaged body."""
    env = bodies.make_body(DAMAGE, max_episode_steps=PH_B - PH_A + 10)
    rng = np.random.default_rng(int(seed))
    env.reset(seed=int(seed)); obs = stand_up(env, rng)
    np.random.seed(int(seed) % (2**31 - 1))
    S, A, S2 = [], [], []
    for t in range(REPAIR_N):
        if t and t % RETRY == 0:
            obs = stand_up(env, rng)
        a = se.choose_action(model, obs)
        o2, _, _, _, _ = env.step(a)
        S.append(obs); A.append(np.asarray(a, dtype=np.float64)); S2.append(o2)
        obs = o2
    env.close()
    return finetune(model, np.array(S), np.array(A), np.array(S2))


if __name__ == "__main__":
    Sd, Ad, S2d = se.collect_data(2000, seed=se.MASTER_SEED)
    healthy = se.train_model(Sd, Ad, S2d, seed=0)
    seeds = se.get_eval_seeds(N_EP)
    res = {k: {"cost": [], "held": []} for k in
           ("MB-healthy", "MB-stale", "MB-repaired", "SAC-healthy", "SAC-0", "SAC-100")}

    for i, sd in enumerate(seeds):
        c, h = score(lambda o: se.choose_action(healthy, o), sd, {})
        res["MB-healthy"]["cost"].append(c); res["MB-healthy"]["held"].append(h)
        c, h = score(lambda o: se.choose_action(healthy, o), sd)
        res["MB-stale"]["cost"].append(c); res["MB-stale"]["held"].append(h)
        rep = repair(healthy, sd)
        c, h = score(lambda o: se.choose_action(rep, o), sd)
        res["MB-repaired"]["cost"].append(c); res["MB-repaired"]["held"].append(h)

        ag = SAC.load("/tmp/sac_healthy")
        c, h = score(lambda o: ag.predict(o, deterministic=True)[0], sd, {})
        res["SAC-healthy"]["cost"].append(c); res["SAC-healthy"]["held"].append(h)
        c, h = score(lambda o: ag.predict(o, deterministic=True)[0], sd)
        res["SAC-0"]["cost"].append(c); res["SAC-0"]["held"].append(h)

        ag2 = SAC.load("/tmp/sac_healthy", env=bodies.make_body(DAMAGE), seed=int(sd) % 10000)
        ag2.load_replay_buffer("/tmp/sac_healthy_buf")
        ag2.learn(total_timesteps=REPAIR_N, reset_num_timesteps=False, progress_bar=False)
        c, h = score(lambda o: ag2.predict(o, deterministic=True)[0], sd)
        res["SAC-100"]["cost"].append(c); res["SAC-100"]["held"].append(h)
        print(f"  seed {i + 1}/{N_EP} done", flush=True)

    print(f"\n{'condition':<14}{'mean cost':>11}{'median':>9}{'held':>8}")
    for k, v in res.items():
        print(f"{k:<14}{np.mean(v['cost']):>11.3f}{np.median(v['cost']):>9.3f}"
              f"{sum(v['held']):>6}/10")

    def cmp(a, b):
        ca, cb = np.array(res[a]["cost"]), np.array(res[b]["cost"])
        wp = wilcoxon_exact(list(ca), list(cb))
        ha, hb = sum(res[a]["held"]), sum(res[b]["held"])
        disc = [(x, y) for x, y in zip(res[a]["held"], res[b]["held"]) if x != y]
        n01 = sum(1 for x, y in disc if x == 0)
        pb = binom_exact(n01, len(disc)) if disc else 1.0
        print(f"{a:>12} vs {b:<12} cost {np.mean(ca):>7.3f} vs {np.mean(cb):<7.3f} "
              f"Wilcoxon p={wp:.4f} | held {ha}/10 vs {hb}/10 McNemar p={pb:.4f}")

    print("\npaired comparisons (n=10, same start states)")
    for a, b in [("MB-stale", "MB-repaired"), ("MB-repaired", "SAC-0"),
                 ("MB-repaired", "SAC-100"), ("MB-stale", "SAC-0"),
                 ("MB-healthy", "SAC-healthy")]:
        cmp(a, b)
    json.dump(res, open("results_head_to_head.json", "w"), indent=1)
