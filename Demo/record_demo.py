"""Record a real damage-detection trace for the investor demo.

Nothing here is staged. One forward model is trained on 2,000 random-action
transitions from the healthy pendulum, then drives CEM control for 240 steps.
At step 120 the pendulum's mass is increased 50% mid-episode. Everything the
demo page shows is read out of the arrays this script writes.

Recorded per step, for each of 10 independent episodes:
  theta, thdot      the true body state
  action            the torque the planner commanded
  err               one-step prediction error (MSE in next-state space)
  cost              the Rung 1 control cost
  ghost             where the model expected the body to be, from a K-step
                    rollout started K steps earlier using the actions actually
                    taken (a post-hoc read-out, no future information)
"""
import json
import os
import sys

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Rung 1"))
import sample_effeciency as se

TRAIN_SAMPLES = 2000
STEPS = 240
DAMAGE_AT = 120
MASS_SCALE = 1.5
N_EPISODES = 10
WINDOW = 10          # rolling window for the detection statistic
GHOST_KS = (10, 20)  # 0.5 s and 1.0 s lookaheads for the "what it expected" overlay
SIGMA = 3.0
MODE = os.environ.get("MODE", "probe")   # "probe" = random torque, "control" = CEM          # detection threshold: pre-damage mean + SIGMA * pre-damage std


def angle(obs):
    return float(np.arctan2(obs[1], obs[0]))


def rollout_model(model, state, actions):
    """Roll the learned model forward through a fixed action sequence."""
    s = np.asarray(state, dtype=np.float64)
    for a in actions:
        s = se.predict(model, s, np.asarray(a, dtype=np.float64))
    return s


def run_episode(model, seed, mode="probe"):
    env = gym.make("Pendulum-v1", max_episode_steps=STEPS + 10)
    obs, _ = env.reset(seed=int(seed))
    env.action_space.seed(int(seed))
    np.random.seed(int(seed) % (2**31 - 1))       # CEM sampling stream

    states, actions, errs, costs = [], [], [], []
    for t in range(STEPS):
        if t == DAMAGE_AT:
            env.unwrapped.m *= MASS_SCALE          # the damage

        a = (env.action_space.sample() if mode == "probe"
             else se.choose_action(model, obs))
        pred = se.predict(model, obs, a)
        nxt, _, term, trunc, _ = env.step(a)

        states.append(obs)
        actions.append(np.asarray(a, dtype=np.float64))
        errs.append(float(np.mean((pred - nxt) ** 2)))
        costs.append(float(se.cost(nxt)))
        obs = nxt
        if term or trunc:
            obs, _ = env.reset()
    env.close()

    states = np.array(states)

    # Ghost: at step t, where a rollout launched at t-K said the body would be.
    ghosts = {}
    for K in GHOST_KS:
        g = []
        for t in range(STEPS):
            if t < K:
                g.append(angle(states[t]))
            else:
                s = rollout_model(model, states[t - K], actions[t - K:t])
                g.append(float(np.arctan2(s[1], s[0])))
        ghosts[f"ghost{K}"] = g

    return dict(
        theta=[angle(s) for s in states],
        thdot=[float(s[2]) for s in states],
        action=[float(a[0]) for a in actions],
        err=errs,
        cost=costs,
        **ghosts,
    )


def rolling(x, w):
    """Trailing mean; the first w-1 entries use however much history exists."""
    x = np.asarray(x, dtype=float)
    return np.array([x[max(0, i - w + 1):i + 1].mean() for i in range(len(x))])


def main():
    print(f"collecting {TRAIN_SAMPLES} random-action transitions (healthy body)...",
          flush=True)
    S, A, S2 = se.collect_data(TRAIN_SAMPLES, seed=se.MASTER_SEED)

    print("training the forward model (delta targets, 4-64-64-3)...", flush=True)
    model = se.train_model(S, A, S2, seed=0)

    seeds = se.get_eval_seeds(N_EPISODES)
    eps = []
    for i, sd in enumerate(seeds):
        print(f"  episode {i + 1}/{N_EPISODES} (seed {sd})...", flush=True)
        eps.append(run_episode(model, sd, MODE))

    err = np.array([e["err"] for e in eps])           # (N, STEPS)
    cost = np.array([e["cost"] for e in eps])
    roll = np.array([rolling(e["err"], WINDOW) for e in eps])

    # Detection threshold, fixed from PRE-damage behaviour only.
    pre = roll[:, WINDOW:DAMAGE_AT]
    thresh = float(pre.mean() + SIGMA * pre.std())

    latencies, detected = [], 0
    for r in roll:
        after = np.where(r[DAMAGE_AT:] > thresh)[0]
        if len(after):
            detected += 1
            latencies.append(int(after[0]) + 1)       # steps after the damage
    false_alarms = int((pre > thresh).sum())

    summary = dict(
        train_samples=TRAIN_SAMPLES,
        steps=STEPS,
        damage_at=DAMAGE_AT,
        mass_scale=MASS_SCALE,
        mode=MODE,
        n_episodes=N_EPISODES,
        window=WINDOW,
        ghost_ks=list(GHOST_KS),
        sigma=SIGMA,
        threshold=thresh,
        dt=0.05,
        err_pre=float(err[:, WINDOW:DAMAGE_AT].mean()),
        err_post=float(err[:, DAMAGE_AT:].mean()),
        err_ratio=float(err[:, DAMAGE_AT:].mean() / err[:, WINDOW:DAMAGE_AT].mean()),
        cost_pre=float(cost[:, WINDOW:DAMAGE_AT].mean()),
        cost_post=float(cost[:, DAMAGE_AT:].mean()),
        detected=detected,
        detect_median=float(np.median(latencies)) if latencies else None,
        detect_min=int(min(latencies)) if latencies else None,
        detect_max=int(max(latencies)) if latencies else None,
        detect_seconds=float(np.median(latencies) * 0.05) if latencies else None,
        false_alarm_steps=false_alarms,
        pre_steps_scored=int(pre.size),
    )

    out = dict(
        summary=summary,
        roll_mean=roll.mean(axis=0).tolist(),
        roll_lo=roll.min(axis=0).tolist(),
        roll_hi=roll.max(axis=0).tolist(),
        cost_mean=cost.mean(axis=0).tolist(),
        episodes=eps,
        all_err=err.tolist(),
        all_roll=roll.tolist(),
        seeds=[int(x) for x in seeds],
    )
    path = os.path.join(os.path.dirname(__file__), f"demo_trace_{MODE}.json")
    with open(path, "w") as f:
        json.dump(out, f)

    print("\n" + "=" * 62)
    for k, v in summary.items():
        print(f"  {k:>18}: {v}")
    print("=" * 62)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
