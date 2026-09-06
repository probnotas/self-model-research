"""Uncertainty-aware CEM planning and the Rung 3 evaluation loop.

Planner hyperparameters are identical to Rung 1/2 (horizon 20, 5 iterations,
100 candidates, 10 elites, init std 1.0) so the only thing that changes across
rungs is the model and the cost term.

The uncertainty term follows the PETS intuition: propagate every ensemble
member independently, score the task cost against the ensemble MEAN trajectory,
and penalise the DISAGREEMENT between members. Disagreement compounds over the
horizon in exactly the way single-model error does, so the penalty naturally
discourages plans that run far into regions the ensemble has not pinned down.
"""
import numpy as np
import torch

from models import TARGET, forward, online_adapt

HORIZON = 20
CEM_ITERS = 5
CEM_CANDIDATES = 100
CEM_ELITES = 10
CEM_INIT_STD = 1.0
ACTION_LOW, ACTION_HIGH = -2.0, 2.0


def rollout_costs(params, state, acts, lam=0.0):
    """Score candidate action sequences. Returns (total, task, disagreement).

    acts: (C, H, 1). Each ensemble member propagates its OWN trajectory, so
    member states diverge over the horizon and their spread is a real measure
    of accumulated model uncertainty.
    """
    E = params[0].shape[0]
    C, H, _ = acts.shape
    sim = torch.from_numpy(np.asarray(state, dtype=np.float32)).view(1, 1, 3)
    sim = sim.expand(E, C, 3).contiguous()
    acts_t = torch.from_numpy(np.ascontiguousarray(acts, dtype=np.float32))
    task = torch.zeros(C)
    dis = torch.zeros(C)
    with torch.no_grad():
        for h in range(H):
            a = acts_t[:, h, :].unsqueeze(0).expand(E, C, 1)
            sim = sim + forward(torch.cat([sim, a], dim=2), params)
            task += ((sim.mean(dim=0) - TARGET) ** 2).sum(dim=1)
            if E > 1:
                dis += sim.std(dim=0).sum(dim=1)
    return (task + lam * dis).numpy(), task.numpy(), dis.numpy()


def choose_action(params, state, lam=0.0, rng=None):
    """CEM over action sequences, with an optional disagreement penalty."""
    rng = np.random if rng is None else rng
    mean = np.zeros((HORIZON, 1), dtype=np.float32)
    std = np.full((HORIZON, 1), CEM_INIT_STD, dtype=np.float32)
    for _ in range(CEM_ITERS):
        noise = rng.standard_normal((CEM_CANDIDATES, HORIZON, 1)).astype(np.float32) \
            if hasattr(rng, "standard_normal") else \
            rng.randn(CEM_CANDIDATES, HORIZON, 1).astype(np.float32)
        acts = np.clip(mean + std * noise, ACTION_LOW, ACTION_HIGH).astype(np.float32)
        total, _, _ = rollout_costs(params, state, acts, lam)
        elites = acts[np.argsort(total)[:CEM_ELITES]]
        mean, std = elites.mean(axis=0), elites.std(axis=0)
    return np.clip(mean[0], ACTION_LOW, ACTION_HIGH).astype(np.float32)


def cost(state):
    """Identical to Rung 1/2: squared distance from upright and motionless."""
    return float(np.sum((np.asarray(state) - np.array([1.0, 0.0, 0.0])) ** 2))


def evaluate(env_fn, eval_seeds, action_fn, steps=200):
    """Rung 3 evaluation loop.

    Semantics match Rung 1's evaluate() exactly (same cost, same 200-step
    episodes, same start-state seeds, same reset-on-termination behaviour). It
    is reimplemented here only because online adaptation needs the running
    (s, a, s') history, which the Rung 1 signature does not expose.

    action_fn(state, history) -> action, where history is a list of
    (state, action, next_state) tuples for the current episode.
    """
    env = env_fn()
    run_costs = []
    for seed in eval_seeds:
        state, _ = env.reset(seed=int(seed))
        history, total = [], 0.0
        for _ in range(steps):
            action = action_fn(state, history)
            nxt, _, term, trunc, _ = env.step(action)
            history.append((state, action, nxt))
            total += cost(nxt)
            state = nxt
            if term or trunc:
                state, _ = env.reset()
        run_costs.append(total / steps)
    env.close()
    r = np.array(run_costs)
    return float(r.mean()), float(r.std(ddof=1)), r


# ---------------------------------------------------------------- controllers
def static_controller(params, lam=0.0):
    """Plan with fixed parameters — no online adaptation."""
    def act(state, history):
        return choose_action(params, state, lam)
    return act


def adaptive_controller(params, lam=0.0, K=10, alpha=0.01, steps=1):
    """GrBAL-style: re-adapt on the last K transitions before every decision.

    Adaptation always restarts from the meta-parameters rather than compounding,
    because those are the weights that were optimised for K-sample adaptation.
    """
    def act(state, history):
        p = params
        if len(history) >= K:
            win = history[-K:]
            S = np.array([h[0] for h in win], dtype=np.float32)
            A = np.array([h[1] for h in win], dtype=np.float32).reshape(K, 1)
            S2 = np.array([h[2] for h in win], dtype=np.float32)
            p = online_adapt(params, (S, A, S2), alpha, steps)
        return choose_action(p, state, lam)
    return act
