"""Record the three-pendulum recovery demo from real runs.

All three conditions start from the SAME state on each seed, so the side-by-side
is a controlled comparison and not three unrelated clips:

  A  healthy body,        healthy model     -- balances
  B  motor at 50%,        stale model       -- falls over
  C  motor at 50%,        model repaired on 25 on-policy samples

All ten evaluation seeds are recorded for every condition so the page can state
how typical the shown run is, rather than implying the best case is the norm.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Rung 1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Rung 3"))
import sample_effeciency as se
import bodies
from balance_probe import START, N_EP
from recovery_probe import DAMAGE, collect_damaged, finetune

STEPS = 200
REPAIR_N = 25


def run(model, params, seed):
    env = bodies.make_body(params, max_episode_steps=STEPS + 10)
    rng = np.random.default_rng(int(seed))
    env.reset(seed=int(seed))
    env.unwrapped.state = np.array([rng.uniform(-START, START), rng.uniform(-START, START)])
    obs = env.unwrapped._get_obs()
    np.random.seed(int(seed) % (2**31 - 1))
    th, td, tq, cs = [], [], [], []
    for _ in range(STEPS):
        a = se.choose_action(model, obs)
        th.append(float(np.arctan2(obs[1], obs[0]))); td.append(float(obs[2]))
        tq.append(float(a[0]))
        obs, _, _, _, _ = env.step(a)
        cs.append(float(se.cost(obs)))
    env.close()
    held = float(np.mean([abs(x) for x in th[-50:]])) < 0.35
    return dict(theta=th, thdot=td, torque=tq, cost=cs,
                mean_cost=float(np.mean(cs)), held=bool(held))


if __name__ == "__main__":
    print("training the healthy model (2,000 healthy-body transitions)...", flush=True)
    S, A, S2 = se.collect_data(2000, seed=se.MASTER_SEED)
    healthy = se.train_model(S, A, S2, seed=0)

    print(f"collecting {REPAIR_N} on-policy transitions on the damaged body...", flush=True)
    repair_data = collect_damaged(healthy, REPAIR_N, 4242, "onpolicy")
    repaired = finetune(healthy, repair_data)

    seeds = se.get_eval_seeds(N_EP)
    conds = {"healthy": (healthy, {}), "stale": (healthy, DAMAGE), "repaired": (repaired, DAMAGE)}
    rec = {k: [] for k in conds}
    for k, (m, p) in conds.items():
        for sd in seeds:
            rec[k].append(run(m, p, sd))
        held = sum(r["held"] for r in rec[k])
        print(f"  {k:>8}: held upright {held}/{N_EP}, "
              f"mean cost {np.mean([r['mean_cost'] for r in rec[k]]):.3f}", flush=True)

    # Selection rule, fixed before looking: the first seed on which every
    # condition produces its majority outcome, so the clip is typical.
    want = {k: (sum(r["held"] for r in rec[k]) * 2 > N_EP) for k in conds}
    pick = next((i for i in range(N_EP) if all(rec[k][i]["held"] == want[k] for k in conds)), 0)
    print(f"\nshowing seed index {pick} (seed {seeds[pick]}); majority outcomes {want}")

    rc = json.load(open("results_balance_recovery.json"))
    out = dict(
        steps=STEPS, dt=0.05, repair_n=REPAIR_N, start=START, n_ep=N_EP,
        seed_index=pick, seed=int(seeds[pick]),
        runs={k: {f: rec[k][pick][f] for f in ("theta", "thdot", "torque", "cost")}
              for k in conds},
        summary={k: dict(held=sum(r["held"] for r in rec[k]),
                         mean_cost=round(float(np.mean([r["mean_cost"] for r in rec[k]])), 3),
                         shown_cost=round(rec[k][pick]["mean_cost"], 3)) for k in conds},
        curve={m: {str(n): rc[m][str(n)] for n in sorted(map(int, rc[m]))}
               for m in ("onpolicy", "random")},
        healthy_ref=rc["healthy_body"], stale_ref=rc["stale"],
        repair_samples=[[list(map(float, s)), float(a[0])]
                        for s, a in zip(repair_data[0], repair_data[1])],
    )
    json.dump(out, open("recovery_trace.json", "w"), separators=(",", ":"))
    print("wrote recovery_trace.json")
