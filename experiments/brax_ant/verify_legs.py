"""Gotcha 1 check: does each body event change mostly THAT leg's joint velocities?

    python verify_legs.py

Paired trials (same seeds, same action noise): event vs no event. For steps
ONSET..ONSET+10 (before the controller has much time to spread the effect),
measures each leg's share of the joint-velocity deviation. Fails if any leg's
own share is not the largest. Also prints the leg map derived from the model.
"""
import json
import os
import sys


def main():
    import numpy as np
    import common as C
    env = C.get_train_env()
    legs = C.leg_map(env.sys)
    print("leg map (derived from sys):")
    for l in range(4):
        print(f"  leg {l + 1}: actuators {legs.act[l]}  dofs {legs.dof[l]}  qpos {legs.qpos[l]}  bodies {legs.bodies[l]}  obs {legs.obs[l]}")
    print("  actuator_trnid joints:", np.asarray(env.sys.actuator_trnid)[:, 0].tolist())
    pol = C.policy_fn(env)
    seeds = [5000 + i for i in range(8)]
    base = [dict(event=C.NONE, leg=-1, seed=s, push_dir=[1, 0], push_frac=0) for s in seeds]
    ob0, _ = C.run_trials(env, pol, legs, base)
    table, ok = {}, True
    for ev in C.BODY_EVENTS:
        for l in range(4):
            sp = [dict(b, event=ev, leg=l) for b in base]
            ob, _ = C.run_trials(env, pol, legs, sp)
            d = (ob[:, C.ONSET + 1:C.ONSET + 11, 19:27] - ob0[:, C.ONSET + 1:C.ONSET + 11, 19:27]) ** 2
            share = np.array([d[:, :, [legs.obs[k][2] - 19, legs.obs[k][3] - 19]].sum() for k in range(4)])
            share = share / max(share.sum(), 1e-12)
            table[f"{ev}_leg{l + 1}"] = share.round(3).tolist()
            own_max = int(np.argmax(share)) == l
            ok &= own_max
            print(f"  {ev:13s} leg {l + 1}: joint-vel deviation share per leg {share.round(2)}  {'OK' if own_max else 'FAIL'}")
    json.dump(dict(ok=bool(ok), shares=table), open(os.path.join(C.OUT, "verify_legs.json"), "w"), indent=1)
    if not ok:
        sys.exit("LEG MAP CHECK FAILED: a body event does not mainly affect its own leg. Fix before any real run.")
    print("leg map check passed")


if __name__ == "__main__":
    main()
