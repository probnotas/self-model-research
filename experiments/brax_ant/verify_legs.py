"""Gotcha 1 check: does each body event change mostly THAT leg's joint velocities?

    python verify_legs.py

Branch test: roll healthy to the onset, then take ONE env step (5 physics frames, same
action) from that exact state twice - healthy model vs event model. The difference is
purely the physics change (no policy reaction, and immune to GPU non-determinism, which
broke an earlier version that compared two separate runs). Measures each leg's share of
that joint-velocity difference. Also prints the leg map.

What fails the check: motor_weak or joint_damped not hitting mainly their own leg. Those
events touch only that leg's actuator / dof indices, so they test the leg MAPPING exactly.
leg_heavy is reported but not fatal: a 3x leg (+0.2 kg on a 0.91 kg robot) shifts the
whole body and its contacts, so its joint-velocity effect is physically non-local
(observed in the smoke run). That is a property of the event, not a mapping error.
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
    import jax
    import jax.numpy as jnp
    base = [dict(event=C.NONE, leg=-1, seed=s, push_dir=[1, 0], push_frac=0) for s in seeds]
    branch = C.make_branch(env, pol, legs)
    keys = jnp.stack([jax.random.PRNGKey(s) for s in seeds])
    table, ok = {}, True
    for ev in C.BODY_EVENTS:
        for l in range(4):
            sp = [dict(b, event=ev, leg=l) for b in base]
            ev_arr = {k: v for k, v in C.event_arrays(sp, env.sys, legs).items() if k in ("gear", "mass", "dmul", "dadd")}
            oh, oe = branch(keys, ev_arr)
            d = (np.asarray(oe)[:, 19:27] - np.asarray(oh)[:, 19:27]) ** 2
            share = np.array([d[:, [legs.obs[k][2] - 19, legs.obs[k][3] - 19]].sum() for k in range(4)])
            share = share / max(share.sum(), 1e-12)
            table[f"{ev}_leg{l + 1}"] = share.round(3).tolist()
            own_max = int(np.argmax(share)) == l
            if ev != "leg_heavy":
                ok &= own_max
            print(f"  {ev:13s} leg {l + 1}: joint-vel deviation share per leg {share.round(2)}  {'OK' if own_max else ('not local (reported)' if ev == 'leg_heavy' else 'FAIL')}")
    heavy_local = [int(np.argmax(table[f"leg_heavy_leg{l + 1}"])) == l for l in range(4)]
    json.dump(dict(mapping_ok=bool(ok), leg_heavy_local=heavy_local, shares=table), open(os.path.join(C.OUT, "verify_legs.json"), "w"), indent=1)
    if not ok:
        sys.exit("LEG MAP CHECK FAILED: a body event does not mainly affect its own leg. Fix before any real run.")
    print("leg map check passed")


if __name__ == "__main__":
    main()
