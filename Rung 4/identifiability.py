"""Some body changes are not just hard to tell apart - they are the same event.

The pendulum's angular acceleration is

    thdd = 3g/(2l) * sin(th)  +  3/(m l^2) * (u * torque_scale)

so m and torque_scale enter ONLY through the ratio torque_scale / m. Doubling the
mass and halving the actuator gain produce literally the same trajectory, for
every state and every command.

This is an identifiability result, not a measurement limitation: no forward model,
no amount of data, and no cleverer algorithm can separate "I got heavier" from
"my motor got weaker" on this body. A self-model can know THAT it changed, and
(Rung 4) whether the cause was internal or external, but not always WHICH
parameter moved.
"""
import numpy as np
import world


def compare(pa, pb, n=500, seed=3, state=(0.15, -0.1)):
    a, b = world.make(pa), world.make(pb)
    a.reset(seed=seed); b.reset(seed=seed)
    a.unwrapped.state = np.array(state); b.unwrapped.state = np.array(state)
    a.action_space.seed(seed)
    worst, first = 0.0, None
    for i in range(n):
        u = a.action_space.sample()
        oa, *_ = a.step(u); ob, *_ = b.step(u)
        d = float(np.abs(oa - ob).max())
        if i == 0:
            first = d
        worst = max(worst, d)
    a.close(); b.close()
    return first, worst


if __name__ == "__main__":
    print("pairs of body changes, max state deviation over 500 shared commands\n")
    pairs = [
        ("mass x2", dict(m=2.0), "motor at 50%", dict(torque_scale=0.5)),
        ("mass x1.5", dict(m=1.5), "motor at 66.7%", dict(torque_scale=1 / 1.5)),
        ("mass x2", dict(m=2.0), "length +50%", dict(l=1.5)),
        ("mass x2", dict(m=2.0), "damping 0.5", dict(damping=0.5)),
        ("length +50%", dict(l=1.5), "damping 0.5", dict(damping=0.5)),
    ]
    print(f"  {'pair':<30} {'step 1':>10} {'worst':>10}")
    for na, pa, nb, pb in pairs:
        f, w = compare(pa, pb)
        # The pendulum is chaotic under random torque, so a difference that starts
        # at float rounding still grows. Aliasing has to be judged at step 1.
        tag = ("ALIASED - the same event" if f == 0.0 else
               "aliased up to float rounding" if f < 1e-12 else "distinguishable")
        print(f"  {na + ' vs ' + nb:<30} {f:>10.3e} {w:>10.3e}   {tag}")
    print("\nm and torque_scale enter the dynamics only as torque_scale / m,")
    print("so that whole family collapses to one degree of freedom.")
