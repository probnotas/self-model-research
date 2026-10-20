"""Shared pieces for the Brax Ant body-vs-world experiment.

Importable (no argparse here). Configuration comes from environment variables:
  SCALE      "full" (default) or "smoke"
  COMPLIANT  "1" to drive joints through a weak PD spring (kp, kd below)
  OUT        output directory (default experiments/brax_ant/out[/compliant])

SIMULATION ONLY (Brax / MuJoCo-MJX). Nothing here is hardware.
"""
import os
import pickle
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCALE = os.environ.get("SCALE", "full")
COMPLIANT = os.environ.get("COMPLIANT", "0") == "1"
OUT = os.environ.get("OUT", os.path.join(HERE, "out")) + ("/compliant" if COMPLIANT else "/rigid")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- experiment constants
T_TOTAL, ONSET = 400, 200                     # steps of dt = 0.05 s -> 20 s, event at 10 s
ACTION_NOISE = 0.1
KP, KD = 0.6, 0.04                            # compliant mode: ctrl = kp*(target - q) - kd*qd (normalized units)
PRE, EARLY, LATE = (100, 190), (200, 260), (300, 390)   # window start times t (prediction made at t)
H = 10                                        # multi-step horizon / context
FALL_Z = 0.25

BODY_EVENTS = ("motor_weak", "leg_heavy", "joint_damped")
WORLD_EVENTS = ("push_short", "push_sustained", "payload")
NONE = "none"

if SCALE == "smoke":
    PPO = dict(num_timesteps=200_000, num_evals=2, num_envs=128, batch_size=64, num_minibatches=4,
               episode_length=200)
    N_TRAIN, N_PER_CELL, SM_EPOCHS = 24, 2, 2
    REWARD_TARGET = None                      # no reward gate at smoke scale
else:
    PPO = dict(num_timesteps=50_000_000, num_evals=10, num_envs=4096, batch_size=2048, num_minibatches=32,
               episode_length=1000)
    N_TRAIN, N_PER_CELL, SM_EPOCHS = 512, 32, 30
    REWARD_TARGET = 5000.0
PPO.update(reward_scaling=10, normalize_observations=True, action_repeat=1, unroll_length=5,
           num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2, seed=1)
SPLIT_SEED = {"train": 1000, "dev": 2000, "test": 777000}
SM_SEEDS = (0, 1, 2)


# ---------------------------------------------------------------- environment + leg map
def make_env():
    from brax import envs
    return envs.get_environment("ant", backend="mjx")


@dataclass
class LegMap:
    act: list        # actuator indices per leg
    dof: list        # dof indices per leg
    qpos: list       # qpos indices per leg
    bodies: list     # body indices per leg (whole subtree below the torso)
    obs: list        # the leg's own 4 obs dims (2 angles + 2 joint velocities)
    torso: int


def leg_map(sys):
    """Derive everything per leg from the model, never from assumed ordering."""
    trn = np.asarray(sys.actuator_trnid)[:, 0]
    jb = np.asarray(sys.jnt_bodyid); dofadr = np.asarray(sys.jnt_dofadr); qadr = np.asarray(sys.jnt_qposadr)
    parent = np.asarray(sys.body_parentid)
    torso = 1
    tops = [b for b in range(len(parent)) if parent[b] == torso]
    assert len(tops) == 4, f"expected 4 leg roots under torso, got {tops}"

    def subtree(b):
        out, stack = [], [b]
        while stack:
            x = stack.pop(); out.append(x)
            stack += [c for c in range(len(parent)) if parent[c] == x and c != x]
        return sorted(out)

    legs = []
    for top in tops:
        bodies = subtree(top)
        joints = [j for j in range(len(jb)) if jb[j] in bodies]
        legs.append((bodies, joints))
    legs.sort(key=lambda bj: min(bj[1]))      # leg order = order of their joints in qpos (hip_1, ankle_1, ...)
    m = LegMap(act=[], dof=[], qpos=[], bodies=[], obs=[], torso=torso)
    for bodies, joints in legs:
        m.bodies.append(bodies)
        m.dof.append([int(dofadr[j]) for j in joints])
        m.qpos.append([int(qadr[j]) for j in joints])
        m.act.append([i for i in range(len(trn)) if trn[i] in joints])
        # obs = [q[2:], qd]: joint angle obs index = qpos - 2, joint velocity obs index = 13 + dof
        m.obs.append([q - 2 for q in m.qpos[-1]] + [13 + d for d in m.dof[-1]])
    return m


# ---------------------------------------------------------------- control: rigid or compliant
def act_qpos_dof(sys, legs):
    """Per-actuator qpos and dof index (actuator order)."""
    trn = np.asarray(sys.actuator_trnid)[:, 0]
    return (jnp.array(np.asarray(sys.jnt_qposadr)[trn]), jnp.array(np.asarray(sys.jnt_dofadr)[trn]))


def to_ctrl(action, ps, aq, ad):
    """Rigid: ctrl = action. Compliant: weak PD spring toward target angle = action."""
    if not COMPLIANT:
        return action
    return jnp.clip(KP * (action - ps.q[aq]) - KD * ps.qd[ad], -1.0, 1.0)


def make_compliant_env():
    """Ant whose policy outputs joint targets, applied through the weak spring every physics frame."""
    from brax.envs.ant import Ant

    class CompliantAnt(Ant):
        def __init__(self, **kw):
            super().__init__(backend="mjx", **kw)
            self._aq, self._ad = act_qpos_dof(self.sys, None)

        def pipeline_step(self, pipeline_state, action):
            def f(s, _):
                return self._pipeline.step(self.sys, s, to_ctrl(action, s, self._aq, self._ad), self._debug), None
            return jax.lax.scan(f, pipeline_state, (), self._n_frames)[0]

    return CompliantAnt()


def get_train_env():
    if COMPLIANT:
        assert COMPLIANT
        return make_compliant_env()
    return make_env()


# ---------------------------------------------------------------- policy I/O
def save(obj, name):
    with open(os.path.join(OUT, name), "wb") as f:
        pickle.dump(obj, f)


def load(name):
    with open(os.path.join(OUT, name), "rb") as f:
        return pickle.load(f)


def policy_fn(env):
    """Deterministic frozen policy from saved params."""
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.acme import running_statistics
    params = load("policy_params.pkl")
    nets = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                          preprocess_observations_fn=running_statistics.normalize)
    infer = ppo_networks.make_inference_fn(nets)(params, deterministic=True)
    return lambda obs: infer(obs, jax.random.PRNGKey(0))[0]


# ---------------------------------------------------------------- events
def body_weight(sys):
    return float(np.asarray(sys.body_mass).sum()) * 9.81


def trial_specs(split):
    """List of dicts: event, leg, seed, and the event's random parameters."""
    base = SPLIT_SEED[split]
    cells = [(NONE, -1)] * N_TRAIN if split == "train" else \
        [(e, l) for e in BODY_EVENTS for l in range(4)] + [(e, -1) for e in WORLD_EVENTS] + [(NONE, -1)]
    reps = 1 if split == "train" else N_PER_CELL
    specs, k = [], 0
    for ev, leg in cells:
        for _ in range(reps):
            seed = base + k; k += 1
            r = np.random.default_rng(seed)
            ang = r.uniform(0, 2 * np.pi)
            mag = r.uniform(0.5, 1.5) if ev == "push_short" else r.uniform(0.15, 0.3) if ev == "push_sustained" else 0.0
            specs.append(dict(event=ev, leg=int(leg), seed=int(seed), push_dir=[float(np.cos(ang)), float(np.sin(ang))],
                              push_frac=float(mag)))
    return specs


def event_arrays(specs, sys, legs):
    """Per-trial multipliers / forces, all applied only for t >= ONSET."""
    n, nu, nb, nd = len(specs), sys.actuator_gear.shape[0], sys.body_mass.shape[0], sys.dof_damping.shape[0]
    gear, mass, dmul, dadd = np.ones((n, nu)), np.ones((n, nb)), np.ones((n, nd)), np.zeros((n, nd))
    force, f_end = np.zeros((n, 3)), np.zeros(n, dtype=np.int32)
    W = body_weight(sys)
    for i, s in enumerate(specs):
        e, l = s["event"], s["leg"]
        if e == "motor_weak":
            gear[i, legs.act[l]] = 0.5
        elif e == "leg_heavy":
            mass[i, legs.bodies[l]] = 3.0
        elif e == "joint_damped":
            dmul[i, legs.dof[l]] = 3.0; dadd[i, legs.dof[l]] = 1.0
        elif e == "payload":
            mass[i, legs.torso] = 1.5
        elif e in ("push_short", "push_sustained"):
            force[i, :2] = np.array(s["push_dir"]) * s["push_frac"] * W
            f_end[i] = ONSET + 4 if e == "push_short" else T_TOTAL      # 0.2 s = 4 env steps
    return {k: jnp.array(v, dtype=jnp.float32) if k != "f_end" else jnp.array(v)
            for k, v in dict(gear=gear, mass=mass, dmul=dmul, dadd=dadd, force=force, f_end=f_end).items()}


# ---------------------------------------------------------------- the trial loop
def make_rollout(env, pol, legs):
    """jit-able rollout of one trial; vmapped over trials by the caller."""
    sys0 = env.sys
    aq, ad = act_qpos_dof(sys0, legs)
    torso = legs.torso
    pipe, nfr = env._pipeline, env._n_frames

    def one(key, ev):
        k_reset, k_noise = jax.random.split(key)
        state = env.reset(k_reset)
        ps = state.pipeline_state

        def step(carry, t):
            ps, k = carry
            on = t >= ONSET
            sys_t = sys0.replace(
                actuator_gear=sys0.actuator_gear * jnp.where(on, ev["gear"], 1.0)[:, None],
                body_mass=sys0.body_mass * jnp.where(on, ev["mass"], 1.0),
                body_inertia=sys0.body_inertia * jnp.where(on, ev["mass"], 1.0)[:, None],
                dof_damping=sys0.dof_damping * jnp.where(on, ev["dmul"], 1.0) + jnp.where(on, ev["dadd"], 0.0))
            push = on & (t < ev["f_end"])
            xf = jnp.zeros_like(ps.xfrc_applied).at[torso, :3].set(jnp.where(push, ev["force"], 0.0))
            obs = env._get_obs(ps)
            k, kn = jax.random.split(k)
            a = jnp.clip(pol(obs) + ACTION_NOISE * jax.random.normal(kn, (env.action_size,)), -1.0, 1.0)

            def frame(s, _):
                s = s.replace(xfrc_applied=xf)
                return pipe.step(sys_t, s, to_ctrl(a, s, aq, ad)), None
            ps2 = jax.lax.scan(frame, ps, (), nfr)[0]
            return (ps2, k), (obs, a)

        (psT, _), (obs, act) = jax.lax.scan(step, (ps, k_noise), jnp.arange(T_TOTAL))
        obs = jnp.concatenate([obs, env._get_obs(psT)[None]], 0)       # (T+1, 27)
        return obs, act

    return jax.jit(jax.vmap(one))


def run_trials(env, pol, legs, specs, batch=256):
    roll = make_rollout(env, pol, legs)
    ev = event_arrays(specs, env.sys, legs)
    O, A = [], []
    for s in range(0, len(specs), batch):
        sl = slice(s, s + batch)
        keys = jnp.stack([jax.random.PRNGKey(sp["seed"]) for sp in specs[sl]])
        o, a = roll(keys, {k: v[sl] for k, v in ev.items()})
        O.append(np.asarray(o)); A.append(np.asarray(a))
        print(f"    trials {min(s + batch, len(specs))}/{len(specs)}", flush=True)
    return np.concatenate(O), np.concatenate(A)


def fell(obs):
    """Per trial: torso z < FALL_Z or flipped, at any step after onset."""
    z = obs[:, ONSET:, 0]
    qx, qy = obs[:, ONSET:, 2], obs[:, ONSET:, 3]
    up = 1 - 2 * (qx ** 2 + qy ** 2)
    return ((z < FALL_Z) | (up < 0)).any(1)
