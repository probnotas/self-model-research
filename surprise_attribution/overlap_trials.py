"""Paired triplets: body only / push only / body + push, plus undisturbed.

run_experiment.run_episode can only schedule ONE disturbance, and the earlier
files are left untouched, so this module has its own episode loop. It is the
same loop line for line (same env, same reset, same planner seeding, same
error definition); the only difference is a schedule that can apply a body
change AND a push in the same episode. verify_matches_existing() checks that
for single-disturbance trials it reproduces run_episode bit for bit.

Which onset the classifier is given: the moment surprise starts, i.e. the
earliest real disturbance. body only -> body onset; push only -> push onset;
body + push -> whichever came first. In deployment that is what an onset
detector would report; the classifier never sees the true cause or timing of
the second disturbance.
"""
import numpy as np
import torch

import alarm_config as ac
import env_setup
import overlap_config as oc
import planner
import run_experiment
import trials

# latest classifier onset is the latest push onset (body max + offset max)
STEPS = oc.BODY_ONSET_MAX + oc.OFFSET_MAX + ac.EXT_MAX_LAG + oc.EPISODE_MARGIN


# ---------------------------------------------------------------------------
# specs
# ---------------------------------------------------------------------------

def make_specs():
    """N_TRIPLETS triplets + N_UNDISTURBED controls, each with a fresh seed."""
    rng = np.random.default_rng(oc.SEED_BASE)
    specs = []
    for i in range(oc.N_TRIPLETS):
        seed = oc.SEED_BASE + i
        body_onset = int(rng.integers(oc.BODY_ONSET_MIN, oc.BODY_ONSET_MAX + 1))
        offset = int(rng.integers(oc.OFFSET_MIN, oc.OFFSET_MAX + 1))
        draw = dict(triplet=i, seed=seed,
                    param=str(rng.choice(oc.BODY_PARAMS)),
                    factor=float(rng.uniform(oc.BODY_FACTOR_MIN, oc.BODY_FACTOR_MAX)),
                    torque=float((1 if rng.random() < 0.5 else -1)
                                 * rng.uniform(oc.PUSH_TORQUE_MIN, oc.PUSH_TORQUE_MAX)),
                    duration=int(rng.integers(oc.PUSH_STEPS_MIN, oc.PUSH_STEPS_MAX + 1)),
                    body_onset=body_onset, push_onset=body_onset + offset,
                    offset_s=offset * 0.05)
        # the three members of the triplet: same draw, different ingredients
        specs.append(dict(draw, category="body_only", has_body=True, has_push=False,
                          onset=body_onset))
        specs.append(dict(draw, category="push_only", has_body=False, has_push=True,
                          onset=body_onset + offset))
        specs.append(dict(draw, category="body_push", has_body=True, has_push=True,
                          onset=min(body_onset, body_onset + offset)))
    rng_u = np.random.default_rng(oc.SEED_BASE + 10_000)
    for i in range(oc.N_UNDISTURBED):
        onset = int(rng_u.integers(oc.BODY_ONSET_MIN, oc.BODY_ONSET_MAX + 1))
        specs.append(dict(triplet=-1, seed=oc.SEED_BASE + 10_000 + i, param="", factor=1.0,
                          torque=0.0, duration=0, body_onset=-1, push_onset=-1, offset_s=0.0,
                          category="undisturbed", has_body=False, has_push=False, onset=onset))
    return specs


# ---------------------------------------------------------------------------
# the episode loop
# ---------------------------------------------------------------------------

def run_overlap_episode(model, spec, steps=STEPS):
    """Same loop as run_experiment.run_episode, with a two-ingredient schedule.

    Returns per-step arrays: error, theta (wrapped), theta_dot.
    """
    env = env_setup.make_env(max_steps=steps)
    obs = env_setup.reset_env(env, spec["seed"])
    np.random.seed(spec["seed"])
    err, th, thd = [], [], []
    for t in range(steps):
        # schedule: body change (permanent) and/or push (on for `duration` steps)
        if spec["has_body"] and t == spec["body_onset"]:
            env.change_body({spec["param"]: spec["factor"]})
        if spec["has_push"]:
            if t == spec["push_onset"]:
                env.set_external_torque(spec["torque"])
            if t == spec["push_onset"] + spec["duration"]:
                env.set_external_torque(0.0)
        action = planner.choose_action(model, obs)
        with torch.no_grad():
            predicted = model.predict_next(torch.from_numpy(obs.astype(np.float32))[None],
                                           torch.from_numpy(action)[None])[0].numpy()
        next_obs, *_ = env.step(action)
        err.append(float(np.linalg.norm(next_obs - predicted)))
        theta, theta_dot = env.unwrapped.state
        th.append(float((theta + np.pi) % (2 * np.pi) - np.pi))
        thd.append(float(theta_dot))
        obs = next_obs
    env.close()
    return (np.array(err, dtype=np.float32), np.array(th, dtype=np.float32),
            np.array(thd, dtype=np.float32))


def run_trial(spec):
    """Worker entry point (trials._init_worker has loaded the frozen model)."""
    out = dict(spec)
    out["err"], out["theta"], out["thdot"] = run_overlap_episode(trials._MODEL, spec)
    return out


def run(specs):
    return trials.run_all(specs, "overlap", fn=run_trial)


def verify_matches_existing(model, steps=120):
    """Body-only and push-only through this loop == run_experiment.run_episode.

    Guards against the new loop silently differing from the one every earlier
    result was produced with. Returns the largest absolute difference (0.0).
    """
    worst = 0.0
    base = dict(triplet=0, seed=12345, param="l", factor=1.4, torque=2.5, duration=8,
                body_onset=40, push_onset=45, offset_s=0.25)
    for has_body, has_push, cond, onset in (
            (True, False, {"kind": "body", "factors": {"l": 1.4}}, 40),
            (False, True, {"kind": "world", "torque": 2.5, "duration": 8}, 45)):
        spec = dict(base, has_body=has_body, has_push=has_push)
        e, th, w = run_overlap_episode(model, spec, steps)
        rows = run_experiment.run_episode(model, cond, seed=12345, steps=steps, onset=onset)
        ref = np.array([[r["surprise"], r["theta"], r["theta_dot"]] for r in rows], dtype=np.float32)
        worst = max(worst, float(np.abs(np.stack([e, th, w], 1) - ref).max()))
    return worst
