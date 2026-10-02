"""Run the three conditions, log the surprise signal, save CSVs, plot, summarise.

    python run_experiment.py            # uses the cached model if present
    python run_experiment.py --retrain  # train a fresh forward model first

Pipeline:
  1. get a frozen forward model trained on the healthy body   (model.py)
  2. for each condition, run one closed-loop MPC episode and log the one-step
     prediction error at every step                           (this file)
  3. write each episode to results/<condition>.csv            (this file)
  4. plot the three error series on one figure                (plot.py)
  5. compute and print the summary features                   (features.py)

THE SURPRISE SIGNAL
At step t the controller sees the real state s_t and picks torque a_t. Before
stepping, we ask the model what it expects: s_hat = f(s_t, a_t). Then the
real environment produces s_{t+1}. The surprise at step t is

        e_t = || s_{t+1} - s_hat ||_2

over the 3-D observation [cos th, sin th, thdot]. The model is always given
the TRUE current state, so errors never compound - e_t measures only how
wrong the model is about this single transition. A disturbance that starts at
ONSET_STEP acts on the transition ONSET_STEP -> ONSET_STEP+1, so e_{ONSET_STEP}
is the first affected value.

Note the thdot component dominates the norm (it is in rad/s, the other two are
bounded in [-1, 1]); that is fine here because every disturbance acts on
angular acceleration, but keep it in mind if you add others.
"""
import argparse
import csv
import os

import numpy as np
import torch

import config
import env_setup
import features
import model as model_lib
import planner
import plot

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# The three conditions. Each is a small schedule: what to do at which step.
# ---------------------------------------------------------------------------

CONDITIONS = {
    # Nothing happens. This is the model's noise floor on the exact trajectory
    # the controller follows.
    "baseline": {"kind": "none"},
    # At ONSET_STEP the body's parameters are scaled and stay that way.
    "body_change": {"kind": "body", "factors": config.BODY_CHANGE},
    # From ONSET_STEP for WORLD_DURATION steps an external torque acts,
    # then it is removed. The body is untouched throughout.
    "world_event": {"kind": "world", "torque": config.WORLD_TORQUE,
                    "duration": config.WORLD_DURATION},
}


def apply_schedule(env, cond, t, onset=config.ONSET_STEP):
    """Trigger / end the disturbance for condition `cond` at step t.

    Returns a short string describing what changed this step (for logging).
    `onset` defaults to the fixed ONSET_STEP; the classifier evaluation passes
    a randomised onset per trial.
    """
    if cond["kind"] == "body" and t == onset:
        env.change_body(cond["factors"])
        return "body changed"
    if cond["kind"] == "world":
        if t == onset:
            env.set_external_torque(cond["torque"])
            return "push on"
        if t == onset + cond["duration"]:
            env.set_external_torque(0.0)
            return "push off"
    return ""


def run_episode(model, cond, seed=config.SEED, steps=config.EPISODE_STEPS,
                onset=config.ONSET_STEP):
    """One closed-loop episode. Returns a list of per-step log rows."""
    env = env_setup.make_env(max_steps=steps)
    obs = env_setup.reset_env(env, seed)
    # Seed the planner's RNG identically for every condition: combined with the
    # identical start state, the three episodes are the same trajectory until
    # the disturbance arrives.
    np.random.seed(seed)

    rows = []
    for t in range(steps):
        event = apply_schedule(env, cond, t, onset)

        action = planner.choose_action(model, obs)
        with torch.no_grad():
            predicted = model.predict_next(
                torch.from_numpy(obs.astype(np.float32))[None],
                torch.from_numpy(action)[None],
            )[0].numpy()
        next_obs, *_ = env.step(action)

        surprise = float(np.linalg.norm(next_obs - predicted))
        theta, theta_dot = env.unwrapped.state
        rows.append({
            "step": t,
            "time_s": round(t * env.unwrapped.dt, 4),
            "surprise": surprise,
            # the per-dimension residual, in case you want to look at direction
            "res_cos": float(next_obs[0] - predicted[0]),
            "res_sin": float(next_obs[1] - predicted[1]),
            "res_thdot": float(next_obs[2] - predicted[2]),
            "action": float(action[0]),
            "ext_torque": env.ext_torque,
            "length": env.unwrapped.l,
            "mass": env.unwrapped.m,
            # angle AFTER the step, wrapped to [-pi, pi]; 0 = upright
            "theta": float((theta + np.pi) % (2 * np.pi) - np.pi),
            "theta_dot": float(theta_dot),
            "event": event,
        })
        obs = next_obs
    env.close()
    return rows


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--retrain", action="store_true", help="train a fresh forward model")
    args = ap.parse_args()

    torch.set_num_threads(2)
    out_dir = os.path.join(HERE, config.OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    # 0. sanity check: our push-capable physics matches stock Pendulum-v1
    worst = env_setup.verify_physics()
    assert worst == 0.0, f"physics re-implementation differs from Pendulum-v1 by {worst}"
    print("physics check: push-capable step == stock Pendulum-v1 when push is 0")

    # 1. the frozen self-model
    print("\n[1] forward model")
    model = model_lib.get_model(os.path.join(HERE, config.MODEL_PATH), args.retrain)
    print(f"  held-out one-step error on random healthy-body data: "
          f"{model_lib.held_out_error(model, seed=config.SEED + 1):.2e}")

    # 2-3. run and save each condition
    print(f"\n[2] running {len(CONDITIONS)} conditions, {config.EPISODE_STEPS} steps each, "
          f"disturbance at step {config.ONSET_STEP}")
    series = {}
    for name, cond in CONDITIONS.items():
        rows = run_episode(model, cond)
        path = os.path.join(out_dir, f"{name}.csv")
        save_csv(rows, path)
        series[name] = np.array([r["surprise"] for r in rows])
        final_theta = np.abs([r["theta"] for r in rows[-20:]]).mean()
        print(f"  {name:<12} -> {os.path.relpath(path, HERE)}   "
              f"(mean |theta| over last 20 steps: {final_theta:.3f} rad)")

    # 4. figure
    print("\n[3] plotting")
    fig_path = plot.plot_surprise(series, os.path.join(out_dir, "surprise_signatures.png"))
    print(f"  -> {os.path.relpath(fig_path, HERE)}")

    # 5. summary features
    summaries = {name: features.summarise(e) for name, e in series.items()}
    features.print_table(summaries)
    with open(os.path.join(out_dir, "summary_features.csv"), "w", newline="") as f:
        w = csv.writer(f)
        keys = list(next(iter(summaries.values())).keys())
        w.writerow(["condition"] + keys)
        for name, s in summaries.items():
            w.writerow([name] + [s[k] for k in keys])
    print(f"  -> {config.OUT_DIR}/summary_features.csv")


if __name__ == "__main__":
    main()
