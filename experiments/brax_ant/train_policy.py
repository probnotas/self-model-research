"""Train the PPO walking controller (standard Brax Ant hyperparameters), save it, check the reward.

    python train_policy.py            # SCALE / COMPLIANT / OUT from the environment

Exits with code 2 if the full-scale eval reward is below REWARD_TARGET (stop and report).
"""
import json
import os
import sys
import time


def main():
    import common as C
    from brax.training.agents.ppo import train as ppo
    rep_path = os.path.join(C.OUT, "policy_report.json")
    if os.path.exists(os.path.join(C.OUT, "policy_params.pkl")) and os.path.exists(rep_path) and os.environ.get("RETRAIN") != "1":
        r = json.load(open(rep_path))["final_eval_reward"]
        if C.REWARD_TARGET is None or r >= C.REWARD_TARGET:
            print(f"policy already trained (eval reward {r:.1f}); skipping PPO. Set RETRAIN=1 to retrain.")
            return
    env = C.get_train_env()
    log = []

    def progress(step, metrics):
        r = float(metrics.get("eval/episode_reward", float("nan")))
        log.append(dict(step=int(step), eval_reward=r, t=time.time()))
        print(f"  step {step:>10}  eval reward {r:9.1f}", flush=True)

    t0 = time.time()
    make_inf, params, _ = ppo.train(environment=env, progress_fn=progress, **C.PPO)
    C.save(params, "policy_params.pkl")
    final = log[-1]["eval_reward"] if log else float("nan")
    rep = dict(scale=C.SCALE, compliant=C.COMPLIANT, ppo=C.PPO, final_eval_reward=final,
               reward_target=C.REWARD_TARGET, curve=log, train_seconds=time.time() - t0)
    json.dump(rep, open(os.path.join(C.OUT, "policy_report.json"), "w"), indent=1)
    print(f"final eval reward {final:.1f} (target {C.REWARD_TARGET})")
    if C.REWARD_TARGET is not None and not final >= C.REWARD_TARGET:
        print("REWARD BELOW TARGET: stop here and report (do not continue to rollouts).")
        sys.exit(2)


if __name__ == "__main__":
    main()
