"""Run the frozen noisy policy through train / dev / test trials and save them.

    python rollouts.py train
    python rollouts.py dev
    python rollouts.py test      # refused unless prereg.json exists (frozen)
"""
import json
import os
import sys


def main():
    import numpy as np
    import common as C
    split = sys.argv[1]
    assert split in ("train", "dev", "test")
    if split == "test" and not os.path.exists(os.path.join(C.OUT, "prereg.json")):
        sys.exit("refusing to generate TEST rollouts: prereg.json not frozen yet (run freeze_prereg.py after review)")
    env = C.get_train_env()
    legs = C.leg_map(env.sys)
    pol = C.policy_fn(env)
    specs = C.trial_specs(split)
    print(f"{split}: {len(specs)} trials", flush=True)
    print(f"action noise {C.ACTION_NOISE}", flush=True)
    obs, act, base = C.run_trials(env, pol, legs, specs)
    np.savez_compressed(os.path.join(C.OUT, f"rollouts_{split}.npz"), obs=obs, act=act, base=base, fell=C.fell(obs),
                        fell_any=C.fell(obs, 0), action_noise=C.ACTION_NOISE)
    json.dump(specs, open(os.path.join(C.OUT, f"specs_{split}.json"), "w"))
    print(f"saved rollouts_{split}.npz  noise {C.ACTION_NOISE}  fell after onset {int(C.fell(obs).sum())}/{len(specs)}"
          f"  fell at any time {int(C.fell(obs, 0).sum())}/{len(specs)}")


if __name__ == "__main__":
    main()
