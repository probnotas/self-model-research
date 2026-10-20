"""Score a split with the self-models (3 seeds each) and test the hypotheses.

    python evaluate.py dev
    python evaluate.py test      # refused unless prereg.json exists; stores its sha256

AUCs are reported RAW (positive class first; < 0.5 means the signal points backwards).
"""
import hashlib
import json
import os
import sys


def auc(pos, neg):
    import numpy as np
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


def main():
    import numpy as np
    import common as C
    import selfmodels as SM
    split = sys.argv[1]
    assert split in ("dev", "test")
    prereg_path = os.path.join(C.OUT, "prereg.json")
    prereg_sha = None
    if split == "test":
        if not os.path.exists(prereg_path):
            sys.exit("refusing to score TEST: prereg.json not frozen")
        prereg_sha = hashlib.sha256(open(prereg_path, "rb").read()).hexdigest()

    tr = np.load(os.path.join(C.OUT, "rollouts_train.npz"))
    d = np.load(os.path.join(C.OUT, f"rollouts_{split}.npz"))
    specs = json.load(open(os.path.join(C.OUT, f"specs_{split}.json")))
    obs, act, fell = d["obs"], d["act"], d["fell"]
    ev = np.array([s["event"] for s in specs]); leg = np.array([s["leg"] for s in specs])
    is_body = np.isin(ev, C.BODY_EVENTS)
    legs = C.leg_map(C.get_train_env().sys)

    # model-free motion signal: joint-velocity variance late vs pre
    jv = obs[:, :, 19:27]
    motion = np.log(jv[:, C.LATE[0]:C.LATE[1]].var((1, 2)) / jv[:, C.PRE[0]:C.PRE[1]].var((1, 2)))

    ts = np.arange(C.PRE[0], C.LATE[1])
    w = lambda E, W: E[:, W[0] - ts[0]:W[1] - ts[0]].mean(1)       # mean over window of (n, dims)
    res = dict(split=split, scale=C.SCALE, compliant=C.COMPLIANT, prereg_sha256=prereg_sha, n=len(specs),
               fall_rate={e: float(fell[ev == e].mean()) for e in np.unique(ev)},
               policy=json.load(open(os.path.join(C.OUT, "policy_report.json")))["final_eval_reward"],
               motion=dict(auc_body_vs_none=auc(motion[is_body], motion[ev == C.NONE]),
                           auc_body_vs_push_short=auc(motion[is_body & ~fell], motion[(ev == "push_short") & ~fell])),
               seeds={})
    for seed in C.SM_SEEDS:
        out = {}
        for kind in ("one", "multi"):
            m = SM.get(kind, seed, tr["obs"], tr["act"])
            E = SM.errors(m, obs, act, ts)                          # (n, len(ts), 27)
            pre, early, late = w(E, C.PRE), w(E, C.EARLY), w(E, C.LATE)
            detect = np.log(early.mean(1) / pre.mean(1))
            persist = np.log(late.mean(1) / pre.mean(1))
            leg_ratio = np.stack([np.log(early[:, legs.obs[l]].mean(1) / pre[:, legs.obs[l]].mean(1)) for l in range(4)], 1)
            guess = leg_ratio.argmax(1)
            nf = ~fell
            o = dict(
                detect_auc_body_vs_none=auc(detect[is_body], detect[ev == C.NONE]),
                persist_auc_body_vs_push_short_nonfallen=auc(persist[is_body & nf], persist[(ev == "push_short") & nf]),
                leg_accuracy_body=float((guess[is_body] == leg[is_body]).mean()),
                exploratory=dict(
                    detect_auc_body_vs={e: auc(detect[is_body], detect[ev == e]) for e in C.WORLD_EVENTS},
                    persist_auc_body_vs_nonfallen={e: auc(persist[is_body & nf], persist[(ev == e) & nf]) for e in C.WORLD_EVENTS},
                    persist_auc_body_vs_none=auc(persist[is_body], persist[ev == C.NONE])),
                per_event={e: dict(n=int((ev == e).sum()), fall_rate=float(fell[ev == e].mean()),
                                   median_detect=float(np.median(detect[ev == e])),
                                   median_persist=float(np.median(persist[ev == e])),
                                   median_motion=float(np.median(motion[ev == e])),
                                   detect_auc_vs_none=auc(detect[ev == e], detect[ev == C.NONE]) if e != C.NONE else None,
                                   leg_accuracy=float((guess[ev == e] == leg[ev == e]).mean()) if e in C.BODY_EVENTS else None)
                           for e in np.unique(ev)})
            out[kind] = o
            print(f"  seed {seed} {kind:5s}: detect AUC body/none {o['detect_auc_body_vs_none']:.3f} | "
                  f"persist AUC body/short-push (non-fallen) {o['persist_auc_body_vs_push_short_nonfallen']} | "
                  f"leg acc {o['leg_accuracy_body']:.3f}", flush=True)
        mu, on = out["multi"], out["one"]
        out["hypotheses"] = dict(
            H1=mu["detect_auc_body_vs_none"] >= 0.90,
            H2=(mu["persist_auc_body_vs_push_short_nonfallen"] or 0) >= 0.80,
            H3=mu["leg_accuracy_body"] >= 0.50,
            H4=mu["detect_auc_body_vs_none"] > on["detect_auc_body_vs_none"])
        res["seeds"][seed] = out
    res["hypotheses_all_seeds"] = {h: all(res["seeds"][s]["hypotheses"][h] for s in C.SM_SEEDS) for h in ("H1", "H2", "H3", "H4")}
    res["push_short_fall_rate"] = res["fall_rate"].get("push_short")
    res["recommend_lower_push"] = (res["push_short_fall_rate"] or 0) > 0.20
    json.dump(res, open(os.path.join(C.OUT, f"{split}_results.json"), "w"), indent=1, default=str)
    print(json.dumps(dict(hypotheses_all_seeds=res["hypotheses_all_seeds"], fall_rate=res["fall_rate"],
                          policy_reward=res["policy"], recommend_lower_push=res["recommend_lower_push"]), indent=1))


if __name__ == "__main__":
    main()
