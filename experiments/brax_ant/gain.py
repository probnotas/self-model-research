"""Command gain: the signal that passed the pre-registered soft-pendulum test, ported to the Ant.

    python gain.py dev
    python gain.py test      # refused unless prereg.json exists; stores its sha256

Idea. The frozen policy asks for base_t; the motors get a_t = base_t + small random noise.
That noise is a free, continuous self-probe: an inverse model trained on healthy walking
learns how much each motor's nudge r_t = a_t - base_t shows up in the next state. Per leg,

    gain_l(W) = sum_{t in W, j in leg l} r_hat_tj * r_tj  /  sum r_tj^2

is ~1 when "my nudges move this leg the way they used to". A weak motor, a heavier leg or a
damped joint make that leg's nudges move it less, so its gain should drop and stay down.
A push on the torso changes the world, not how a leg answers its own motor, so the gain
should not move much. The leg with the biggest change is the leg guess.

Score per trial: d_l = log gain_l(LATE) - log gain_l(PRE), z-scored per leg against healthy
TRAIN trials held out from the inverse-model fit (no dev labels used). score = max_l |z_l|.
Trials that fell are reported separately: a flipped Ant's gain says nothing about its legs.

DEV ONLY until frozen. AUCs raw. Inverse model: MLP 3x512 (same trainer as the self-models).
"""
import hashlib
import json
import os
import sys

FIT_FRAC = 0.75                               # first 75% of train trials fit the inverse model; the rest calibrate
THRESHOLDS = (2.0, 2.5, 3.0)


def main():
    import numpy as np
    import common as C
    import selfmodels as SM
    from evaluate import auc
    split = sys.argv[1]
    assert split in ("dev", "test")
    prereg_sha = None
    if split == "test":
        p = os.path.join(C.OUT, "prereg.json")
        if not os.path.exists(p):
            sys.exit("refusing to score TEST: prereg.json not frozen")
        prereg_sha = hashlib.sha256(open(p, "rb").read()).hexdigest()

    tr = np.load(os.path.join(C.OUT, "rollouts_train.npz"))
    d = np.load(os.path.join(C.OUT, f"rollouts_{split}.npz"))
    for name, z in (("train", tr), (split, d)):
        if "base" not in z.files or float(z["action_noise"]) != C.ACTION_NOISE:
            sys.exit(f"rollouts_{name}.npz lacks base actions or has a different action noise; regenerate it")
    specs = json.load(open(os.path.join(C.OUT, f"specs_{split}.json")))
    ev = np.array([s["event"] for s in specs]); leg = np.array([s["leg"] for s in specs])
    is_body = np.isin(ev, C.BODY_EVENTS)
    fell = d["fell"]; up = ~fell
    legs = C.leg_map(C.get_train_env().sys)
    T = C.T_TOTAL

    def xy(z):
        """X = (s_t, base_t, s_{t+1}), Y = r_t, for every t: shapes (n, T, .)."""
        o, a, b = z["obs"], z["act"], z["base"]
        return np.concatenate([o[:, :T], b, o[:, 1:T + 1]], -1).astype(np.float32), (a - b).astype(np.float32)

    healthy = np.where(~tr["fell_any"])[0]                    # train Ants that never fell (whole trial)
    n_fit = int(FIT_FRAC * len(tr["obs"]))                    # 384 of 512 at full scale
    fit_i, cal_i = healthy[healthy < n_fit], healthy[healthy >= n_fit]
    if len(fit_i) < 2 or len(cal_i) < 2:
        sys.exit(f"too few healthy train trials to fit/calibrate ({len(fit_i)}/{len(cal_i)})")
    Xtr, Ytr = xy(tr)
    Xd, Yd = xy(d)

    def scores(params, norm, X, Y):
        R = SM.predict(params, norm, Y.shape[-1], X)
        g = lambda W, l: (R[:, W[0]:W[1]][..., legs.act[l]] * Y[:, W[0]:W[1]][..., legs.act[l]]).sum((1, 2)) / \
            np.maximum((Y[:, W[0]:W[1]][..., legs.act[l]] ** 2).sum((1, 2)), 1e-12)
        lg = lambda W: np.stack([np.log(np.clip(g(W, l), 1e-3, None)) for l in range(4)], 1)
        pre = lg(C.PRE)
        return dict(late=lg(C.LATE) - pre, early=lg(C.EARLY) - pre, pre_gain=np.exp(pre))

    res = dict(split=split, scale=C.SCALE, compliant=C.COMPLIANT, action_noise=C.ACTION_NOISE, prereg_sha256=prereg_sha,
               n=len(specs), n_fit=len(fit_i), n_calibrate=len(cal_i), n_fallen=int(fell.sum()),
               fallen_by_event={str(e): int(fell[ev == e].sum()) for e in np.unique(ev)}, seeds={})
    for seed in C.SM_SEEDS:
        path = os.path.join(C.OUT, f"inverse_s{seed}_noise{C.ACTION_NOISE}.pkl")
        if os.path.exists(path):
            params, norm = C.load(os.path.basename(path))
        else:
            X = Xtr[fit_i].reshape(-1, Xtr.shape[-1]); Y = Ytr[fit_i].reshape(-1, Ytr.shape[-1])
            params, norm = SM.fit_mlp(X, Y, seed, "inverse")
            C.save((params, norm), os.path.basename(path))
        cal, dv = scores(params, norm, Xtr[cal_i], Ytr[cal_i]), scores(params, norm, Xd, Yd)
        out = {"healthy_pre_gain_median": float(np.median(cal["pre_gain"]))}
        for win in ("late", "early"):
            mu, sd = cal[win].mean(0), cal[win].std(0, ddof=1)
            zz = (dv[win] - mu) / sd
            score, guess = np.abs(zz).max(1), np.abs(zz).argmax(1)
            own = zz[np.arange(len(zz)), np.clip(leg, 0, 3)]           # z of the leg that actually changed
            o = dict(
                auc_body_vs_none=auc(score[is_body & up], score[(ev == C.NONE) & up]),
                auc_body_vs={e: auc(score[is_body & up], score[(ev == e) & up]) for e in C.WORLD_EVENTS},
                leg_accuracy=float((guess[is_body & up] == leg[is_body & up]).mean()) if (is_body & up).any() else None,
                flagged_body={str(th): {e: [int((score[(ev == e) & up] >= th).sum()), int(((ev == e) & up).sum())]
                                        for e in np.unique(ev)} for th in THRESHOLDS},
                per_event={e: dict(n_upright=int(((ev == e) & up).sum()),
                                   median_score=float(np.median(score[(ev == e) & up])) if ((ev == e) & up).any() else None,
                                   auc_vs_none=auc(score[(ev == e) & up], score[(ev == C.NONE) & up]) if e != C.NONE else None,
                                   leg_accuracy=float((guess[(ev == e) & up] == leg[(ev == e) & up]).mean())
                                   if e in C.BODY_EVENTS and ((ev == e) & up).any() else None,
                                   median_own_leg_z=float(np.median(own[(ev == e) & up]))
                                   if e in C.BODY_EVENTS and ((ev == e) & up).any() else None)
                           for e in np.unique(ev)})
            out[win] = o
        L = out["late"]
        f = lambda x: "n/a" if x is None else f"{x:.3f}"
        print(f"  seed {seed} gain (late, upright): AUC body/none {f(L['auc_body_vs_none'])} | body/short push "
              f"{f(L['auc_body_vs']['push_short'])} | body/steady push {f(L['auc_body_vs']['push_sustained'])} | "
              f"body/payload {f(L['auc_body_vs']['payload'])} | leg acc {f(L['leg_accuracy'])}", flush=True)
        res["seeds"][seed] = out
    json.dump(res, open(os.path.join(C.OUT, f"gain_{split}_results.json"), "w"), indent=1, default=str)
    print(json.dumps({s: dict(by_event={e: v["median_own_leg_z"] for e, v in res["seeds"][s]["late"]["per_event"].items()
                                        if v["median_own_leg_z"] is not None},
                              healthy_pre_gain=res["seeds"][s]["healthy_pre_gain_median"]) for s in res["seeds"]}, indent=1))
    print(f"fallen (excluded from the gain scores): {res['fallen_by_event']}")


if __name__ == "__main__":
    main()
