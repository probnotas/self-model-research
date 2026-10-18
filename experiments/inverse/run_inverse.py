"""Inverse-dynamics mismatch and command-gain as body-change detectors. Rules: PREREG.md.

    python experiments/inverse/run_inverse.py

Train on healthy seed-90000 recordings; score the seed-777000 trials. Nothing
is simulated here. SIMULATED compliant body, not hardware.
"""
import json
import os
import platform
import sys

import numpy as np
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view as swv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "jepa"))
import run_followup as J  # noqa: E402  (specs_for, recovery, auc, win, SA)

K = 10
EARLY, LATE = (20, 100), (100, 200)
SEEDS = (0, 1, 2)
CONFIGS = ("soft_none", "soft_low", "rigid_none")
TARGETS = ("soft_none", "soft_low")
torch.set_num_threads(4)


def windows(obs, act, ts):
    """States s_{t-K+1}..s_{t+K} (2K states) and actions a_t..a_{t+K-1}, for each t in ts."""
    S = swv(obs, 2 * K, axis=1).transpose(0, 1, 3, 2)          # S[:, j] = obs[:, j:j+2K]
    A = swv(act[..., 0], K, axis=1)                           # A[:, j] = act[:, j:j+K]
    return S[:, ts - K + 1], A[:, ts]


def train(X, Y, seed):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    m = nn.Sequential(nn.Linear(6 * K, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, K))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    perm = torch.randperm(len(X), generator=g); hold, tr = perm[:2000], perm[2000:]
    for _ in range(40):
        for b in tr[torch.randperm(len(tr), generator=g)].split(512):
            loss = ((m(X[b]) - Y[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return m, float(((m(X[hold]) - Y[hold]) ** 2).mean())


def train_set(obs, act, specs):
    Xs, Ys = [], []
    for i, s in enumerate(specs):
        end = s["onset"] if s["cell"] != "calib" else act.shape[1]
        ts = np.arange(K - 1, end - K)                          # needs states up to t+K <= end
        if len(ts) == 0:
            continue
        S, A = windows(obs[i:i + 1], act[i:i + 1], ts)
        Xs.append(S[0].reshape(len(ts), -1)); Ys.append(A[0])
    return torch.tensor(np.concatenate(Xs), dtype=torch.float32), torch.tensor(np.concatenate(Ys), dtype=torch.float32)


def per_step(m, obs, act):
    """mismatch[i,t], ahat.a[i,t], a.a[i,t] for t = K-1 .. T-K-1."""
    n, T = act.shape[0], act.shape[1]
    ts = np.arange(K - 1, T - K)
    S, A = windows(obs, act, ts)
    with torch.no_grad():
        ah = m(torch.tensor(S.reshape(n * len(ts), -1).copy(), dtype=torch.float32)).numpy().reshape(n, len(ts), K)
    out = {k: np.full((n, T), np.nan) for k in ("mis", "aha", "aa")}
    out["mis"][:, ts] = np.abs(ah - A).mean(2)
    out["aha"][:, ts] = (ah * A).sum(2)
    out["aa"][:, ts] = (A * A).sum(2)
    return out


def wsum(arr, specs, w):
    return np.array([np.nansum(arr[i, s["onset"] + w[0]:s["onset"] + w[1]]) for i, s in enumerate(specs)])


def evaluate(score, cell, thr, motion):
    bo, ne, wo = score[cell == "body_only"], score[cell == "neither"], score[cell == "world_only"]
    m = (cell == "neither") | (cell == "body_only")
    return dict(auc_body_undist=J.auc(bo, ne), auc_body_push=J.auc(bo, wo),
                caught_body_only=int((bo >= thr).sum()), fp_push=int((wo >= thr).sum()),
                undist_flagged=int((ne >= thr).sum()),
                r_motion=float(np.corrcoef(score[m], motion[m])[0, 1]),
                r_motion_undist=float(np.corrcoef(ne, motion[cell == "neither"])[0, 1]))


def main():
    s90, s77 = J.specs_for(90000), J.specs_for(777000)
    cell = np.array([s["cell"] for s in s77]); cal = cell == "calib"
    push_idx = np.where(cell == "world_only")[0]
    res = {"versions": dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                            train_master_seed=90000, test_master_seed=777000, seeds=list(SEEDS))}
    for cfg in CONFIGS:
        d = np.load(os.path.join(J.SA, "results_history", "episodes", f"{cfg}.npz"))
        X, Y = train_set(d["obs"], d["act"], s90)
        d2 = np.load(os.path.join(J.SA, "results_replication", "episodes", f"{cfg}.npz"))
        obs2, act2 = d2["obs"], d2["act"]
        mot = np.linalg.norm(obs2[:, 1:] - obs2[:, :-1], axis=2)
        motion = {w: J.win(mot, s77, W) for w, W in (("early", EARLY), ("late", LATE))}
        rec = J.recovery(obs2, s77)
        res[cfg] = {"n_train": len(X), "seeds": {}}
        for seed in SEEDS:
            m, hl = train(X, Y, seed)
            P = per_step(m, obs2, act2)
            o = {"heldout_mse": hl}
            for w, W in (("early", EARLY), ("late", LATE)):
                M = J.win(P["mis"], s77, W)
                g = wsum(P["aha"], s77, W) / wsum(P["aa"], s77, W)
                lg = np.log(np.clip(g, 1e-6, None))
                z = (lg - lg[cal].mean()) / lg[cal].std(ddof=1)
                thrM = M[cal].mean() + 2 * M[cal].std(ddof=1)
                eM, eG = evaluate(M, cell, thrM, motion[w]), evaluate(np.abs(z), cell, 2.0, motion[w])
                eG["median_z"] = {c: float(np.median(z[cell == c])) for c in ("neither", "body_only", "world_only", "body_world")}
                eG["median_gain"] = {c: float(np.median(g[cell == c])) for c in ("neither", "body_only", "world_only")}
                for e, flag in ((eM, M >= thrM), (eG, np.abs(z) >= 2)):
                    e["fp_by_recovery"] = {"recovered": [int(sum(flag[i] for i in push_idx if rec[i] is not None)),
                                                         int(sum(rec[i] is not None for i in push_idx))],
                                           "not_recovered": [int(sum(flag[i] for i in push_idx if rec[i] is None)),
                                                             int(sum(rec[i] is None for i in push_idx))]}
                o[w] = {"M": eM, "G": eG}
            res[cfg]["seeds"][seed] = o
            e = o["early"]
            print(f"{cfg} seed {seed}: M AUC b/u {e['M']['auc_body_undist']:.3f} b/p {e['M']['auc_body_push']:.3f} FP {e['M']['fp_push']} caught {e['M']['caught_body_only']} | "
                  f"G AUC b/u {e['G']['auc_body_undist']:.3f} b/p {e['G']['auc_body_push']:.3f} FP {e['G']['fp_push']} caught {e['G']['caught_body_only']} "
                  f"med z u/b/p {e['G']['median_z']['neither']:+.2f}/{e['G']['median_z']['body_only']:+.2f}/{e['G']['median_z']['world_only']:+.2f}", flush=True)
    crit = {}
    for seed in SEEDS:
        ok = lambda f: all(f(res[c]["seeds"][seed]["early"]) for c in TARGETS)
        crit[seed] = dict(
            M1=ok(lambda e: e["M"]["auc_body_undist"] >= 0.75),
            M2=ok(lambda e: e["M"]["auc_body_push"] >= 0.75 and e["M"]["fp_push"] <= 15),
            G=ok(lambda e: e["G"]["auc_body_undist"] >= 0.75 and e["G"]["auc_body_push"] >= 0.75 and e["G"]["fp_push"] <= 15))
    res["criteria"] = dict(per_seed=crit, **{k: all(crit[s][k] for s in SEEDS) for k in ("M1", "M2", "G")})
    print(json.dumps(res["criteria"], indent=1))
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
