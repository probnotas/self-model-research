"""JEPA follow-up: persistence, push recovery, raw-space ablation, 3 seeds. Rules: PREREG.md.

    python experiments/jepa/run_followup.py

Train on healthy seed-90000 recordings; score the seed-777000 trials. Nothing is
simulated here. SIMULATED compliant body, not hardware.
"""
import copy
import json
import os
import platform
import sys

import numpy as np
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view as swv

HERE = os.path.dirname(os.path.abspath(__file__))
SA = os.path.join(HERE, "..", "..", "surprise_attribution")
sys.path.insert(0, SA)
import factorial_config as fc  # noqa: E402

K, D = 10, 8
EARLY, LATE = (20, 100), (100, 200)
SEEDS = (0, 1, 2)
CONFIGS = ("soft_none", "soft_low", "rigid_none")
TARGETS = ("soft_none", "soft_low")
EPOCHS, BATCH, LR, TAU = 40, 512, 1e-3, 0.99
torch.set_num_threads(4)


def specs_for(seed):
    fc.MASTER_SEED = seed
    import importlib
    import run_factorial as rf
    importlib.reload(rf)
    return rf.make_specs()


def feats(obs, act):
    return np.concatenate([obs[:, :-1, :], act], axis=2).astype(np.float32)        # (n, T, 4)


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, o))


class JEPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc, self.pred = mlp(4 * K, D), mlp(D + K, D)
        self.tgt = copy.deepcopy(self.enc)
        for p in self.tgt.parameters():
            p.requires_grad_(False)

    def err(self, X, Y):                       # per-sample squared error of the predicted summary
        return ((self.pred(torch.cat([self.enc(X.flatten(1)), Y[:, :, 3]], 1)) - self.tgt(Y.flatten(1))) ** 2).mean(1)

    def ema(self):
        with torch.no_grad():
            for pt, pe in zip(self.tgt.parameters(), self.enc.parameters()):
                pt.mul_(TAU).add_((1 - TAU) * pe)


class Raw(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = mlp(4 * K + K, 3 * K)

    def err(self, X, Y):
        return ((self.f(torch.cat([X.flatten(1), Y[:, :, 3]], 1)) - Y[:, :, :3].flatten(1)) ** 2).mean(1)


def train_pairs(F, specs):
    ranges = [(i, s["onset"]) for i, s in enumerate(specs) if s["cell"] != "calib"] + \
             [(i, F.shape[1]) for i, s in enumerate(specs) if s["cell"] == "calib"]
    X, Y = [], []
    for i, b in ranges:
        for t in range(K - 1, b - K):
            X.append(F[i, t - K + 1:t + 1]); Y.append(F[i, t + 1:t + K + 1])
    return torch.tensor(np.array(X)), torch.tensor(np.array(Y))


def train(kind, X, Y, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=g); hold, tr = perm[:2000], perm[2000:]
    m = JEPA() if kind == "jepa" else Raw()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    info = {}
    if kind == "jepa":
        with torch.no_grad():
            info["std_init"] = m.tgt(Y[hold].flatten(1)).std(0).mean().item()
    for _ in range(EPOCHS):
        for b in tr[torch.randperm(len(tr), generator=g)].split(BATCH):
            loss = m.err(X[b], Y[b]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if kind == "jepa":
                m.ema()
    m.eval()
    with torch.no_grad():
        info["heldout_loss"] = m.err(X[hold], Y[hold]).mean().item()
        if kind == "jepa":
            info["std_final"] = m.tgt(Y[hold].flatten(1)).std(0).mean().item()
            info["valid"] = info["std_final"] >= 0.05 * info["std_init"]
    return m, info


def step_errors(m, F):
    """err[i, t] for context ending at t (t = K-1 .. T-K-1); NaN elsewhere."""
    n, T, _ = F.shape
    W = swv(F, K, axis=1).transpose(0, 1, 3, 2)              # (n, T-K+1, K, 4), W[:, s] = F[:, s:s+K]
    ts = np.arange(K - 1, T - K)
    X = W[:, ts - K + 1]; Y = W[:, ts + 1]
    out = np.full((n, T), np.nan, np.float64)
    with torch.no_grad():
        e = m.err(torch.tensor(X.reshape(-1, K, 4).copy()), torch.tensor(Y.reshape(-1, K, 4).copy()))
    out[:, ts] = e.numpy().reshape(n, len(ts))
    return out


def win(arr, specs, w):
    return np.array([np.nanmean(arr[i, s["onset"] + w[0]:s["onset"] + w[1]]) for i, s in enumerate(specs)])


def auc(p, n):
    p, n = np.asarray(p), np.asarray(n)
    return float((p[:, None] > n[None]).mean() + 0.5 * (p[:, None] == n[None]).mean())


def recovery(obs, specs):
    th = np.arctan2(obs[:, :, 1], obs[:, :, 0]); w = obs[:, :, 2]
    calm = (np.abs(th) < 0.3) & (np.abs(w) < 0.5)
    out = {}
    for i, s in enumerate(specs):
        if s["cell"] != "world_only":
            continue
        end, run, rec = s["push_onset"] + s["duration"], 0, None
        for t in range(end, obs.shape[1]):
            run = run + 1 if calm[i, t] else 0
            if run > 10:
                rec = (t - run + 1 - end) * 0.05; break
        out[i] = rec
    return out


def board(score, cell, thr_src, motion):
    thr = thr_src.mean() + 2 * thr_src.std(ddof=1)
    bo, ne, wo = score[cell == "body_only"], score[cell == "neither"], score[cell == "world_only"]
    m = (cell == "neither") | (cell == "body_only")
    return dict(auc_body_undist=auc(bo, ne), auc_body_push=auc(bo, wo),
                med_undist=float(np.median(ne)), med_body=float(np.median(bo)), med_push=float(np.median(wo)),
                ratio_body_undist=float(np.median(bo) / np.median(ne)),
                caught_body_only=int((bo >= thr).sum()), fp_push=int((wo >= thr).sum()),
                undist_flagged=int((ne >= thr).sum()),
                r_motion=float(np.corrcoef(score[m], motion[m])[0, 1]),
                r_motion_undist=float(np.corrcoef(ne, motion[cell == "neither"])[0, 1])), thr


def main():
    s90, s77 = specs_for(90000), specs_for(777000)
    cell = np.array([s["cell"] for s in s77]); is_cal = cell == "calib"
    res = {"versions": dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                            train_seed_master=90000, test_seed_master=777000, seeds=list(SEEDS))}
    for cfg in CONFIGS:
        d = np.load(os.path.join(SA, "results_history", "episodes", f"{cfg}.npz"))
        X, Y = train_pairs(feats(d["obs"], d["act"]), s90)
        d2 = np.load(os.path.join(SA, "results_replication", "episodes", f"{cfg}.npz"))
        F2 = feats(d2["obs"], d2["act"])
        mot_step = np.linalg.norm(d2["obs"][:, 1:] - d2["obs"][:, :-1], axis=2)
        motion = {"early": win(mot_step, s77, EARLY), "late": win(mot_step, s77, LATE)}
        rec = recovery(d2["obs"], s77)
        raw1 = win(d2["err"].astype(np.float64), s77, EARLY)
        rb = json.load(open(os.path.join(SA, "results_replication", f"{cfg}.json")))["rules"]["B_motion_signed"]
        res[cfg] = dict(n_train_pairs=len(X),
                        recovery={"n_push": len(rec), "recovered": sum(v is not None for v in rec.values()),
                                  "median_time_s": float(np.median([v for v in rec.values() if v is not None]))
                                  if any(v is not None for v in rec.values()) else None,
                                  "by_combo": {}},
                        context=dict(raw_one_step_early=board(raw1, cell, raw1[is_cal], motion["early"])[0],
                                     rung6_B=dict(auc_body_undist=rb["auc_body_vs_undisturbed"],
                                                  auc_body_push=rb["auc_body_vs_push"],
                                                  caught_body_only=rb["detect_body_only"]["k"],
                                                  fp_push=rb["false_positive"]["k"])),
                        seeds={})
        push_idx = sorted(rec)
        for tq in fc.PUSH_TORQUES:
            for dn in fc.PUSH_DURATIONS:
                ids = [i for i in push_idx if abs(s77[i]["torque"]) == tq and s77[i]["duration_name"] == dn]
                r = [rec[i] for i in ids]
                res[cfg]["recovery"]["by_combo"][f"{tq:g}Nm_{dn}"] = dict(
                    n=len(ids), recovered=sum(v is not None for v in r),
                    median_time_s=float(np.median([v for v in r if v is not None])) if any(v is not None for v in r) else None)
        for seed in SEEDS:
            out = {}
            for kind in ("jepa", "raw"):
                m, info = train(kind, X, Y, seed)
                E = step_errors(m, F2)
                sc = {"early": win(E, s77, EARLY), "late": win(E, s77, LATE)}
                sc["ratio"] = sc["late"] / sc["early"]
                o = {"train": info}
                for k in ("early", "late"):
                    o[k], thr = board(sc[k], cell, sc[k][is_cal], motion[k])
                    if kind == "jepa" and k == "late":
                        fp = {i: sc["late"][i] >= thr for i in push_idx}
                        o["late_fp_by_torque"] = {f"{tq:g}": int(sum(fp[i] for i in push_idx if abs(s77[i]["torque"]) == tq))
                                                  for tq in fc.PUSH_TORQUES}
                        o["late_fp_by_duration"] = {dn: int(sum(fp[i] for i in push_idx if s77[i]["duration_name"] == dn))
                                                    for dn in fc.PUSH_DURATIONS}
                        o["late_fp_by_recovery"] = {
                            "recovered": [int(sum(fp[i] for i in push_idx if rec[i] is not None)),
                                          sum(rec[i] is not None for i in push_idx)],
                            "not_recovered": [int(sum(fp[i] for i in push_idx if rec[i] is None)),
                                              sum(rec[i] is None for i in push_idx)]}
                o["ratio_auc_body_push"] = auc(sc["ratio"][cell == "body_only"], sc["ratio"][cell == "world_only"])
                o["ratio_auc_body_undist"] = auc(sc["ratio"][cell == "body_only"], sc["ratio"][cell == "neither"])
                out[kind] = o
                print(f"{cfg} seed {seed} {kind}: early AUC b/u {o['early']['auc_body_undist']:.3f} b/p {o['early']['auc_body_push']:.3f} | "
                      f"late AUC b/u {o['late']['auc_body_undist']:.3f} b/p {o['late']['auc_body_push']:.3f} FP {o['late']['fp_push']} "
                      f"caught {o['late']['caught_body_only']} | ratio b/p {o['ratio_auc_body_push']:.3f} | {info}", flush=True)
            res[cfg]["seeds"][seed] = out
    # ---- pre-registered criteria
    crit = {}
    for seed in SEEDS:
        P = all(res[c]["seeds"][seed]["jepa"]["late"]["auc_body_push"] >= 0.90 and
                res[c]["seeds"][seed]["jepa"]["late"]["fp_push"] <= 10 for c in TARGETS)
        Lc = []
        for c in TARGETS:
            a, j = res[c]["seeds"][seed]["raw"]["early"], res[c]["seeds"][seed]["jepa"]["early"]
            Lc.append(a["auc_body_undist"] < 0.60 or (a["ratio_body_undist"] - 1) <= 0.25 * (j["ratio_body_undist"] - 1))
        valid = all(res[c]["seeds"][seed]["jepa"]["train"]["valid"] for c in CONFIGS)
        crit[seed] = dict(valid=valid, P=P and valid, L=all(Lc) and valid)
    res["criteria"] = dict(per_seed=crit, P_holds=all(v["P"] for v in crit.values()),
                           L_holds=all(v["L"] for v in crit.values()))
    print(json.dumps(res["criteria"], indent=1))
    json.dump(res, open(os.path.join(HERE, "followup_results.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
