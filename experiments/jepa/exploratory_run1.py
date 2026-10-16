import sys, json, copy, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/home/user/self-model-research/surprise_attribution")
import factorial_config as fc
R = "/home/user/self-model-research/surprise_attribution/"
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
K, D, W0, W1 = 10, 8, 20, 100

def specs_for(seed):
    fc.MASTER_SEED = seed
    import importlib, run_factorial as rf; importlib.reload(rf); return rf.make_specs()

def feats(obs, act):                      # per step t: [cos, sin, thdot] of obs[t], u[t]
    return np.concatenate([obs[:, :-1, :], act], axis=2).astype(np.float32)   # (n, T, 4)

class Enc(nn.Module):
    def __init__(s): super().__init__(); s.f = nn.Sequential(nn.Linear(4*K, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, D))
    def forward(s, x): return s.f(x.flatten(1))
class Pred(nn.Module):
    def __init__(s): super().__init__(); s.f = nn.Sequential(nn.Linear(D+K, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, D))
    def forward(s, z, u): return s.f(torch.cat([z, u], 1))

def pairs(F, ranges):
    """Context window t-K+1..t, target window t+1..t+K (commands of target window go to predictor)."""
    X, Y, U = [], [], []
    for i, a, b in ranges:                 # usable t with t-K+1 >= a and t+K <= b-1
        for t in range(a + K - 1, b - K):
            X.append(F[i, t-K+1:t+1]); Y.append(F[i, t+1:t+K+1]); U.append(F[i, t+1:t+K+1, 3])
    return torch.tensor(np.array(X)), torch.tensor(np.array(Y)), torch.tensor(np.array(U))

def train(F, ranges):
    X, Y, U = pairs(F, ranges)
    perm = torch.randperm(len(X)); hold = perm[:2000]; tr = perm[2000:]
    E, P = Enc(), Pred(); T = copy.deepcopy(E)
    for p in T.parameters(): p.requires_grad_(False)
    std0 = T(Y[hold]).std(0).mean().item()
    opt = torch.optim.Adam(list(E.parameters()) + list(P.parameters()), lr=1e-3)
    for ep in range(40):
        for b in tr[torch.randperm(len(tr))].split(512):
            with torch.no_grad(): tgt = T(Y[b])
            loss = ((P(E(X[b]), U[b]) - tgt) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for pt, pe in zip(T.parameters(), E.parameters()): pt.mul_(0.99).add_(0.01 * pe)
    with torch.no_grad():
        std1 = T(Y[hold]).std(0).mean().item()
        hl = ((P(E(X[hold]), U[hold]) - T(Y[hold])) ** 2).mean().item()
    return E, P, T, dict(n_pairs=len(X), std_init=std0, std_final=std1, valid=std1 >= 0.05*std0, heldout_loss=hl)

def trial_scores(E, P, T, F, specs):
    out = []
    with torch.no_grad():
        for i, s in enumerate(specs):
            ts = range(s["onset"] + W0, s["onset"] + W1)
            X = torch.tensor(np.array([F[i, t-K+1:t+1] for t in ts])); Y = torch.tensor(np.array([F[i, t+1:t+K+1] for t in ts]))
            out.append(((P(E(X), Y[:, :, 3]) - T(Y)) ** 2).mean(1).mean().item())
    return np.array(out)

def auc(p, n):
    p, n = np.asarray(p), np.asarray(n); return float((p[:, None] > n[None]).mean() + 0.5*(p[:, None] == n[None]).mean())

res = {}
for cfg in ("soft_none", "soft_low", "rigid_none"):
    s90 = specs_for(90000); d = np.load(R + f"results_history/episodes/{cfg}.npz"); F = feats(d["obs"], d["act"])
    ranges = [(i, 0, s["onset"]) for i, s in enumerate(s90) if s["cell"] != "calib"] + \
             [(i, 0, F.shape[1]) for i, s in enumerate(s90) if s["cell"] == "calib"]
    E, P, T, info = train(F, ranges)
    s77 = specs_for(777000); d2 = np.load(R + f"results_replication/episodes/{cfg}.npz"); F2 = feats(d2["obs"], d2["act"])
    sc = trial_scores(E, P, T, F2, s77)
    raw = np.array([d2["err"][i, s["onset"]+W0:s["onset"]+W1].mean() for i, s in enumerate(s77)])
    mot = np.linalg.norm(d2["obs"][:, 1:] - d2["obs"][:, :-1], axis=2)
    mot = np.array([mot[i, s["onset"]+W0:s["onset"]+W1].mean() for i, s in enumerate(s77)])
    cell = np.array([s["cell"] for s in s77])
    def board(x):
        thr = x[cell == "calib"].mean() + 2 * x[cell == "calib"].std(ddof=1)
        bo, ne, wo = x[cell == "body_only"], x[cell == "neither"], x[cell == "world_only"]
        m = (cell == "neither") | (cell == "body_only")
        return dict(med_undist=float(np.median(ne)), med_body=float(np.median(bo)), med_push=float(np.median(wo)),
                    auc_body_undist=auc(bo, ne), auc_body_push=auc(bo, wo),
                    detect_body_only=int((bo >= thr).sum()), fp_push=int((wo >= thr).sum()), undist_flag=int((ne >= thr).sum()),
                    r_motion=float(np.corrcoef(x[m], mot[m])[0, 1]),
                    r_motion_undist_only=float(np.corrcoef(x[cell == "neither"], mot[cell == "neither"])[0, 1]))
    rb = json.load(open(R + f"results_replication/{cfg}.json"))["rules"]["B_motion_signed"]
    res[cfg] = dict(train=info, jepa=board(sc), raw=board(raw),
                    rung6_B=dict(auc_body_undist=rb["auc_body_vs_undisturbed"], auc_body_push=rb["auc_body_vs_push"],
                                 detect_body_only=rb["detect_body_only"]["k"], fp_push=rb["false_positive"]["k"]))
    print(cfg, json.dumps(res[cfg], indent=1), flush=True)
json.dump(res, open("/tmp/claude-0/-home-user-self-model-research/63d918ac-a60f-5663-83c2-977bd2dbf5f5/scratchpad/jepa/results.json", "w"), indent=1)
