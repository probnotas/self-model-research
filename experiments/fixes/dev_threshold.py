"""DEV ONLY: pick the gain threshold on the already-used seed-777000 trials (not the final test)."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("../jepa", "../inverse"): sys.path.insert(0, os.path.join(HERE, p))
import run_followup as J, run_inverse as I
s90, s77 = J.specs_for(90000), J.specs_for(777000)
cell = np.array([s["cell"] for s in s77]); cal = cell == "calib"
for cfg in ("soft_none", "soft_low", "rigid_none", "medium_none"):
    d90 = np.load(os.path.join(J.SA, "results_history", "episodes", f"{cfg}.npz"))
    X, Y = I.train_set(d90["obs"], d90["act"], s90)
    d = np.load(os.path.join(J.SA, "results_replication", "episodes", f"{cfg}.npz"))
    th = np.abs(np.arctan2(d["obs"][:, :, 1], d["obs"][:, :, 0]))
    fell = np.array([th[i, s["onset"]+180:s["onset"]+200].max() >= np.pi/2 for i, s in enumerate(s77)])
    m, _ = I.train(X, Y, 0); P = I.per_step(m, d["obs"], d["act"])
    g = I.wsum(P["aha"], s77, (100, 200)) / I.wsum(P["aa"], s77, (100, 200)); lg = np.log(np.clip(g, 1e-6, None))
    z = (lg - lg[cal].mean()) / lg[cal].std(ddof=1)
    for T in (2.0, 2.5, 3.0, 3.5):
        b = (np.abs(z) >= T) & ~fell
        print(cfg, f"|z|>={T}:", {c: int(b[cell == c].sum()) for c in ("neither", "body_only", "world_only")})
