"""DEV ONLY: does the probe separate fallen-push from fallen-body? Small batch, seed 555000 (never a test seed)."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import sim_probe as SP
sys.path.insert(0, os.path.join(HERE, "../jepa")); import run_followup as J
sys.path.insert(0, SP.SA); import compliant_config as kc, run_history as rh
specs = J.specs_for(555000)
pick = []
for c in ("neither", "body_only", "world_only"):
    pick += [s for s in specs if s["cell"] == c][:30]
for name, (k, n) in {"soft_none": (4.0, "none"), "medium_none": (30.0, "none"), "rigid_none": (None, "none")}.items():
    cfg = dict(stiffness=k, noise=kc.NOISE[n]); mp = rh.single_model_path(name, cfg)
    sp = [dict(s, cfg=cfg, model_path=mp) for s in pick]
    hang = [dict(seed=555900 + i, cfg=cfg, model_path=mp, hanging=True, onset=0, cell="hang") for i in range(30)]
    res = SP.run_all(sp + hang)
    amp = np.array([SP.probe_amplitude(r["obs"], r["T_ctrl"]) for r in res])
    th = [np.abs(np.arctan2(r["obs"][:, 1], r["obs"][:, 0])) for r in res]
    fell = np.array([r["cell"] == "hang" or th[i][r["onset"]+180:r["onset"]+200].max() >= np.pi/2 for i, r in enumerate(res)])
    cells = np.array([r["cell"] for r in res]); la = np.log(amp)
    mu, sd = la[cells == "hang"].mean(), la[cells == "hang"].std(ddof=1)
    for c in ("hang", "neither", "body_only", "world_only"):
        sel = (cells == c) & fell
        print(name, c, "fell", int(sel.sum()), "/", int((cells == c).sum()), "probe z:", np.round(np.sort((la[sel]-mu)/sd), 1)[:12] if sel.any() else "-")
