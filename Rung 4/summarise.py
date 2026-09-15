"""Freeze the Rung 4 operating point and export what the write-up needs."""
import json, os
import numpy as np

D = json.load(open("results_rung4.json"))
ONSET, W, N = D["onset"], D["window"], D["n_seeds"]
SIG = float(os.environ.get("SIGMA", 4.0))          # 4 sigma = zero false alarms
LAG = int(os.environ.get("LAG", 30))
BASE = "baseline (nothing happens)"
CONDS = D["conditions"]
truth = lambda k: "body" if k.startswith("body") else ("world" if k.startswith("world") else "none")

roll = lambda x: np.array([np.asarray(x, float)[max(0, i - W + 1):i + 1].mean()
                           for i in range(len(x))])
R = {k: np.array([roll(r["err"]) for r in v]) for k, v in CONDS.items()}
base = R[BASE][:, W:]
th_of = lambda i: (lambda p: p.mean() + SIG * p.std())(np.delete(base, i % len(base), axis=0))
THRESH = float(base.mean() + SIG * base.std())

rows, conf = {}, {"body": [0, 0], "world": [0, 0]}
for k, r in R.items():
    det, called = [], []
    for i in range(len(r)):
        th = th_of(i)
        w = np.where(r[i, ONSET:] > th)[0]
        det.append(bool(len(w)))
        if len(w):
            j = min(ONSET + int(w[0]) + LAG, len(r[i]) - 1)
            c = "body" if r[i, j] > th else "world"
            called.append(c)
            if truth(k) != "none":
                conf[truth(k)][0 if c == truth(k) else 1] += 1
    rows[k] = dict(
        truth=truth(k),
        peak=float(r[:, ONSET:ONSET + 15].max(axis=1).mean()),
        late=float(r[:, ONSET + 60].mean()),
        detected=float(np.mean(det)),
        right=(sum(1 for c in called if c == truth(k)) / len(called)) if called else None,
        n_det=int(sum(det)),
    )

sens = {}
for sg in (3, 4, 5, 6):
    t_ = lambda i: (lambda p: p.mean() + sg * p.std())(np.delete(base, i % len(base), axis=0))
    sens[sg] = {k: float(np.mean([(R[k][i, ONSET:] > t_(i)).any() for i in range(N)]))
                for k in R}

lags = {}
for lg in (10, 20, 30, 40, 60):
    ok = tot = 0
    for k, r in R.items():
        if truth(k) == "none":
            continue
        for i in range(len(r)):
            th = th_of(i)
            w = np.where(r[i, ONSET:] > th)[0]
            if not len(w):
                continue
            j = min(ONSET + int(w[0]) + LAG * 0 + lg, len(r[i]) - 1)
            tot += 1; ok += int(("body" if r[i, j] > th else "world") == truth(k))
    lags[lg] = ok / tot

coh = {}
for k, v in CONDS.items():
    res = np.array([r["res"] for r in v])
    f = lambda a, b: float(np.mean(np.linalg.norm(res[:, a:b].mean(axis=1), axis=1) /
                                   np.maximum(np.linalg.norm(res[:, a:b], axis=2).mean(axis=1), 1e-12)))
    coh[k] = dict(during=f(ONSET, ONSET + 10), later=f(ONSET + 40, ONSET + 80))

sig = lambda a: [float(f"{v:.4g}") for v in a]
out = dict(onset=ONSET, window=W, sigma=SIG, lag=LAG, n_seeds=N, steps=D["steps"],
           dt=0.05, threshold=float(f"{THRESH:.4g}"), push=D["push"],
           push_steps=D["push_steps"], rows=rows, sens=sens, lags=lags, coh=coh,
           conf=conf,
           acc_detected=(conf["body"][0] + conf["world"][0]) /
                        (sum(conf["body"]) + sum(conf["world"])),
           traces={k: sig(r.mean(axis=0)) for k, r in R.items()},
           lo={k: sig(r.min(axis=0)) for k, r in R.items()},
           hi={k: sig(r.max(axis=0)) for k, r in R.items()})
json.dump(out, open("summary_rung4.json", "w"), separators=(",", ":"))

print(f"sigma {SIG}, lag {LAG} steps ({LAG * 0.05:.2f}s), threshold {THRESH:.3e}\n")
print(f"{'condition':<30}{'peak':>10}{'+60':>10}{'detect':>8}{'correct':>9}")
for k, v in rows.items():
    r = "-" if v["right"] is None else f"{v['right'] * 100:.0f}%"
    print(f"{k:<30}{v['peak']:>10.2e}{v['late']:>10.2e}{v['detected'] * 100:>7.0f}%{r:>9}")
print(f"\nconfusion (detected events only): body {conf['body']}, world {conf['world']}")
print(f"accuracy on detected events: {out['acc_detected'] * 100:.0f}%")
print(f"false alarms on undisturbed runs: {rows[BASE]['detected'] * 100:.0f}%")
