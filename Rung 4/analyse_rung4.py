"""Detect, then classify: was that my body or the world?

Pipeline, deliberately as dumb as it can be while still working:

  1. detect    the 10-step trailing mean of prediction error crosses a threshold
               calibrated on undisturbed runs only (leave-one-out, mean + 3 sigma)
  2. classify  LAG steps after that alarm, is the error STILL above the threshold?
               yes -> body change (permanent)    no -> world event (transient)

No learned classifier, no features beyond the error the detector already computes.
"""
import json
import numpy as np

D = json.load(open("results_rung4.json"))
ONSET, W, SIG, N = D["onset"], D["window"], D["sigma"], D["n_seeds"]
CONDS = D["conditions"]
BASE = "baseline (nothing happens)"
truth = lambda k: "body" if k.startswith("body") else ("world" if k.startswith("world") else "none")


def rolling(x, w=W):
    x = np.asarray(x, float)
    return np.array([x[max(0, i - w + 1):i + 1].mean() for i in range(len(x))])


R = {k: np.array([rolling(r["err"]) for r in v]) for k, v in CONDS.items()}
base_pre = R[BASE][:, W:]


def thresh_for(i):
    """Leave-one-out over the undisturbed runs, so no run sets its own bar."""
    p = np.delete(base_pre, i % len(base_pre), axis=0)
    return p.mean() + SIG * p.std()


print(f"threshold (all baseline runs): {base_pre.mean() + SIG * base_pre.std():.3e}")
print(f"baseline error level:          {base_pre.mean():.3e}\n")

print(f"{'condition':<30}{'peak':>10}{'+60 steps':>11}{'detected':>10}")
for k, r in R.items():
    pk = r[:, ONSET:ONSET + 15].max(axis=1).mean()
    late = r[:, ONSET + 60].mean()
    det = np.mean([(r[i, ONSET:] > thresh_for(i)).any() for i in range(len(r))])
    print(f"{k:<30}{pk:>10.2e}{late:>11.2e}{det * 100:>9.0f}%")

print("\ndetector sensitivity - false alarms are the cost of catching mild changes")
print(f"{'sigma':>6}{'false alarm':>13}{'mass':>7}{'length':>8}{'damping':>9}")
for sg in (3, 4, 5, 6):
    def th_(i):
        p_ = np.delete(base_pre, i % len(base_pre), axis=0)
        return p_.mean() + sg * p_.std()
    fa = np.mean([(R[BASE][i, ONSET:] > th_(i)).any() for i in range(N)])
    row = [np.mean([(R[k][i, ONSET:] > th_(i)).any() for i in range(N)])
           for k in ("body: mass doubled", "body: length +50%", "body: damping added")]
    print(f"{sg:>6}{fa * 100:>12.0f}%{row[0] * 100:>6.0f}%{row[1] * 100:>7.0f}%{row[2] * 100:>8.0f}%")

def det_only(lag):
    """Classification is a separate job from detection: score it only on the
    events the detector fired on, so an undetectable change is not charged twice."""
    ok = tot = 0
    for k, r in R.items():
        t = truth(k)
        if t == "none":
            continue
        for i in range(len(r)):
            th = thresh_for(i)
            w = np.where(r[i, ONSET:] > th)[0]
            if not len(w):
                continue
            td = ONSET + int(w[0]); j = min(td + lag, len(r[i]) - 1)
            tot += 1
            ok += int(("body" if r[i, j] > th else "world") == t)
    return ok / tot


print("\nclassifier accuracy by lag (how long it waits after the alarm)")
print(f"{'lag':>5}{'steps':>8}{'body right':>12}{'world right':>13}{'overall':>9}")
best = None
for lag in (10, 20, 30, 40, 60, 80):
    hit = {"body": [0, 0], "world": [0, 0]}
    for k, r in R.items():
        t = truth(k)
        if t == "none":
            continue
        for i in range(len(r)):
            th = thresh_for(i)
            w = np.where(r[i, ONSET:] > th)[0]
            if not len(w):
                hit[t][1] += 1                       # missed entirely -> counts against
                continue
            td = ONSET + int(w[0])
            j = min(td + lag, len(r[i]) - 1)
            pred = "body" if r[i, j] > th else "world"
            hit[t][0 if pred == t else 1] += 1
    b, wo = hit["body"], hit["world"]
    acc = (b[0] + wo[0]) / (sum(b) + sum(wo))
    print(f"{lag:>5}{lag * 0.05:>7.2f}s{b[0] / sum(b) * 100:>11.0f}%"
          f"{wo[0] / sum(wo) * 100:>12.0f}%{acc * 100:>8.0f}%")
    if lag == 30:
        print(f"        ^ counting only events the detector actually fired on: "
              f"{det_only(lag) * 100:.0f}%")
    if best is None or acc > best[1]:
        best = (lag, acc)
print(f"\nbest lag {best[0]} steps ({best[0] * 0.05:.2f}s), accuracy {best[1] * 100:.0f}%")

print("\nper-condition at the best lag")
lag = best[0]
for k, r in R.items():
    t = truth(k)
    if t == "none":
        ok = np.mean([not (r[i, ONSET:] > thresh_for(i)).any() for i in range(len(r))])
        print(f"  {k:<30} correctly silent {ok * 100:>3.0f}%")
        continue
    good = 0
    for i in range(len(r)):
        th = thresh_for(i)
        w = np.where(r[i, ONSET:] > th)[0]
        if not len(w):
            continue
        td = ONSET + int(w[0]); j = min(td + lag, len(r[i]) - 1)
        good += int(("body" if r[i, j] > th else "world") == t)
    print(f"  {k:<30} called {t:<5} {good}/{len(r)}")

print("\nresidual coherence - is the error one-directional?")
print("  (norm of the mean residual / mean residual norm; 1 = same direction every step)")
for k, v in CONDS.items():
    res = np.array([r["res"] for r in v])                  # (runs, steps, 3)
    def coh(a, b):
        seg = res[:, a:b]
        num = np.linalg.norm(seg.mean(axis=1), axis=1)
        den = np.linalg.norm(seg, axis=2).mean(axis=1)
        return float(np.mean(num / np.maximum(den, 1e-12)))
    print(f"  {k:<30} during {coh(ONSET, ONSET + 10):.2f}   later {coh(ONSET + 40, ONSET + 80):.2f}")
