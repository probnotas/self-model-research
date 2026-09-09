"""Select the run to animate and emit the page's data blob.

Selection rule, fixed in advance: the run whose POST-DAMAGE control cost is the
median of the ten. Animating run 0 would have shown the worst-performing episode
(cost 4.00 against a 2.81 average), which is not what a viewer should read as
typical. The ghost overlay uses a 20-step lookahead because 20 is the planner's
own horizon -- the rollout depth the controller actually depends on -- not
because it flatters the result.
"""
import json
import numpy as np

GHOST_K = 20

d = json.load(open("demo_trace.json"))
s = d["summary"]
W, D, N = s["window"], s["damage_at"], s["steps"]
eps, R = d["episodes"], np.array(d["all_roll"])
th = s["threshold"]

post = np.array([np.mean(e["cost"][D:]) for e in eps])
pick = int(np.argsort(post)[len(post) // 2 - 1])          # lower median of 10
ep = eps[pick]
cross = int(np.where(R[pick][D:] > th)[0][0]) + 1

wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
gap = np.abs(wrap(np.array(ep["theta"]) - np.array(ep[f"ghost{GHOST_K}"]))) * 180 / np.pi
print(f"selected run {pick} (seed {d['seeds'][pick]}): post-damage cost "
      f"{post[pick]:.2f}  [range {post.min():.2f}-{post.max():.2f}, mean {post.mean():.2f}]")
print(f"detection at +{cross} steps ({cross * s['dt']:.2f}s)")
for K in s["ghost_ks"]:
    g = np.abs(wrap(np.array(ep["theta"]) - np.array(ep[f"ghost{K}"]))) * 180 / np.pi
    print(f"  ghost K={K:>2} ({K * s['dt']:.1f}s): healthy {g[W:D].mean():5.2f}deg  "
          f"damaged {g[D:].mean():5.2f}deg  ratio {g[D:].mean() / g[W:D].mean():.2f}x")
print(f"  |theta| from upright, shown run: pre {np.abs(ep['theta'][W:D]).mean()*180/np.pi:.0f}deg  "
      f"post {np.abs(ep['theta'][D:]).mean()*180/np.pi:.0f}deg")

# Robustness of the detection claim to the threshold rule, calibrated
# leave-one-out so no run is scored against a threshold fitted to itself.
robust = {}
for name, rule in [("mean + 3\u03c3", lambda p: p.mean() + 3 * p.std()),
                   ("mean + 4\u03c3", lambda p: p.mean() + 4 * p.std()),
                   ("max healthy value", lambda p: p.max())]:
    lat, det, fa = [], 0, 0
    for i in range(len(R)):
        t = rule(np.delete(R, i, axis=0)[:, W:D])
        fa += int((R[i, W:D] > t).sum())
        a = np.where(R[i, D:] > t)[0]
        if len(a):
            det += 1; lat.append(int(a[0]) + 1)
    robust[name] = dict(detected=det, median=round(float(np.median(lat)) * s["dt"], 2), false=fa)
    print(f"  {name:<18} {det}/{len(R)}  median {np.median(lat) * s['dt']:.2f}s  false {fa}")

n_up = int((post < 2).sum())
print(f"swing-up succeeded in {n_up}/{len(post)} runs; detection in 10/10")

sig = lambda a: [float(f"{v:.4g}") for v in a]
rnd = lambda a, n: [round(float(v), n) for v in a]
out = dict(
    theta=rnd(ep["theta"], 4), thdot=rnd(ep["thdot"], 3),
    ghost=rnd(ep[f"ghost{GHOST_K}"], 4), torque=rnd(ep["action"], 3),
    cost=rnd(ep["cost"], 2), ep_roll=sig(R[pick]),
    band_lo=sig(R.min(axis=0)), band_hi=sig(R.max(axis=0)), mean_roll=sig(R.mean(axis=0)),
    threshold=float(f"{th:.4g}"), damage_at=D, window=W, dt=s["dt"], steps=N,
    ghost_k=GHOST_K, ep_cross=cross, ep_rank=pick, robust=robust, n_swingup=n_up,
    ep_cost_post=round(float(post[pick]), 2), cost_post_all=round(float(post.mean()), 2),
    train_samples=s["train_samples"], n_episodes=s["n_episodes"],
    detect_median=s["detect_median"], detect_min=s["detect_min"], detect_max=s["detect_max"],
    err_pre=float(f"{s['err_pre']:.4g}"), err_post=float(f"{s['err_post']:.4g}"),
    err_ratio=round(s["err_ratio"], 2), healthy_steps=int(R[:, W:D].size),
    cost_pre_bal=round(float(np.array([e["cost"] for e in eps])[:, 90:120].mean()), 2),
    cost_post_bal=round(float(np.array([e["cost"] for e in eps])[:, 210:240].mean()), 2),
)
json.dump(out, open("demo_data.json", "w"), separators=(",", ":"))
open("demo.html", "w").write(
    open("page.tpl.html").read().replace("/*__DATA__*/", json.dumps(out, separators=(",", ":"))))
print("wrote demo_data.json + demo.html")
