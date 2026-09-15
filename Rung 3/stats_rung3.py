"""What, if anything, in Rung 3 cleared the noise floor?

Rung 3's tables were reported as mean +/- sd over 15 runs and never tested. This
does the arithmetic: pairwise z on the difference of means, and the count of
comparisons that actually separate.
"""
import json
import numpy as np

d = json.load(open("results_rung3.json"))
N = 15
main = d["main"]
keys = [k for k in "abcdef"]                      # g is the model-free reference, sd not recorded

def z(m1, s1, m2, s2, n=N):
    se = np.sqrt(s1**2 / n + s2**2 / n)
    return (m1 - m2) / se

print(f"{'':>3} {'condition':<46}{'mean':>7}{'sd':>7}{'sem':>7}")
for k in keys + ["g"]:
    v = main[k]
    sd = v["std"]
    sem = "  n/a" if not np.isfinite(sd) else f"{sd / np.sqrt(N):>7.3f}"
    print(f"{k:>3}) {v['name']:<46}{v['mean']:>7.3f}"
          f"{'    n/a' if not np.isfinite(sd) else f'{sd:>7.3f}'}{sem}")

print("\nall pairwise comparisons among (a)-(f), |z| > 1.96 = separates")
sep = tot = 0
worst = []
for i, k1 in enumerate(keys):
    for k2 in keys[i + 1:]:
        zz = z(main[k1]["mean"], main[k1]["std"], main[k2]["mean"], main[k2]["std"])
        tot += 1
        if abs(zz) > 1.96:
            sep += 1
        worst.append((abs(zz), k1, k2, zz))
worst.sort(reverse=True)
for a, k1, k2, zz in worst[:4]:
    print(f"  {k1} vs {k2}: z = {zz:+.2f}{'  SEPARATES' if a > 1.96 else ''}")
print(f"  ... {sep} of {tot} comparisons separate")

print("\ngeneralisation to damage never seen in meta-training")
gen = d["generalization"]
for body, rows in gen.items():
    print(f"  {body}:")
    for k, (m, s) in rows.items():
        print(f"    {k}) {main[k]['name'][:44]:<46}{m:>7.3f}  sd {s:.3f}")
    if "f" in rows and "b" in rows:
        zz = z(rows["f"][0], rows["f"][1], rows["b"][0], rows["b"][1])
        print(f"    meta+uncertainty vs naive fine-tune: z = {zz:+.2f}"
              f"{'  SEPARATES' if abs(zz) > 1.96 else '  (not significant)'}")
    if "b" in rows and "a" in rows:
        zz = z(rows["b"][0], rows["b"][1], rows["a"][0], rows["a"][1])
        print(f"    naive fine-tune vs NO adaptation:   z = {zz:+.2f}"
              f"{'  fine-tuning is WORSE' if zz > 1.96 else '  (not significant)'}")

print("\nuncertainty penalty sweep (selection runs, lower is better)")
for name, sw in (("ensemble", d["selection"]["lambda_ensemble"]),
                 ("meta", d["selection"]["lambda_meta"])):
    base = sw["0.0"]
    print(f"  {name}: " + "  ".join(
        f"lam={k} {v:.3f} ({v - base:+.3f})" for k, v in sw.items()))
