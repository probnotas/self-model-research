"""Freeze the pre-registration. Run ONLY after the dev review, when the user says "freeze".

    python freeze_prereg.py

Writes prereg.json once (refuses to overwrite). It records the hypotheses, thresholds,
event parameters and the exact code (sha256 of every .py file in this folder), plus the
sha256 of dev_results.json, so the test run can prove nothing changed afterwards.
"""
import glob
import hashlib
import json
import os
import sys
import time


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    import common as C
    p = os.path.join(C.OUT, "prereg.json")
    if os.path.exists(p):
        sys.exit("prereg.json already exists; it is frozen and will not be overwritten")
    dev = os.path.join(C.OUT, "dev_results.json")
    if not os.path.exists(dev):
        sys.exit("run and review the dev evaluation first")
    pre = dict(
        frozen_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        scale=C.SCALE, compliant=C.COMPLIANT,
        rule="each hypothesis passes only if it holds on all 3 self-model seeds (0, 1, 2); AUCs raw",
        hypotheses=dict(
            H1="multi-step detect = log(early/pre err): AUC body vs none >= 0.90",
            H2="multi-step persist = log(late/pre err): AUC body vs push_short >= 0.80, non-fallen trials only",
            H3="multi-step leg localization (argmax per-leg early/pre ratio) accuracy on body events >= 0.50",
            H4="multi-step detect AUC > one-step detect AUC (body vs none)"),
        exploratory="sustained push and payload vs body; motion signal; per-event breakdowns; fall rates",
        windows=dict(pre=C.PRE, early=C.EARLY, late=C.LATE, onset=C.ONSET, T=C.T_TOTAL),
        events=dict(body=list(C.BODY_EVENTS), world=list(C.WORLD_EVENTS),
                    motor_weak="actuator_gear x0.5 (that leg)", leg_heavy="body_mass and body_inertia x3 (that leg's bodies)",
                    joint_damped="dof_damping x3 + 1 (that leg)",
                    push_short="0.2 s, horizontal, 0.5-1.5 x body weight, random direction",
                    push_sustained="0.15-0.3 x body weight, held to the end", payload="torso mass and inertia x1.5"),
        seeds=dict(split=C.SPLIT_SEED, selfmodel=list(C.SM_SEEDS)),
        action_noise=C.ACTION_NOISE, compliant_pd=dict(kp=C.KP, kd=C.KD) if C.COMPLIANT else None,
        code_sha256={os.path.basename(f): sha(f) for f in sorted(glob.glob(os.path.join(C.HERE, "*.py")))},
        dev_results_sha256=sha(dev))
    json.dump(pre, open(p, "w"), indent=1)
    print(f"frozen: {p}\nsha256 {sha(p)}")


if __name__ == "__main__":
    main()
