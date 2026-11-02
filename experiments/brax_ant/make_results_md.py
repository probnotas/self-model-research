"""Write results.md (plain language) from dev_results.json. Numbers only from the file.

    python make_results_md.py            # writes <OUT>/results.md
"""
import json
import os


def f(x, n=3):
    return "n/a" if x is None else f"{x:.{n}f}"


def main():
    import numpy as np
    import common as C
    R = json.load(open(os.path.join(C.OUT, "dev_results.json")))
    S = sorted(R["seeds"], key=int)
    rng = lambda key, kind="multi": [R["seeds"][s][kind][key] for s in S]
    span = lambda v: f"{f(np.mean(v))} ({f(min(v))}–{f(max(v))})" if all(x is not None for x in v) else "n/a"
    hyp = R["hypotheses_all_seeds"]
    L = []
    L.append(f"# Brax Ant: telling a body change from a world event, DEV results ({'compliant' if R['compliant'] else 'rigid'})\n")
    L.append("*Simulation only (Brax MJX), not hardware. These are DEV numbers, used to tune and decide. "
             "The test split hasn't been run, and the pre-registration isn't frozen yet.*\n")
    L.append("## What we tested\n")
    L.append("A walking Ant robot is trained to walk (PPO). Then its controller is frozen, and it gets a little action noise. "
             "A self-model learns, from healthy walking only, to predict what the Ant's body does next. "
             "Half-way through each 20 s trial (at 10 s), one thing happens:\n")
    L.append("- **A body change on one leg:** a weak motor, a 3× heavier leg, or a stiff (damped) joint.\n")
    L.append("- **A world event:** a short shove, a steady push, or a payload on the torso.\n")
    L.append("- **Nothing.**\n")
    L.append("We ask whether the self-model's prediction error (1) notices, (2) tells body from world, and (3) points to the right leg.\n")
    L.append(f"**Controller:** final eval reward {f(R['policy'], 0)} (spec asked for about 5000+; accepted below that, see README). "
             f"Action noise {R.get('action_noise', 'unrecorded')} (spec said 0.1; lowered to cut healthy falls).\n")
    L.append("## What happened (mean, min–max over 3 self-model seeds; AUC 0.5 = coin flip, below 0.5 = backwards)\n")
    L.append("| hypothesis | number | passes on all seeds? |\n|---|---|---|")
    L.append(f"| H1: multi-step error notices body changes (AUC body vs nothing ≥ 0.90) | {span(rng('detect_auc_body_vs_none'))} | {'yes' if hyp['H1'] else 'no'} |")
    L.append(f"| H2: error stays up for body changes but not short shoves (AUC ≥ 0.80, upright trials) | {span(rng('persist_auc_body_vs_push_short_nonfallen'))} | {'yes' if hyp['H2'] else 'no'} |")
    L.append(f"| H3: names the right leg (accuracy ≥ 0.50; chance 0.25) | {span(rng('leg_accuracy_body'))} | {'yes' if hyp['H3'] else 'no'} |")
    L.append(f"| H4: multi-step beats one-step at noticing (one-step AUC: {span(rng('detect_auc_body_vs_none', 'one'))}) | — | {'yes' if hyp['H4'] else 'no'} |\n")
    L.append("**Falls** (torso below 0.25 m or flipped, after the event):\n")
    L.append("| event | fall rate |\n|---|---|")
    for e, v in sorted(R["fall_rate"].items()):
        L.append(f"| {e} | {f(v, 2)} |")
    if R["recommend_lower_push"]:
        L.append(f"\n**Short shoves knock the Ant over {f(R['push_short_fall_rate'], 2)} of the time (above 0.20). "
                 "Recommendation: lower the short-push range before freezing.**")
    L.append("\n**By event** (multi-step, seed 0):\n")
    L.append("| event | median detect | median persist | detect AUC vs nothing | right leg |\n|---|---|---|---|---|")
    for e, v in sorted(R["seeds"][S[0]]["multi"]["per_event"].items()):
        L.append(f"| {e} | {f(v['median_detect'], 2)} | {f(v['median_persist'], 2)} | {f(v['detect_auc_vs_none'])} | {f(v['leg_accuracy'], 2)} |")
    ex = R["seeds"][S[0]]["multi"]["exploratory"]
    L.append("\n**Exploratory (seed 0, no pass/fail):**\n")
    L.append(f"- **Body vs steady push (persist, upright):** {f(ex['persist_auc_body_vs_nonfallen'].get('push_sustained'))}")
    L.append(f"- **Body vs payload (persist, upright):** {f(ex['persist_auc_body_vs_nonfallen'].get('payload'))}")
    L.append(f"- **Model-free motion signal:** AUC body vs nothing {f(R['motion']['auc_body_vs_none'])}; "
             f"body vs short shove {f(R['motion']['auc_body_vs_push_short'])}\n")
    gp = os.path.join(C.OUT, "gain_dev_results.json")
    if os.path.exists(gp):
        G = json.load(open(gp))
        gs = sorted(G["seeds"], key=int)
        gr = lambda fn: [fn(G["seeds"][s]["late"]) for s in gs]
        L.append("## Command gain: the signal that worked on the pendulum\n")
        L.append("The motors get a little random noise on top of what the controller asks for. An inverse model, trained on "
                 "healthy walking only, learns how much each nudge moves each leg. If a leg's nudges stop moving it like "
                 "they used to (weak motor, heavy leg, stiff joint), that leg's \"gain\" drops. A push on the torso shouldn't "
                 "change how a leg answers its own motor. Scored on trials where the Ant stayed upright "
                 f"({G['n'] - G['n_fallen']} of {G['n']}); gain change from before the event (1.0-9.5 s) to late after it (15-19.5 s).\n")
        L.append("| question | AUC or accuracy (mean, min-max over 3 seeds) |\n|---|---|")
        L.append(f"| notices body changes (body vs nothing) | {span(gr(lambda o: o['auc_body_vs_none']))} |")
        for e, name in (("push_short", "short shove"), ("push_sustained", "steady push"), ("payload", "torso payload")):
            L.append(f"| body change vs {name} | {span(gr(lambda o: o['auc_body_vs'][e]))} |")
        L.append(f"| names the right leg (chance 0.25) | {span(gr(lambda o: o['leg_accuracy']))} |\n")
        L.append("**By event** (seed 0; own-leg z below 0 means that leg's gain dropped):\n")
        L.append("| event | upright trials | median score | AUC vs nothing | right leg | median own-leg z |\n|---|---|---|---|---|---|")
        for e, v in sorted(G["seeds"][gs[0]]["late"]["per_event"].items()):
            L.append(f"| {e} | {v['n_upright']} | {f(v['median_score'], 2)} | {f(v['auc_vs_none'])} | {f(v['leg_accuracy'], 2)} | {f(v['median_own_leg_z'], 2)} |")
        L.append(f"\nFallen trials (left out of the gain scores): {G['fallen_by_event']}\n")
    L.append("## Caveats\n")
    L.append("- **Simulation, not hardware.** One controller and one robot design.")
    L.append("- **These are dev results.** They're used to check the pipeline and set the push range; the test split will be scored once, after the pre-registration is frozen.")
    L.append("- **A payload on the torso is physically a body-mass change,** so the line between \"body\" and \"world\" is a choice of definition here.")
    L.append("- **Sustained pushes and payloads never end,** so persistence can't separate them from body changes by design. That's why they're exploratory.")
    open(os.path.join(C.OUT, "results.md"), "w").write("\n".join(L) + "\n")
    print(f"wrote {os.path.join(C.OUT, 'results.md')}")


if __name__ == "__main__":
    main()
