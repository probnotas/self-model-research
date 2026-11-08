# Brax Ant: telling a body change from a world event, DEV results (rigid)

*Simulation only (Brax MJX), not hardware. These are DEV numbers, used to tune and decide. The test split hasn't been run, and the pre-registration isn't frozen yet.*

## What we tested

A walking Ant robot is trained to walk (PPO). Then its controller is frozen, and it gets a little action noise. A self-model learns, from healthy walking only, to predict what the Ant's body does next. Half-way through each 20 s trial (at 10 s), one thing happens:

- **A body change on one leg:** a weak motor, a 3× heavier leg, or a stiff (damped) joint.

- **A world event:** a short shove, a steady push, or a payload on the torso.

- **Nothing.**

We ask whether the self-model's prediction error (1) notices, (2) tells body from world, and (3) points to the right leg.

**Controller:** final eval reward 3198 (spec asked for about 5000+; accepted below that, see README). Action noise 0.05 (spec said 0.1; lowered to cut healthy falls).

## What happened (mean, min–max over 3 self-model seeds; AUC 0.5 = coin flip, below 0.5 = backwards)

| hypothesis | number | passes on all seeds? |
|---|---|---|
| H1: multi-step error notices body changes (AUC body vs nothing ≥ 0.90) | 0.742 (0.741–0.743) | no |
| H2: error stays up for body changes but not short shoves (AUC ≥ 0.80, upright trials) | 0.807 (0.806–0.808) | yes |
| H3: names the right leg (accuracy ≥ 0.50; chance 0.25) | 0.512 (0.505–0.521) | yes |
| H4: multi-step beats one-step at noticing (one-step AUC: 0.716 (0.712–0.722)) | — | yes |

**Falls** (torso below 0.25 m or flipped, after the event):

| event | fall rate |
|---|---|
| joint_damped | 0.08 |
| leg_heavy | 0.16 |
| motor_weak | 0.13 |
| none | 0.03 |
| payload | 0.09 |
| push_short | 0.06 |
| push_sustained | 0.12 |

**By event** (multi-step, seed 0):

| event | median detect | median persist | detect AUC vs nothing | right leg |
|---|---|---|---|---|
| joint_damped | 0.01 | 0.04 | 0.520 | 0.37 |
| leg_heavy | 0.64 | 0.66 | 0.843 | 0.26 |
| motor_weak | 0.82 | 0.79 | 0.867 | 0.89 |
| none | 0.03 | 0.10 | n/a | n/a |
| payload | -0.09 | -0.05 | 0.457 | n/a |
| push_short | 0.07 | -0.03 | 0.602 | n/a |
| push_sustained | 0.22 | 0.36 | 0.739 | n/a |

**Exploratory (seed 0, no pass/fail):**

- **Body vs steady push (persist, upright):** 0.586
- **Body vs payload (persist, upright):** 0.804
- **Model-free motion signal:** AUC body vs nothing 0.351; body vs short shove 0.403

## Command gain: the signal that worked on the pendulum

The motors get a little random noise on top of what the controller asks for. An inverse model, trained on healthy walking only, learns how much each nudge moves each leg. If a leg's nudges stop moving it like they used to (weak motor, heavy leg, stiff joint), that leg's "gain" drops. A push on the torso shouldn't change how a leg answers its own motor. Scored on trials where the Ant stayed upright (455 of 512); gain change from before the event (1.0-9.5 s) to late after it (15-19.5 s).

| question | AUC or accuracy (mean, min-max over 3 seeds) |
|---|---|
| notices body changes (body vs nothing) | 0.737 (0.716–0.762) |
| body change vs short shove | 0.728 (0.726–0.729) |
| body change vs steady push | 0.641 (0.639–0.643) |
| body change vs torso payload | 0.717 (0.708–0.729) |
| names the right leg (chance 0.25) | 0.546 (0.540–0.555) |

**By event** (seed 0; own-leg z below 0 means that leg's gain dropped):

| event | upright trials | median score | AUC vs nothing | right leg | median own-leg z |
|---|---|---|---|---|---|
| joint_damped | 118 | 1.61 | 0.614 | 0.41 | -0.70 |
| leg_heavy | 108 | 1.69 | 0.637 | 0.28 | -0.44 |
| motor_weak | 111 | 6.31 | 0.955 | 0.94 | -6.31 |
| none | 31 | 1.34 | n/a | n/a | n/a |
| payload | 29 | 1.28 | 0.514 | n/a | n/a |
| push_short | 30 | 1.40 | 0.514 | n/a | n/a |
| push_sustained | 28 | 1.79 | 0.614 | n/a | n/a |

Fallen trials (left out of the gain scores): {'joint_damped': 10, 'leg_heavy': 20, 'motor_weak': 17, 'none': 1, 'payload': 3, 'push_short': 2, 'push_sustained': 4}

## Caveats

- **Simulation, not hardware.** One controller and one robot design.
- **These are dev results.** They're used to check the pipeline and set the push range; the test split will be scored once, after the pre-registration is frozen.
- **A payload on the torso is physically a body-mass change,** so the line between "body" and "world" is a choice of definition here.
- **Sustained pushes and payloads never end,** so persistence can't separate them from body changes by design. That's why they're exploratory.
