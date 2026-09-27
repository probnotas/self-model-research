# Surprise attribution: body change vs world event

A frozen forward model (the pendulum's self-model) is used to balance
Pendulum-v1 with CEM-MPC. At every step we log its one-step prediction error,
the "surprise". Partway through, one of two things happens:

- **body change**: a physical parameter is permanently altered (default: length ×1.5)
- **world event**: an external torque the model can't see acts for 5 steps, then stops

The question is whether the *shape* of the surprise tells you which one it was.

```
python run_experiment.py            # train/load model, run 3 conditions, CSVs, figure, features
python run_experiment.py --retrain  # same, but train a fresh forward model first
python plot.py                      # re-draw the figure from the CSVs
python env_setup.py                 # check the push-capable physics == stock Pendulum-v1
```

Dependencies: numpy, torch, gymnasium, matplotlib. The whole run takes about 10 s on CPU.

## Files

| file | what it does |
|---|---|
| `config.py` | **every knob**: disturbance timing, type, magnitude, feature windows, seeds |
| `env_setup.py` | stock `Pendulum-v1` plus the two disturbance channels, and a physics equivalence check |
| `model.py` | the forward model (MLP 4→64→64→3, MSE), data collection, training, save/load. **This is where your own model goes** |
| `planner.py` | CEM-MPC that plans through the forward model |
| `run_experiment.py` | runs the conditions, computes the surprise signal, writes the CSVs, calls plot and features |
| `features.py` | summary features: before vs after, and persistence |
| `plot.py` | the figure |

### Using your own forward model

The rest of the code only calls `model.predict_next(states, actions) -> next_states`,
on batched float32 tensors of shape `(N,3)`, `(N,1)` → `(N,3)`, in state space.
Wrap your model so it exposes that method and return it from `model.load_model()`.
The docstring at the top of `model.py` has the details.

## Outputs (`results/`)

- `baseline.csv`, `body_change.csv`, `world_event.csv`: one row per step. Columns:
  `surprise` (the error norm), per-dimension residuals, action, external torque,
  current length/mass, θ, θ̇, and an `event` marker.
- `surprise_signatures.png`: all three signals, raw (top) and smoothed (bottom), log scale.
- `summary_features.csv`: the table below.

## Result (seed 0, defaults)

![surprise signatures](results/surprise_signatures.png)

| condition | pre mean | post/pre | peak | late/pre | still elevated? |
|---|---|---|---|---|---|
| baseline | 1.13e-02 | 0.95 | 1.74e-02 | 0.91 | no |
| body change (l ×1.5) | 1.13e-02 | 4.53 | 9.22e-02 | **5.71** | **YES** |
| world event (1 N·m, 5 steps) | 1.13e-02 | 4.09 | **1.64e-01** | **1.06** | **no** |

- **Before vs after (feature a)** can't separate them. Both jump about 4× right
  after the onset. It tells you *that* something happened.
- **Persistence (feature b)** does separate them. Three seconds later the body-change error is
  still 5.7× its pre-onset level. The world-event error is back to 1.06×.
- The world event has the **bigger** spike (0.164 vs 0.092), so peak size isn't
  the thing doing the separating.

Quick robustness check, 8 seeds per condition (not part of the saved outputs):

| condition | still elevated | mean peak | mean late/pre |
|---|---|---|---|
| baseline | 0/8 | 0.020 | 1.04 |
| body: length ×1.5 | 8/8 | 0.088 | 5.26 |
| body: mass ×1.5 | 7/8 | 0.067 | 4.56 |
| world: 1 N·m push | 0/8 | 0.159 | 0.98 |
| world: 2 N·m push | 0/8 | 0.306 | 0.99 |

## Caveats you should be ready to defend

- **The body-change pendulum eventually falls over** (~10 s in the default run).
  A longer pendulum needs more torque to hold at a given angle, the controller
  is limited to ±2 N·m, and the model's persistent mismatch causes a slow drift
  that ends up past the recoverable angle. The late window (7–8 s) sits *before*
  the fall (|θ| ≈ 0.15–0.2 rad), so the "still elevated" result isn't caused
  by falling. But if you move `LATE_LAG` later, you'll be measuring a fallen
  pendulum. `run_experiment.py` prints the final |θ| of each run so you can check.
- **Mass only matters through torque.** In Pendulum-v1, `m` appears only in
  `3/(m l²)·u`. A mass change is invisible while `u ≈ 0`. It shows up here
  because the drifting controller is actively pushing. For the same reason
  a mass change is exactly equivalent to a weaker motor. No self-model can tell
  those two apart on this body. (See `../Rung 4/identifiability.py`.)
- **The error norm is dominated by θ̇.** θ̇ is unbounded rad/s, while cos/sin lie in
  [−1, 1]. Every disturbance here acts on angular acceleration, so that's fine,
  but the per-dimension residuals are in the CSVs if you need them.
- **One episode per condition** is what the saved outputs contain. The 8-seed table
  above is the evidence that the pattern isn't a one-seed accident.
- **Threshold choice.** "Still elevated" means late mean > pre mean + 4 σ, both taken from
  the episode's own pre-onset window. The rule doesn't depend on any disturbed data.

## Relation to `Rung 4/`

`Rung 4/` in this repo already runs a bigger version of the same experiment:
7 conditions × 20 seeds, a peak-matched push calibration, and a
detect-then-classify pipeline. It depends on code in `Rung 1/` and `Rung 3/`.
This folder is a self-contained, minimal version of the core measurement: one
episode per condition, the Euclidean error norm (Rung 4 used per-step MSE), and every
parameter in one config file. Treat it as the clean base to extend.
