# Fix 3: hardware test plan (not run; there is no hardware in this environment)

**Goal:** run the same fell/body/world rule on a real compliant joint.
- **Body change:** a clip-on weight.
- **Push:** a flick.

Nothing here has been built or tested.

## Why it ports

The rule needs only:
1. **What I commanded:** a motor torque, voltage or current command.
2. **How the joint moved:** an encoder angle, plus velocity derived from it.
3. **Whether I fell:** the arm hits a stop, or leaves its working range.
4. **A healthy self-test response:** a short, known wiggle, run on the healthy rig, used as the baseline.

No camera is needed, and no learned model of pixels.

## Rig (simplest version)

| part | option A: VEX (fastest if you own it) | option B: hobby parts |
|---|---|---|
| actuator and sensor | V5 Smart Motor (reports position, velocity, current and torque estimate) | DC gear motor with quadrature encoder, plus a motor driver |
| controller and logger | V5 Brain, or a laptop over USB | Arduino or Teensy streaming to a laptop |
| compliance | rubber or 3D-printed flexure coupling between motor shaft and arm | same |
| arm | 20–30 cm rod or 3D-printed beam | same |
| body change | clip-on weights (for example 20, 50 and 100 g) at the tip | same |
| push | a flick, or a tap with a stick, of consistent strength | same |
| "fell" | a hard stop at ±90°, or the arm leaving its working range | same |

## Task

**Skip upright balancing,** which is hard on real hardware. Instead, the controller holds the arm at a set angle, or tracks a slow sine, for example ±20° at 0.3 Hz. Command gain works for any closed-loop task.

## Protocol (mirrors the simulation)

1. **Healthy data:** about 30 min of normal operation, for training the inverse model and setting the 150-run calibration baseline.
2. **Hanging-calibration data:** 50 healthy self-tests from the rest position. Each is 2 s of zero command, then 3 s of a fixed sine command.
3. **Trials,** in random order, 30 or more per cell:
   - nothing;
   - weight only;
   - flick only;
   - weight and flick together.
4. **Before the trials:** write the pre-registration (same rule, thresholds and criteria as `PREREG.md`, adapted for the rig).
5. **After the trials:** score once.

## What will differ from simulation (risks)

- **Friction and backlash** in the gearbox add a dead-zone that the simulation doesn't have.
- **Current and torque estimates are noisy.** The command may need to be the voltage or setpoint instead.
- **Pushes won't be identical.** Their strength should be logged, for example with an accelerometer on the arm.
- **The weight's position on the arm changes the effect size.** Fix it with a marked clip point.

## Timeline

- **With parts on hand:** 1–2 weeks to the first logged dataset.
- **After that:** the analysis reuses `experiments/inverse/run_inverse.py` and `experiments/fixes/` almost unchanged.
