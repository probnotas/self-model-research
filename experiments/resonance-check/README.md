# Resonance check: is the inverted signal just resonance detuning?

*Simulation only, not hardware. Open `compliant-pendulum-playground.html` in any browser (no server, no install).*

## Purpose

On the compliant (series-elastic) pendulum, changing the body's mass gave an **inverted fault-detection signal**: AUC 0.11. The robot got *quieter* after the body changed, so "more surprise = body change" pointed the wrong way, and motion dropped from about 0.050 to 0.015.

In a 25-minute call on Nov 11, 2026, Prof. Shai Revzen (UMich BIRDS Lab) suggested a simpler explanation: **resonance detuning**. A springy joint has a natural frequency, and the controller may have been driving the arm near it. Adding mass lowers that frequency, the drive falls off resonance, and the arm's swing shrinks. If that's all it is, the "quieter body" finding is ordinary spring-mass physics, not something the self-model discovered.

This experiment tests whether the two can be told apart.

## Hypothesis being tested

**If the finding is just resonance detuning,** a compliant pendulum with no learned model and no controller should reproduce the same amplitude drop after the same parameter change, driven only by a sine wave.

## Method

1. **Build the playground** (`compliant-pendulum-playground.html`). A motor commands `theta_m = A sin(omega t)`. A torsional spring `k` connects the motor to an arm of mass `m` at length `L`, with damping `c` and gravity:

   ```
   I = m L^2
   theta_a'' = (k (theta_m - theta_a) - m g L sin(theta_a) - c theta_a') / I
   ```

   It's integrated with semi-implicit Euler at dt = 0.002 s.
2. **Map the resonance curve** (arm amplitude vs drive frequency) by hand, or with the Sweep button, which simulates each frequency to steady state.
3. **Drive the arm, then bump the mass** to 1.5× mid-run with "Change the body mid-run". Read the after/before amplitude ratio.
4. **Compare qualitatively** with the main experiments: the AUC 0.11 inversion, and motion dropping from 0.050 to 0.015 (ratio about 0.30).

The page also draws the **linear-theory curve** (dashed): the small-angle prediction of pure resonance,

```
amplitude = k A / sqrt((k + m g L - I omega^2)^2 + (c omega)^2)
```

Anything the simulator does that this curve doesn't predict is nonlinear behaviour, not plain detuning.

## Pass and fail for my finding

- **Pass (there's more than resonance).** The playground shows amplitude changes that resonance physics alone can't explain. Examples: hysteresis (the curve differs sweeping up vs down), effects from large-angle nonlinearity, or motion-drop ratios outside what pure resonance allows for any reasonable drive frequency.
- **Fail (Revzen is right).** The playground reproduces the 0.050 → 0.015 drop (ratio about 0.30) from resonance detuning alone, at drive frequencies and stiffness comparable to the main experiment.

## Caveats (read before concluding anything)

- **The drive here is open-loop:** a fixed sine wave. In the main experiment, a controller reacts to the arm's motion, which can shift the effective frequency and damping. A match here makes detuning plausible; it doesn't prove that's what happened in the closed-loop runs.
- **The playground's spring is plain linear.** The main experiment's compliance model also had hysteresis and sensor noise (see `surprise_attribution/compliant_config.py`).
- **Two natural frequencies are shown.** `omega_n = sqrt(k / (m L^2))` is the spring alone. `omega_n` with gravity, `sqrt((k + m g L) / (m L^2))`, is where the peak actually sits at small angles. They differ noticeably when `m g L` is comparable to `k`.
- **This is a qualitative check.** It's for building intuition and deciding what to test next, not a pre-registered test.
