# JEPA quiet-body test — pre-registration (scratch, not saved to repo)

Train: healthy data only, seed-90000 recordings (results_history/episodes): pre-onset steps of all
530 trials + the 50 calibration episodes in full. Per config: soft_none, soft_low, rigid_none (control).
Test: fresh seed-777000 trials (results_replication/episodes), never seen by any JEPA.

Model: encoder E = MLP on the last 10 steps of [cos th, sin th, thdot, u] (40 numbers) -> 8-d summary.
Target encoder = EMA copy of E (tau 0.99), stop-grad. Predictor P(z_t, next 10 commands) -> summary of
the window 10 steps later. Loss: MSE. Collapse check: per-dim std of target summaries on held-out healthy
windows; INVALID if mean std < 0.05 x its value at init. If invalid -> report, rerun ONCE with a VICReg
variance term, labelled as such.

Score per trial: mean JEPA error over windows ending in onset+20 .. onset+100 (same window as Rung 6).
Checks (soft_none and soft_low):
 1 Direction: median score body-only vs undisturbed; "rises" if one-sided AUC(body-only > undisturbed) > 0.5.
 2 Motion: Pearson r(trial score, trial mean motion) over undisturbed+body-only; "escapes" if |r| < 0.3.
 3 Scoreboard vs raw one-step error and Rung-6 rule B: AUC body/undisturbed, AUC body/push,
   FP on pushes at threshold mean+2sd of fresh calibration scores.
