# Robot truth (demo) — definitions v0.1

## What “robot truth” means here

**Robot truth** = a small set of **explicit definitions** so a metric means the same
thing in the chart, in the text report, and in conversation.

This is a **portfolio demo**, not an internal Robust AI metric spec.

## Implemented checks

### A) Heuristic review events (NOT certified safety)

- **sudden_stop:** large negative per-step speed change  
- **sharp_turn:** large magnitude angular velocity  
- **stall:** sustained near-zero speed  
- **velocity_spike:** large translational acceleration proxy  

### B) Velocity self-consistency

- **speed_r:** reported translational speed from telemetry  
- **speed_d:** speed implied by `(x, y)` changes vs time  
- **mismatch windows:** where `abs(speed_r - speed_d)` exceeds a scaled threshold  

## Version

- **v0.1:** initial demo definitions shipped with this repo.
