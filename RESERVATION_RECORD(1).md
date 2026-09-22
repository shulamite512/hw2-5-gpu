# Reservation and GPU-Hour Record

| Workstation | Reservation ID | GPU UUID | Reserved window (UTC) | Actual use (UTC, approx.) | Reserved GPU-hours | Consumed GPU-hours |
|---|---|---|---|---|---:|---:|
| ISB 840, machine 36 (RTX 5090) | Not available at time of submission | `GPU-a3a60fad-ef40-08fa-3eb7-9c22d773bd54` | 2026-09-15 19:00 to 2026-09-16 18:59 | 2026-09-16 01:54:18 to 02:14:46 | 24 | ~0.34 |

- Consumed time is dominated by the 1200 s thermal run; the precision, bandwidth and attention sweeps together used
  under a minute of GPU compute.
- Actual start/end times are reconstructed from `results/nvidia-smi-q.txt` and output file modification times
  (see `RUN_LOG.txt`), not from a live log.
- No other GPU was used for any measurement in this submission.

Evidence: `results/nvidia-smi-q.txt`, `results/thermal.csv`, `RUN_LOG.txt`.
