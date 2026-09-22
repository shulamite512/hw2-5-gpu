# AI Use Disclosure

I used two AI tools on this assignment: Claude Code and Claude (claude.ai chat). All GPU measurements were
produced by me running `benchmark_hw2_5.py` on the reserved RTX 5090 (UUID `GPU-a3a60fad-ef40-08fa-3eb7-9c22d773bd54`).
No AI tool executed anything on the GPU, and no measurement in this repository was generated or invented by AI.

## Original harness and notebook

[EDIT BEFORE SUBMITTING — keep the one sentence that is true:]
- `benchmark_hw2_5.py` and `HW2_5_GPU_Assignment.ipynb` were written with AI assistance, and I reviewed and ran them.
- `benchmark_hw2_5.py` and `HW2_5_GPU_Assignment.ipynb` were written by me without AI assistance.

## Claude Code

- Audited the repository against the assignment and identified gaps (unfilled report templates, OOM boundary not
  reached, no FP8 attempt, no git tag).
- Wrote `HW2_5_GPU_Assignment_documented.ipynb`, including an FP8 cell and an OOM-boundary search cell.
  **These cells were never executed**, and no results from them are reported.
- Drafted earlier versions of `METRICS.md`, `RUN_LOG.txt` and `RESERVATION_RECORD.md` by computing derived
  statistics from my raw result files, and reconstructed approximate run timestamps from file modification times
  (flagged as approximate in `RUN_LOG.txt`).
- Looked up NVIDIA specifications for Part A.

## Claude (chat)

- Reviewed my executed notebook, raw result files, figures and harness against the assignment requirements.
- Found errors in my results: the attention fit in `results/attention_fit.txt` mixed naive and fused rows
  (reported 2.057 instead of the naive-only 4.0 bytes/L²); the TF32 percentage-of-peak used the wrong peak;
  duplicated runs caused the zig-zag plot lines; attention latencies included input generation.
- Recomputed the naive-only memory fit, the fused fit, the TF32 percentages, the clock-proxy throughput ratio,
  and the thermal summary from my existing raw files.
- Wrote a revised harness (time-based repetitions, OOM bisection, FP8 attempt, throughput logging).
  **It was not run**, and it is not the harness that produced the submitted data.
- Drafted the final versions of `METRICS.md`, `RUN_LOG.txt`, `AI_USE.md` and `RESERVATION_RECORD.md`.

## What I did

- Reserved the workstation and ran all measurements.
- Reviewed the AI-drafted reports against my raw files and am responsible for their contents.

## Verification

Every number in `METRICS.md` can be recomputed from `results/*.jsonl`, `results/thermal.csv` and
`results/nvidia-smi-q.txt`. Values that are projections rather than measurements (the attention OOM lengths)
are labelled as projections.
