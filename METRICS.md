# HW2.5 — Precision, Bandwidth, and the Cost of Attention

All values are computed from the UUID-labelled records in `results/` and `RUN_LOG.txt`.

**GPU:** NVIDIA GeForce RTX 5090, UUID `GPU-a3a60fad-ef40-08fa-3eb7-9c22d773bd54`, workstation ISB 840 machine 36.
(The JSONL records store the UUID without the `GPU-` prefix: `a3a60fad-ef40-08fa-3eb7-9c22d773bd54`. Same device.)

**Incomplete items, stated up front:** the naive and fused attention OOM boundaries were not reached (Part D), the
FP8 attempt was not executed (Part B), and per-sample throughput was not logged during the thermal run, so Part E's
throughput ratio uses graphics clock as a proxy. Each is explained in its section. No value in this report is
estimated in place of a measurement unless it is explicitly labelled as a projection.

---

## Part A — Provenance

| Field | Value | Source |
|---|---|---|
| GPU model | NVIDIA GeForce RTX 5090 | `results/nvidia-smi-q.txt` |
| GPU UUID | `GPU-a3a60fad-ef40-08fa-3eb7-9c22d773bd54` | `results/nvidia-smi-q.txt` |
| Driver version | 610.60 | `results/nvidia-smi-q.txt` |
| CUDA version (driver-reported) | 13.3 | `results/nvidia-smi-q.txt` |
| CUDA runtime (PyTorch build) | 12.8 (torch 2.11.0+cu128) | `cuda_runtime` field in every JSONL record |
| VRAM capacity | 32607 MiB total framebuffer (32 GB), 420 MiB reserved | `results/nvidia-smi-q.txt`, FB Memory Usage |
| Reported power limit | 575.00 W current/default (range 400–575 W) | `results/nvidia-smi-q.txt` |
| Driver model / display | WDDM, display attached and active on this card | `results/nvidia-smi-q.txt` |
| Architecture | Blackwell (GB202) | NVIDIA RTX Blackwell GPU Architecture whitepaper [1] |
| Memory type / bandwidth | 32 GB GDDR7, 512-bit, **1792 GB/s** | [1], [2] |
| Tensor core generation | 5th generation | [1], [2] |
| Reduced tensor-core precisions | TF32, FP16, BF16, FP8, FP6, FP4, INT8 | [1] |
| FP32 peak (CUDA cores) | 104.8 TFLOPS at 2.41 GHz boost | [1] |
| TF32 tensor peak, dense | 104.8 TFLOPS | [1] |
| FP16/BF16 tensor peak, dense, FP32 accumulate | 209.5 TFLOPS (419 with FP16 accumulate) | [1] |

[1] NVIDIA, *NVIDIA RTX Blackwell GPU Architecture* whitepaper,
https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf, accessed 2026-09-22.
[2] NVIDIA GeForce RTX 5090 product page, https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/, accessed 2026-09-22.

PyTorch's FP16/BF16 matmuls accumulate in FP32, so the FP32-accumulate row (209.5) is the correct peak for Part B.

---

## Part B — Precision and achieved throughput

`torch.mm`, N×N, 5 warm-up iterations, timed with CUDA events. Repetitions: 10 for N < 8192, 5 for N ≥ 8192.
The benchmark was invoked twice; both runs are in `results/precision.jsonl` and the table reports their mean.
The repetition counts are low (see Limitations).

**Achieved TFLOPS (mean of 2 runs)**

| N | FP32 | TF32 | FP16 | BF16 |
|---:|---:|---:|---:|---:|
| 1024 | 43.90 | 71.20 | 115.69 | 87.97 |
| 4096 | 68.77 | 91.31 | 205.98 | 178.16 |
| 8192 | 66.75 | 111.48 | 223.78 | 229.53 |
| 16384 | 66.38 | 115.64 | 216.03 | 228.80 |

**% of theoretical peak** (FP32 and TF32: 104.8 TFLOPS; FP16 and BF16: 209.5 TFLOPS)

| N | FP32 | TF32 | FP16 | BF16 |
|---:|---:|---:|---:|---:|
| 1024 | 41.9% | 67.9% | 55.2% | 42.0% |
| 4096 | 65.6% | 87.1% | 98.3% | 85.0% |
| 8192 | 63.7% | 106.4% | 106.8% | 109.6% |
| 16384 | 63.3% | 110.3% | 103.1% | 109.2% |

**Results above 100% of peak.** TF32, FP16 and BF16 exceed their whitepaper peaks by up to ~10%. The whitepaper
peaks assume the 2.41 GHz boost clock. This card ran at about 2.53 GHz sustained and up to 2.60 GHz when cold
(Part E), which is 5–8% above the boost clock. That explains most, but not clearly all, of the excess:
clock-adjusted to 2.60 GHz, the BF16 peak would be about 226 TFLOPS, and the best BF16 run (233.8) is still ~3% above it.
The remaining gap is within the run-to-run spread of these measurements (up to ~4% between the two runs at the same
configuration), which is large because of the low repetition count. I report the numbers as measured rather than
capping them at 100%.

**Plateaus**

- **FP32** plateaus by N = 4096 (~67–69 TFLOPS through N = 16384).
- **FP16** plateaus by N = 8192 (223.8, then 216.0 at N = 16384, within run-to-run noise).
- **BF16** plateaus by N = 8192 (229.5, then 228.8).
- **TF32** is close to plateau at N = 8192–16384: it rises only 3.7% from 8192 to 16384, after rising 22% from 4096 to 8192.

**Why small matrices never plateau.** At N = 1024 a matmul takes only ~18–50 µs, so fixed kernel-launch overhead is a
large fraction of the runtime. The problem is also too small to fill the GPU: with a typical 128×128 output tile, a
1024×1024 result has only 64 tiles to distribute across the card's 170 SMs, so most SMs sit idle. At N = 4096 there are
1024 tiles (about six waves), and the remaining losses come from wave quantization.

**Lower precision (FP8).** Not executed. The attempt I prepared uses `torch._scaled_mm` with `float8_e4m3fn` inputs
(supported in principle on compute capability 12.0 with torch 2.11 / CUDA 12.8), but I ran out of reserved GPU time
before running it, so I have no result or error message to report. Tooling note from preparing it: `_scaled_mm`
requires the second operand in column-major layout and explicit scale tensors, which is less mature than the ordinary
`torch.mm` path used for the other precisions.

---

## Part C — Bandwidth-bound vs. compute-bound

Invoked twice; values are the mean, with the better run in brackets.

| Operation | Measured | vs. spec |
|---|---:|---:|
| Elementwise add, 256M FP32 elements, preallocated output | **1526.3 GB/s** effective [1568.3] | **85.2%** [87.5%] of 1792 GB/s |
| 8192×8192 FP16 matmul | 223.9 TFLOPS [229.8] | 106.9% of 209.5 TFLOPS (same caveat as Part B) |

Effective bandwidth counts 3 × 4 bytes per element (read two inputs, write one output).

**Arithmetic intensity and roofline.** The ridge point is 209.5 TFLOPS ÷ 1792 GB/s ≈ **117 FLOP/byte**.

- Elementwise add: 1 FLOP per 12 bytes = **0.083 FLOP/byte**, far below the ridge, so it is **memory-bound**.
  That matches its behaviour: it reaches 85% of peak bandwidth while using a tiny fraction of peak compute.
- 8192 matmul: 2N³ FLOPs ÷ (3N² × 2 bytes) = **2731 FLOP/byte**, far above the ridge, so it is **compute-bound**.

---

## Part D — The cost of attention

Single head, **head_dim = 128, batch = 1**, FP16. Naive: `softmax(QKᵀ·scale)·V`, materializing the full L×L matrix.
Fused: `torch.nn.functional.scaled_dot_product_attention` (backend chosen automatically by PyTorch; not recorded).
1 warm-up, 3 timed repetitions, invoked twice; values are the mean.

| Length | Naive (ms) | Fused (ms) | Speedup | Naive peak (MiB) | Fused peak (MiB) |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.230 | 0.090 | 2.56× | 9.62 | 8.62 |
| 1024 | 0.111 | 0.091 | 1.22× | 13.12 | 9.12 |
| 2048 | 0.259 | 0.151 | 1.71× | 26.12 | 10.12 |
| 4096 | 0.178 | 0.273 | 0.65× | 76.12 | 12.12 |
| 8192 | 0.632 | 0.515 | 1.23× | 272.12 | 16.12 |
| 16384 | 2.301 | 1.992 | 1.16× | 1048.12 | 24.12 |

**Latency caveat.** The harness creates Q, K and V with `torch.randn` inside the timed function, so every latency
above includes input generation, and only 3 repetitions were taken. The sub-millisecond rows, including the 4096 row
where fused appears slower, are dominated by this overhead and by noise and should not be read as a real crossover.
Peak-memory measurements are unaffected (they are deterministic and identical across both runs).

**Memory fit — the quadratic term from my own data.** Fitting peak memory against length over the six naive points:

```
naive:  peak_memory_bytes = 4.00 * L^2 + 1024 * L + 8.5e6      (R^2 ≈ 1.0)
        two-term model:     4.057 * L^2 + 1.06e7
fused:  peak_memory_bytes = 1024 * L + 8.5e6                    (no L^2 term)
```

**Measured quadratic coefficient: 4.0 bytes per L².** It has a direct interpretation: two FP16 L×L matrices are alive
at peak, the scaled score matrix and the softmax output, at 2 bytes each. The linear term, 1024 bytes per token,
equals Q, K, V and the output at head_dim 128 in FP16 (4 × 128 × 2 bytes). The fused implementation has exactly the
same linear term and no quadratic term.

*Correction to the committed `results/attention_fit.txt`:* that file reports `2.057 * L^2`. It was produced by the
harness's `plot` command, which fitted the naive and fused rows together; the flat fused points pull the coefficient
down. The values above come from refitting the naive rows only.

**OOM boundaries — not reached.**

- **Naive:** no OOM at any tested length; the largest tested, L = 16384, succeeded using 1.02 GiB.
  *Projection only, not a measurement:* at 4 bytes/L² the fit predicts OOM around L ≈ 90,000 on this card's ~31 GiB of usable memory.
- **Fused:** no OOM at any tested length; L = 16384 succeeded using 24 MiB.
  *Projection only:* at 1024 bytes/token, OOM would not occur until roughly 30 million tokens, far beyond any practical run time given the O(L²) compute.

So I can report only the largest successful length (16384 for both) and no smallest failing length. The extended
sweep and bisection needed to locate the boundaries were prepared but not run within my reserved GPU time.

**What the fused kernel avoids.** The naive path writes the full L×L score matrix to GPU memory, reads it back for the
softmax, writes the softmax output back out, and reads it again for the second matmul, so both its memory and its
memory traffic grow with L². The fused kernel tiles Q, K and V into blocks that fit in on-chip SRAM, computes the
softmax incrementally with a running maximum and normalizer, and accumulates the output tile by tile, so the L×L
matrix never exists in global memory. That is exactly what the fits show: identical linear terms, and the 4-byte-per-L²
term present only in the naive path.

---

## Part E — Sustained load and thermal behaviour

8192×8192 FP16 matmul in a continuous loop for 1200 s, sampled every 5 s with `nvidia-smi` (`results/thermal.csv`,
241 samples: GPU clock, memory clock, temperature, power, utilization).

- **Power** reached the 575 W limit at the first loaded sample (t = 5 s) and stayed at 572–585 W for the whole run.
- **Clock** peaked at 2595 MHz at t = 5 s, then decreased as the die warmed, settling at 2520–2535 MHz by about
  t = 50–80 s. Apart from brief single-sample spikes (up to 2610 MHz near t ≈ 780 s), it stayed there until the end.
- **Temperature** rose from 60 °C at t = 5 s to 73–74 °C by about t = 70 s and held there.
- **Memory clock** was constant at 13801 MHz, and utilization was 100% throughout.

**Throttling.** No thermal throttling occurred: temperature plateaued at 73–74 °C. The card was **power-limited from
about 5 s into the run**, pinned at its 575 W ceiling, and the ~2.5% clock decline over the first ~60 s is the boost
algorithm trading frequency for power as temperature rises. I did not log NVIDIA's clock-event (throttle-reason)
flags during the run, so this classification rests on the power and temperature traces.

**Peak vs. steady-state throughput.** Achieved TFLOPS was not logged per sample, so I use graphics clock as a proxy.
This is reasonable because the workload is a fixed-size, compute-bound matmul at 100% utilization, whose throughput
scales with clock.

- Peak clock in the first 30 s: 2595 MHz. Mean clock in the final 5 minutes (t = 900–1200 s): 2528.8 MHz.
- **Steady-state / peak = 97.4%.** Using the mean of the first 30 s (2574.8 MHz) instead of the peak gives 98.2%.

---

## Part F — Table HW2.5.1

| Measurement | RTX 5090 (`GPU-a3a60fad-ef40-08fa-3eb7-9c22d773bd54`) | Notes |
|---|---:|---|
| Peak achieved TFLOPS (BF16) | 233.77 | N = 8192, best single run, `precision.jsonl` |
| % of theoretical peak (BF16) | 111.6% | vs. 209.5 TFLOPS dense FP32-accumulate [1]; card ran above the 2.41 GHz boost clock (Part B) |
| Effective bandwidth (GB/s) | 1526.3 (best 1568.3) | 85.2% (87.5%) of 1792 GB/s, `roofline.jsonl` |
| Naive attention OOM length | Not reached; largest success 16384 | Projected ~90k from the measured 4.0 B/L² fit, not tested |
| Fused attention OOM length | Not reached; largest success 16384 | Linear 1024 B/token; boundary impractically far |
| Steady-state / peak throughput | 97.4% (clock proxy) | Throughput not logged per sample, `thermal.csv` |
| Throttle onset (s, or none) | No thermal throttle; power-limited from ~5 s | 575 W cap reached at first sample; max 74 °C |

---

## Interpretation

The two numbers that bound everything later in the course disagree by three orders of magnitude on this card. It can
perform about 117 FLOPs in the time it moves one byte, so any operation below that intensity is limited by memory
rather than arithmetic. The elementwise add sits at 0.083 FLOP/byte and is purely a bandwidth test; large matmuls sit
at thousands of FLOP/byte and are purely compute tests. Precision matters only on the compute side: moving from FP32
CUDA cores to FP16/BF16 tensor cores gives about 3.4× the throughput at large N, but only once N is large enough to fill
the SMs. Attention shows why this matters: the naive implementation's memory grows as 4 bytes per L², which at batch 1
and a single head is still only 1 GiB at L = 16384, but multiplies by batch × heads × layers in a real model, while the
fused kernel keeps the same computation linear in memory. Finally, sustained performance on this card is set by its
575 W power limit, not by temperature, and costs only about 2–3% relative to a cold start.

## Limitations

- Repetition counts were low (5–10 for matmuls, 3 for attention) and no per-configuration variance was recorded; the
  two independent invocations differ by up to ~4% at the same configuration.
- Attention latencies include input generation inside the timed region.
- The committed plots draw a line connecting the end of one invocation to the start of the next, which causes the
  zig-zag lines in `precision_tflops.png` and `attention_memory.png`. The markers are the data.
- The card was also driving a display under the Windows WDDM driver during all measurements.
