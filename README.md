# HW2.5 GPU Assignment I

This repository contains a reproducible measurement harness and report artifacts for HW2.5. Run it on each reserved RTX 4090 or RTX 5090 workstation with the course PyTorch environment. Do not fill in measurements by inference from a specification sheet.

## Run order

```powershell
python benchmark_hw2_5.py capture-smi
python benchmark_hw2_5.py precision
python benchmark_hw2_5.py bandwidth
python benchmark_hw2_5.py attention
python benchmark_hw2_5.py thermal
python benchmark_hw2_5.py plot
```

The default thermal run is 1,200 seconds and samples every 5 seconds. Use `--duration` only for a documented pilot run; the final submission must use the full duration. Preserve each machine's `results` directory under a UUID-labelled directory and record the exact commands in `RUN_LOG.txt`.

The script records raw measurements in `results/`. Complete the report only with values traceable to those records.