"""HW2.5 measurement harness for CUDA GPUs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DEFAULT_SIZES = [1024, 4096, 8192, 16384]
ATTENTION_SIZES = [512, 1024, 2048, 4096, 8192, 16384]


def gpu_uuid() -> str:
    return str(getattr(torch.cuda.get_device_properties(0), "uuid", "unknown"))


def gpu_metadata() -> dict:
    properties = torch.cuda.get_device_properties(0)
    return {"uuid": gpu_uuid(), "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda}


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run on the reserved RTX workstation.")


def timed_cuda(fn, repetitions: int, warmups: int = 5) -> float:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repetitions / 1000.0


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def precision_benchmark(args: argparse.Namespace) -> None:
    require_cuda()
    results = []
    for size in args.sizes:
        for precision in ("fp32", "tf32", "fp16", "bf16"):
            dtype = {"fp32": torch.float32, "tf32": torch.float32,
                     "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
            torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
            left = torch.randn((size, size), device="cuda", dtype=dtype)
            right = torch.randn((size, size), device="cuda", dtype=dtype)
            repetitions = args.repetitions if size < 8192 else max(1, args.repetitions // 2)
            seconds = timed_cuda(lambda: torch.mm(left, right), repetitions)
            results.append({**gpu_metadata(), "kind": "matmul", "size": size,
                            "precision": precision, "repetitions": repetitions,
                            "seconds_per_matmul": seconds,
                            "achieved_tflops": 2.0 * size**3 / seconds / 1e12,
                            "tf32_enabled": torch.backends.cuda.matmul.allow_tf32})
            del left, right
            torch.cuda.empty_cache()
    append_jsonl(RESULTS / "precision.jsonl", results)


def bandwidth_benchmark(args: argparse.Namespace) -> None:
    require_cuda()
    elements = args.elements
    left = torch.randn(elements, device="cuda", dtype=torch.float32)
    right = torch.randn(elements, device="cuda", dtype=torch.float32)
    output = torch.empty_like(left)
    seconds = timed_cuda(lambda: torch.add(left, right, out=output), args.repetitions)
    bandwidth = elements * 3 * left.element_size() / seconds / 1e9
    size = args.matmul_size
    matrix_a = torch.randn((size, size), device="cuda", dtype=torch.float16)
    matrix_b = torch.randn((size, size), device="cuda", dtype=torch.float16)
    matmul_seconds = timed_cuda(lambda: torch.mm(matrix_a, matrix_b), args.repetitions)
    matmul_flops = 2 * size**3
    matmul_bytes = 3 * size**2 * matrix_a.element_size()
    append_jsonl(RESULTS / "roofline.jsonl", [{**gpu_metadata(), "kind": "elementwise_add",
        "elements": elements, "repetitions": args.repetitions, "seconds": seconds,
        "effective_bandwidth_gb_s": bandwidth, "arithmetic_intensity_flops_per_byte": 1 / 12},
        {**gpu_metadata(), "kind": "square_matmul", "size": size,
         "repetitions": args.repetitions, "seconds": matmul_seconds,
         "achieved_tflops": matmul_flops / matmul_seconds / 1e12,
         "arithmetic_intensity_flops_per_byte": matmul_flops / matmul_bytes}])


def attention_once(length: int, head_dim: int, batch: int, fused: bool) -> None:
    query = torch.randn(batch, length, head_dim, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    scale = 1.0 / math.sqrt(head_dim)
    if fused:
        torch.nn.functional.scaled_dot_product_attention(
            query[:, None], key[:, None], value[:, None], scale=scale)
    else:
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1)
        torch.matmul(weights, value)


def attention_benchmark(args: argparse.Namespace) -> None:
    require_cuda()
    rows = []
    for fused in (False, True):
        for length in args.sizes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                seconds = timed_cuda(lambda: attention_once(length, args.head_dim, args.batch, fused),
                                     args.repetitions, warmups=1)
                rows.append({**gpu_metadata(), "kind": "attention", "fused": fused,
                    "length": length, "batch": args.batch, "head_dim": args.head_dim,
                    "seconds": seconds, "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                    "oom": False})
            except RuntimeError as error:
                if "out of memory" not in str(error).lower():
                    raise
                rows.append({**gpu_metadata(), "kind": "attention", "fused": fused,
                    "length": length, "batch": args.batch, "head_dim": args.head_dim,
                    "seconds": None, "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                    "oom": True})
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
    append_jsonl(RESULTS / "attention.jsonl", rows)


def thermal_benchmark(args: argparse.Namespace) -> None:
    require_cuda()
    RESULTS.mkdir(exist_ok=True)
    left = torch.randn((args.matmul_size, args.matmul_size), device="cuda", dtype=torch.float16)
    right = torch.randn_like(left)
    start = time.monotonic()
    next_sample = start
    log_path = RESULTS / "thermal.csv"
    fields = ["elapsed_s", "gpu_uuid", "clock_mhz", "memory_clock_mhz", "temperature_c", "power_w", "utilization_pct"]
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        while time.monotonic() - start < args.duration:
            torch.mm(left, right)
            now = time.monotonic()
            if now >= next_sample:
                values = subprocess.run(["nvidia-smi", "--query-gpu=uuid,clocks.gr,clocks.mem,temperature.gpu,power.draw,utilization.gpu", "--format=csv,noheader,nounits", "-i", "0"], capture_output=True, text=True, check=True).stdout.strip().split(",")
                writer.writerow(dict(zip(fields, [round(now - start, 3), *(value.strip() for value in values)])))
                stream.flush()
                next_sample += args.interval


def capture_smi(_: argparse.Namespace) -> None:
    RESULTS.mkdir(exist_ok=True)
    output = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True, check=True).stdout
    (RESULTS / "nvidia-smi-q.txt").write_text(output, encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def plot_results(_: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    precision_rows = read_jsonl(RESULTS / "precision.jsonl")
    if precision_rows:
        for precision in ("fp32", "tf32", "fp16", "bf16"):
            rows = [row for row in precision_rows if row["precision"] == precision]
            plt.plot([row["size"] for row in rows], [row["achieved_tflops"] for row in rows], marker="o", label=precision)
        plt.xlabel("Matrix size N")
        plt.ylabel("Achieved TFLOPS")
        plt.xscale("log", base=2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS / "precision_tflops.png", dpi=160)
        plt.close()

    attention_rows = read_jsonl(RESULTS / "attention.jsonl")
    if attention_rows:
        for fused in (False, True):
            rows = [row for row in attention_rows if row["fused"] == fused and not row["oom"]]
            plt.plot([row["length"] for row in rows], [row["peak_memory_bytes"] / 2**30 for row in rows], marker="o", label="fused" if fused else "naive")
        successful = [row for row in attention_rows if not row["oom"]]
        if len(successful) >= 3:
            import numpy as np
            lengths = np.array([row["length"] for row in successful], dtype=float)
            memory = np.array([row["peak_memory_bytes"] for row in successful], dtype=float)
            coefficients = np.polyfit(lengths**2, memory, 1)
            (RESULTS / "attention_fit.txt").write_text(f"peak_memory_bytes = {coefficients[0]:.9g} * length^2 + {coefficients[1]:.9g}\n", encoding="utf-8")
        plt.xlabel("Sequence length")
        plt.ylabel("Peak allocated memory (GiB)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS / "attention_memory.png", dpi=160)
        plt.close()

    thermal_path = RESULTS / "thermal.csv"
    if thermal_path.exists():
        with thermal_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        figure, clock_axis = plt.subplots()
        temperatures = [float(row["temperature_c"]) for row in rows]
        clock_axis.plot([float(row["elapsed_s"]) for row in rows], [float(row["clock_mhz"]) for row in rows], label="graphics clock")
        clock_axis.set_xlabel("Elapsed time (s)")
        clock_axis.set_ylabel("Graphics clock (MHz)")
        temperature_axis = clock_axis.twinx()
        temperature_axis.plot([float(row["elapsed_s"]) for row in rows], temperatures, color="tab:red", label="temperature")
        temperature_axis.set_ylabel("Temperature (C)")
        figure.tight_layout()
        figure.savefig(RESULTS / "thermal_clock_temperature.png", dpi=160)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    precision = subparsers.add_parser("precision")
    precision.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    precision.add_argument("--repetitions", type=int, default=10)
    bandwidth = subparsers.add_parser("bandwidth")
    bandwidth.add_argument("--elements", type=int, default=256 * 1024 * 1024)
    bandwidth.add_argument("--matmul-size", type=int, default=8192)
    bandwidth.add_argument("--repetitions", type=int, default=10)
    attention = subparsers.add_parser("attention")
    attention.add_argument("--sizes", nargs="+", type=int, default=ATTENTION_SIZES)
    attention.add_argument("--head-dim", type=int, default=128)
    attention.add_argument("--batch", type=int, default=1)
    attention.add_argument("--repetitions", type=int, default=3)
    thermal = subparsers.add_parser("thermal")
    thermal.add_argument("--duration", type=int, default=1200)
    thermal.add_argument("--interval", type=int, default=5)
    thermal.add_argument("--matmul-size", type=int, default=8192)
    subparsers.add_parser("capture-smi")
    subparsers.add_parser("plot")
    args = parser.parse_args()
    handlers = {"precision": precision_benchmark, "bandwidth": bandwidth_benchmark,
                "attention": attention_benchmark, "thermal": thermal_benchmark,
                "capture-smi": capture_smi, "plot": plot_results}
    handlers[args.command](args)


if __name__ == "__main__":
    main()