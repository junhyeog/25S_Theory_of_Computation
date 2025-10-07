# benchmark_lcr.py
"""
Benchmark harness for the *linear‑time* Longest Common Repeat implementation
===========================================================================
This **v2** adds error bars (±1 σ) to the runtime plot and records the
standard deviation in the CSV.  Everything else—CLI usage, random‐case
generation—remains identical.

Usage
-----
```bash
python3 benchmark_lcr.py          # pops up a log‑log plot
```
A file `timings.csv` with three columns (total_len, avg_time_s, std_time_s)
is written next to the script.  Adjust the `TOTAL_LENGTHS`, `NUM_STRINGS_…`,
`REPEATS`, and `K` constants to suit your experiment.
"""

from __future__ import annotations

import random
import statistics
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt

from main import longest_common_repeat

###############################################################################
# Experiment parameters                                                       #
###############################################################################

# TOTAL_LENGTHS: List[int] = [1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000]
# TOTAL_LENGTHS: List[int] = [1000*i for i in range(1, 101)]
TOTAL_LENGTHS: List[int] = [10*i for i in range(1, 101)]
NUM_STRINGS_CHOICES: List[int] = [2, 3, 4, 5, 6, 7, 8, 9]
K: int = 2           # repeat must appear in ≥k strings
REPEATS: int = 30     # independent trials per total length
ALPHABET: str = "ACGT"
CSV_PATH = Path("timings.csv")

###############################################################################
# Helper functions                                                            #
###############################################################################

def random_dna(n: int) -> str:
    return "".join(random.choice(ALPHABET) for _ in range(n))


def distribute_lengths(total: int, parts: int) -> List[int]:
    base, rem = divmod(total, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def run_single_case(total_len: int, num_strings: int, k: int) -> float:
    lens = distribute_lengths(total_len, num_strings)
    strings = [random_dna(m) for m in lens]
    start = time.perf_counter()
    if k == -1:
        k = random.randint(2, num_strings) # sample k from [2, num_strings]
    longest_common_repeat(strings, k)
    return time.perf_counter() - start


def run_benchmark() -> List[Tuple[int, float, float]]:
    rows: List[Tuple[int, float, float]] = []
    print(f"Running {REPEATS} trials for each Σ|Tᵢ| …")
    for N in TOTAL_LENGTHS:
        times = [
            # run_single_case(N, random.choice(NUM_STRINGS_CHOICES), K)
            run_single_case(N, random.choice(NUM_STRINGS_CHOICES), -1)
            for _ in range(REPEATS)
        ]
        avg = statistics.mean(times)
        std = statistics.stdev(times) if REPEATS > 1 else 0.0
        rows.append((N, avg, std))
        print(f"Σ|Tᵢ|={N:>7}  avg={avg:.6f}s  ±σ={std:.6f}")
    return rows


def save_csv(rows: List[Tuple[int, float, float]]):
    with CSV_PATH.open("w", encoding="utf-8") as f:
        f.write("total_len,avg_time_s,std_time_s\n")
        for N, avg, std in rows:
            f.write(f"{N},{avg},{std}\n")
    print(f"CSV saved to {CSV_PATH}")


# def plot(rows: List[Tuple[int, float, float]]):
#     xs = [N for N, _, _ in rows]
#     avgs = [avg for _, avg, _ in rows]
#     stds = [std for _, _, std in rows]

#     plt.figure()
#     plt.errorbar(xs, avgs, yerr=stds, fmt="o-", linewidth=2, capsize=5)
#     # plt.xscale("log")
#     # plt.yscale("log")
#     plt.xscale("linear")
#     plt.yscale("linear")
#     plt.xlabel("Total length Σ|Tᵢ|")
#     plt.ylabel("Average runtime ±1σ (s)")
#     plt.title("Average runtime vs. total input length")
#     plt.grid(True, which="both", linestyle=":", alpha=0.6)
#     plt.tight_layout()
#     plt.show()
#     plt.savefig("benchmark_lcr.png", dpi=300)


def plot(rows: List[Tuple[int, float, float]]):
    xs = [total for total, _, _ in rows]
    avgs = [avg for _, avg, _ in rows]
    stds = [std for _, _, std in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, avgs, marker="o", linewidth=2, label="average runtime")
    # Shade ±1 σ band
    lower = [a - s for a, s in zip(avgs, stds)]
    upper = [a + s for a, s in zip(avgs, stds)]
    # ax.fill_between(xs, lower, upper, alpha=0.25, label="±1 σ")
    ax.fill_between(xs, lower, upper, alpha=0.25)

    plt.xscale("linear")
    plt.yscale("linear")
    ax.set_xlabel("Total length Σ|Tᵢ|")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Average runtime vs. Total input length")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.savefig("benchmark_lcr.png", dpi=300)

###############################################################################
# Entry‐point                                                                 #
###############################################################################

if __name__ == "__main__":
    data = run_benchmark()
    save_csv(data)
    plot(data)
