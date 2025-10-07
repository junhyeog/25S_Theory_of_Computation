from __future__ import annotations

import argparse
import gc
import os
import random
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import psutil

from checker import check_answer
from main import longest_common_repeat

# Experiment parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Run random checker for longest common repeat.")
    parser.add_argument(
        "--base_length",
        "-b",
        type=int,
        default=1000,
        help="Base length for total lengths (default: 1000).",
    )
    parser.add_argument(
        "--repeats",
        "-r",
        type=int,
        default=30,
        help="Number of independent trials per total length (default: 30).",
    )
    parser.add_argument(
        "--k",
        "-k",
        type=int,
        default=-1,
        help="Minimum number of strings a repeat must appear in (default: 2).",
    )
    parser.add_argument(
        "--p",
        "-p",
        type=float,
        default=0.0,
        help="Probability of generating a string with only 'A's (default: 0.0).",
    )
    parser.add_argument(
        "--check",
        "-c",
        action="store_true",
        help="Run the checker to verify correctness of the longest common repeat function.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output for debugging.",
    )
    return parser.parse_args()


args = parse_args()
BASE_LENGTH: int = args.base_length
REPEATS: int = args.repeats
K: int = args.k
p: float = args.p
check: bool = args.check
verbose: bool = args.verbose

TOTAL_LENGTHS: List[int] = [BASE_LENGTH * i for i in range(1, 101)]
NUM_STRINGS_CHOICES: List[int] = [2, 3, 4, 5, 6, 7, 8, 9]
ALPHABET: str = "ACGT"
this_file_name = Path(__file__).name.replace(".py", "")
save_file_name = f"{this_file_name}_B_{BASE_LENGTH}_R_{REPEATS}_K_{K}_p_{p}"

###############################################################################
# Helper functions                                                            #
###############################################################################


def random_dna(n: int, p: float = 0.2) -> str:
    if p < random.random():
        return "".join(random.choice("A") for _ in range(n))
    return "".join(random.choice(ALPHABET) for _ in range(n))


def distribute_lengths(total: int, parts: int) -> List[int]:
    base, rem = divmod(total, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def run_single_case(total_len: int, num_strings: int, k: int, p: float, check: False, verbose: False) -> float:
    lens = distribute_lengths(total_len, num_strings)
    strings = [random_dna(m, p) for m in lens]
    if k == -1:
        k = random.randint(2, num_strings)

    gc.collect()

    tracemalloc.start()
    start = time.perf_counter()

    ans = longest_common_repeat(strings, k, verbose=verbose)

    end_time = time.perf_counter()
    run_time = end_time - start
    _, peak_memory = tracemalloc.get_traced_memory()  # peak memory in bytes
    tracemalloc.stop()

    mem_used = peak_memory / (1024 * 1024)  # Convert to MB

    if check:
        checker_result = check_answer(strings, k, ans, verbose=verbose)
        if not checker_result:
            raise ValueError(f"[!] Checker failed")

    return run_time, mem_used


def run_benchmark() -> List[Tuple[int, float, float]]:
    rows: List[Tuple[int, float, float]] = []
    print(f"Running {REPEATS} trials for each Σ|Tᵢ| …")
    for N in TOTAL_LENGTHS:
        results = [run_single_case(N, random.choice(NUM_STRINGS_CHOICES), K, p, check, verbose) for _ in range(REPEATS)]
        times = [result[0] for result in results]
        mems = [result[1] for result in results]
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if REPEATS > 1 else 0.0
        avg_mem = statistics.mean(mems)
        std_mem = statistics.stdev(mems) if REPEATS > 1 else 0.0

        rows.append((N, avg_time, std_time, avg_mem, std_mem))
        print(f"Σ|Tᵢ|={N:>7}  avg={avg_time:.6f}s  ±σ={std_time:.6f}s  avg_mem={avg_mem:.2f}MB  ±σ={std_mem:.2f}MB")
    return rows


def save_csv(rows: List[Tuple[int, float, float]]):
    CSV_PATH = Path(f"{save_file_name}.csv")
    with CSV_PATH.open("w", encoding="utf-8") as f:
        f.write("total_len,avg_time,std_time,avg_mem,std_mem\n")
        for row in rows:
            for i in range(len(row) - 1):
                f.write(f"{row[i]},")
            f.write(f"{row[-1]}\n")
    print(f"CSV saved to {CSV_PATH}")


def plot(xs, avgs, stds, xlabel="Total length Σ|Tᵢ|", ylabel="Runtime (s)", file_name_suffix=""):
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.rcParams.update({"font.size": 14})

    ax.plot(xs, avgs, marker="o", linewidth=2)
    lower = [a - s for a, s in zip(avgs, stds)]
    upper = [a + s for a, s in zip(avgs, stds)]
    ax.fill_between(xs, lower, upper, alpha=0.25)

    plt.xscale("linear")
    plt.yscale("linear")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    # ax.legend(fontsize=14)
    fig.tight_layout()
    plt.show()
    plt.savefig(f"{save_file_name}_{file_name_suffix}.png", dpi=300)
    print(f"Plot saved: {save_file_name}_{file_name_suffix}")


###############################################################################
# Entry‐point                                                                 #
###############################################################################

if __name__ == "__main__":

    print(f"Recursion limit: {sys.getrecursionlimit()}")
    data = run_benchmark()
    save_csv(data)
    plot(
        [row[0] for row in data],
        [row[1] for row in data],
        [row[2] for row in data],
        file_name_suffix="runtime",
    )
    plot(
        [row[0] for row in data],
        [row[3] for row in data],
        [row[4] for row in data],
        ylabel="Memory usage (MB)",
        file_name_suffix="memory",
    )
