from __future__ import annotations

import argparse
import os
import random
import statistics
import time
from pathlib import Path
from typing import List, Tuple

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
verbose: bool = args.verbose


# BASE_LENGTH: int = 1000
# TOTAL_LENGTHS: List[int] = [BASE_LENGTH * i for i in range(1, 101)]
TOTAL_LENGTHS: List[int] = [BASE_LENGTH * i for i in range(1, 11)]
NUM_STRINGS_CHOICES: List[int] = [2, 3, 4, 5, 6, 7, 8, 9]
# K: int = 2  # repeat must appear in ≥k strings
# REPEATS: int = 30  # independent trials per total length
ALPHABET: str = "ACGT"
this_file_name = Path(__file__).name.replace(".py", "")
CSV_PATH = Path(f"{this_file_name}_{BASE_LENGTH}_{REPEATS}_{K}.csv")

###############################################################################
# Helper functions                                                            #
###############################################################################


def random_dna(n: int) -> str:
    return "".join(random.choice(ALPHABET) for _ in range(n))


def distribute_lengths(total: int, parts: int) -> List[int]:
    base, rem = divmod(total, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def run_single_case(total_len: int, num_strings: int, k: int, verbose=True) -> float:
    lens = distribute_lengths(total_len, num_strings)
    strings = [random_dna(m) for m in lens]
    start = time.perf_counter()
    if k == -1:
        k = random.randint(2, num_strings)  # sample k from [2, num_strings]

    # Get current process
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # Convert to MB

    result = longest_common_repeat(strings, k, verbose=verbose)
    end = time.perf_counter()

    mem_after = process.memory_info().rss / 1024 / 1024  # Convert to MB
    mem_used = mem_after - mem_before

    checker_result = check_answer(strings, k, result, verbose=verbose)
    if not checker_result:
        raise ValueError(f"[!] Checker failed")

    return checker_result, end - start, mem_used


def run_checker() -> List[Tuple[int, float, float]]:
    rows = []
    print(f"Running {REPEATS} trials for each Σ|Tᵢ| …")
    for N in TOTAL_LENGTHS:
        results = [run_single_case(N, random.choice(NUM_STRINGS_CHOICES), K, verbose) for _ in range(REPEATS)]
        avg_accuracy = statistics.mean(1 if res else 0 for res, _, _ in results) * 100
        avg_time = statistics.mean(t for _, t, _ in results)
        std_time = statistics.stdev(t for _, t, _ in results) if REPEATS > 1 else 0.0
        avg_mem = statistics.mean(m for _, _, m in results)
        std_mem = statistics.stdev(m for _, _, m in results) if REPEATS > 1 else 0.0

        rows.append((N, avg_accuracy, avg_time, std_time, avg_mem, std_mem))
        print(
            f"Σ|Tᵢ|={N:>7}  avg_accuracy={avg_accuracy:.2f}%  avg_time={avg_time:.6f}s  ±σ={std_time:.6f}s  avg_mem={avg_mem:.2f}MB ±σ={std_mem:.2f}MB"
        )
    return rows


def save_csv(rows: List[Tuple[int, float, float]]):
    with CSV_PATH.open("w", encoding="utf-8") as f:
        f.write("total_len,avg_time,std_time,avg_mem,std_mem\n")
        for row in rows:
            for i in range(len(row) - 1):
                f.write(f"{row[i]},")
            f.write(f"{row[-1]}\n")
    print(f"CSV saved to {CSV_PATH}")


if __name__ == "__main__":
    data = run_checker()
    save_csv(data)
