# Copyright (c) 2026 Nathan Sime
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Run a matrix of benchmarks, each in its own process, and collect the
results in one JSON file.

Example:
    python3 run.py --num-ranks 1 2 --output results.json
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import results

BENCH_DIR = pathlib.Path(__file__).resolve().parent
BENCHMARKS = ["generate", "interpolate", "project", "relocate", "rk"]

# Extra arguments giving each variant of a benchmark
VARIANTS = {
    "relocate": [["--motion", "small"], ["--motion", "reflect"]],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS,
                        default=BENCHMARKS)
    parser.add_argument("--cell-types", nargs="+",
                        default=["triangle", "tetrahedron"])
    parser.add_argument("--num-ranks", nargs="+", type=int, default=[1])
    parser.add_argument("--size", default=None,
                        help="small, medium or large (default: small in "
                             "CI, else medium)")
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=1800.0,
                        help="Seconds allowed for each benchmark run")
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("results.json"))
    return parser.parse_args()


def benchmark_runs(args):
    for name in args.benchmarks:
        for variant in VARIANTS.get(name, [[]]):
            for cell_type in args.cell_types:
                for num_ranks in args.num_ranks:
                    bench_args = ["--cell-type", cell_type,
                                  "--degree", str(args.degree),
                                  "--repeats", str(args.repeats), *variant]
                    if args.size is not None:
                        bench_args += ["--size", args.size]
                    yield name, num_ranks, bench_args


def run_one(name, num_ranks, bench_args, timeout, output):
    cmd = [sys.executable, str(BENCH_DIR / f"bench_{name}.py"), *bench_args,
           "--output", str(output)]
    if num_ranks > 1:
        cmd = ["mpirun", "-np", str(num_ranks), *cmd]
    # Serve large allocations with mmap so that freed memory is returned
    # to the system at once and RSS measurements are not hidden by
    # memory glibc kept from earlier work
    env = {**os.environ, "MALLOC_MMAP_THRESHOLD_": "131072"}
    # Run from this directory: if the current directory is on the import
    # path (as an empty PYTHONPATH entry puts it), the repository root
    # would shadow the installed leopart package with its source tree
    subprocess.run(cmd, env=env, check=True, timeout=timeout, cwd=BENCH_DIR)
    return results.load(output)[0]


def main():
    args = parse_args()
    records, failures = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (name, num_ranks, bench_args) in enumerate(
                benchmark_runs(args)):
            label = f"{name} {' '.join(bench_args)} on {num_ranks} rank(s)"
            print(f"Running {label}", flush=True)
            output = pathlib.Path(tmp) / f"{i}.json"
            try:
                record = run_one(name, num_ranks, bench_args, args.timeout,
                                 output)
            except (subprocess.CalledProcessError,
                    subprocess.TimeoutExpired) as e:
                print(f"FAILED: {label}: {e}", flush=True)
                failures.append(label)
                continue
            records.append(record)

    args.output.write_text(json.dumps({"records": records}, indent=2) + "\n")
    print()
    print(results.markdown_table(
        results.SUMMARY_HEADER, [results.summary_row(r) for r in records]))
    print(f"\nWrote {len(records)} records to {args.output}")
    if failures:
        print(f"{len(failures)} benchmark run(s) failed:")
        for label in failures:
            print(f"  {label}")
        sys.exit(1)


if __name__ == "__main__":
    main()
