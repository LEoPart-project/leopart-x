# Copyright (c) 2026 Nathan Sime
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Compare two benchmark result files, matching runs by configuration.

Prints a Markdown table for pasting into a pull request. Exits with
status 1 if memory per particle grew by more than the allowed fraction
(and, optionally, if time grew by more than its allowed fraction).

Example:
    python3 compare.py main.json branch.json --max-memory-increase 0.1
"""

import argparse
import pathlib
import sys

import results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base", type=pathlib.Path)
    parser.add_argument("new", type=pathlib.Path)
    parser.add_argument(
        "--max-memory-increase", type=float, default=None,
        help="Fail if bytes per particle grow by more than this fraction")
    parser.add_argument(
        "--memory-slack", type=float, default=8.0,
        help="Ignore memory changes smaller than this many bytes per "
             "particle (RSS is measured in whole pages)")
    parser.add_argument(
        "--max-time-increase", type=float, default=None,
        help="Fail if the median time grows by more than this fraction")
    return parser.parse_args()


def ratio(base, new):
    if base is None or new is None or base == 0:
        return None
    return new / base


def fmt_change(base, new, fmt):
    r = ratio(base, new)
    change = "" if r is None else f" ({(r - 1.0) * 100:+.0f}%)"
    return f"{fmt(base)} -> {fmt(new)}{change}"


def grew(base, new, fraction, slack=0.0):
    if fraction is None or base is None or new is None:
        return False
    return new - base > max(fraction * base, slack)


def main():
    args = parse_args()
    base = {results.key(r): r for r in results.load(args.base)}
    new = {results.key(r): r for r in results.load(args.new)}

    rows, regressions = [], []
    for k, rec in new.items():
        if k not in base:
            continue
        old = base[k]
        t0, t1 = old["time_s"]["median"], rec["time_s"]["median"]
        s0 = old["memory"]["storage_bytes_per_particle"]
        s1 = rec["memory"]["storage_bytes_per_particle"]
        p0 = old["memory"]["kernel_peak_bytes_per_particle"]
        p1 = rec["memory"]["kernel_peak_bytes_per_particle"]
        rows.append([k, fmt_change(t0, t1, results.fmt_seconds),
                     fmt_change(s0, s1, results.fmt_bytes),
                     fmt_change(p0, p1, results.fmt_bytes)])

        if grew(t0, t1, args.max_time_increase):
            regressions.append(f"time: {k}")
        for label, b, n in (("storage", s0, s1), ("kernel peak", p0, p1)):
            if grew(b, n, args.max_memory_increase, args.memory_slack):
                regressions.append(f"{label} memory: {k}")

    print(results.markdown_table(
        ["configuration", "median time", "storage B/p", "kernel peak B/p"],
        rows))

    only_base = sorted(set(base) - set(new))
    only_new = sorted(set(new) - set(base))
    for label, keys in (("Only in base", only_base),
                        ("Only in new", only_new)):
        if keys:
            print(f"\n{label}:")
            for k in keys:
                print(f"  {k}")

    if regressions:
        print("\nRegressions beyond the allowed increase:")
        for r in regressions:
            print(f"  {r}")
        sys.exit(1)


if __name__ == "__main__":
    main()
