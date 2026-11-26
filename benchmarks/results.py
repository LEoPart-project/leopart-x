# Copyright (c) 2026 Nathan Sime
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Reading and formatting benchmark results. Deliberately free of MPI
and DOLFINx imports, so that run.py can launch mpirun."""

import json

# Configuration entries derived from others, left out of the key
_DERIVED = {"num_cells"}

# Configuration entries shown in their own summary columns or common to
# all benchmarks; the rest describe the variant of a benchmark
_NOT_VARIANT = {"cell_type", "n", "num_cells", "num_ranks", "degree", "ppc",
                "space", "value_shape", "kernel"}


def load(path):
    """Records from a results file written by run.py, or a single record
    written by a benchmark script."""
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "records" in data:
        return data["records"]
    if isinstance(data, dict):
        return [data]
    return data


def key(record):
    """Identifies a benchmark configuration across runs."""
    config = {k: v for k, v in record["config"].items() if k not in _DERIVED}
    items = " ".join(f"{k}={config[k]}" for k in sorted(config))
    return f"{record['benchmark']} {items}"


def fmt_bytes(value):
    return "-" if value is None else f"{value:.0f}"


def fmt_seconds(value):
    if value is None:
        return "-"
    if value < 1e-3:
        return f"{value * 1e6:.0f} us"
    if value < 1.0:
        return f"{value * 1e3:.1f} ms"
    return f"{value:.2f} s"


def summary_row(record):
    config = record["config"]
    memory = record["memory"]
    rate = record["particles_per_s"]
    variant = " ".join(f"{k}={v}" for k, v in sorted(config.items())
                       if k not in _NOT_VARIANT)
    return [
        record["benchmark"],
        variant or "-",
        config["cell_type"],
        str(config["num_ranks"]),
        f"{record['num_particles']:,}",
        fmt_seconds(record["time_s"]["median"]),
        "-" if rate is None else f"{rate / 1e6:.2f}",
        fmt_bytes(memory["storage_bytes_per_particle"]),
        fmt_bytes(memory["kernel_peak_bytes_per_particle"]),
    ]


SUMMARY_HEADER = ["benchmark", "variant", "cell", "ranks", "particles",
                  "median", "Mparticles/s", "storage B/p", "kernel peak B/p"]


def markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)
