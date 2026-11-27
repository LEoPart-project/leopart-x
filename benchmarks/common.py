# Copyright (c) 2026 Nathan Sime
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Shared harness for the LEoPart-x benchmarks.

Each benchmark script defines a ``Benchmark`` subclass and calls ``main``.
The harness builds the mesh, times the kernel over several repeats and
writes one JSON record with the configuration, timings and memory use.

Memory is measured from the process's resident set size (RSS), the only
measure that sees C++ allocations:

- ``storage_bytes_per_particle``: RSS growth when the particles are
  created, summed over ranks, per particle. It includes the particle
  storage and its indexing structures.
- ``kernel_peak_bytes_per_particle``: the largest RSS growth while the
  kernel runs, summed over ranks, per particle. It shows temporary
  memory, such as per-particle work arrays.

The peak is exact on Linux, where it can be reset before each kernel
call. Elsewhere it falls back to the process high-water mark, which only
shows growth beyond earlier peaks. Run each benchmark in its own process
(``run.py`` does) so that earlier work cannot hide a kernel's peak.
"""

import argparse
import gc
import importlib.metadata
import json
import os
import pathlib
import platform
import resource
import statistics
import subprocess
import sys
import time

from mpi4py import MPI

import numpy as np

import dolfinx
import leopart.cpp as pyleopart

CELL_TYPES = {
    "triangle": dolfinx.mesh.CellType.triangle,
    "quadrilateral": dolfinx.mesh.CellType.quadrilateral,
    "tetrahedron": dolfinx.mesh.CellType.tetrahedron,
    "hexahedron": dolfinx.mesh.CellType.hexahedron,
}

# Cells in each direction of the unit square or cube, by topological
# dimension. "small" runs in seconds and is used in CI.
SIZES = {
    "small": {2: 64, 3: 16},
    "medium": {2: 256, 3: 32},
    "large": {2: 1024, 3: 64},
}

# Timer sections recorded by the C++ code (dolfinx::common::Timer)
_OWNERSHIP = "leopart::Particles::determine_point_ownership"
LEOPART_TIMERS = [
    "leopart::advect::rk",
    "leopart::Particles::relocate_bbox",
    "leopart::Particles::relocate_bbox_on_proc",
    "leopart::Particles::relocate_bbox_on_proc collisions",
    "leopart::Particles::relocate_bbox_on_proc post process",
    "leopart::Particles::sort_by_cell",
    _OWNERSHIP,
    f"{_OWNERSHIP} global collisions",
    f"{_OWNERSHIP} Send geometry to global process colliding ranks",
    f"{_OWNERSHIP} Compute closest entities on owning ranks",
    f"{_OWNERSHIP} Communicate reverse the squared differences",
    f"{_OWNERSHIP} Find owning process minimise square dist",
    f"{_OWNERSHIP} Communicate forward destination ranks",
    (f"{_OWNERSHIP} On the destination ranks select the appropriate data "
     "and return"),
]


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def current_rss():
    """Current resident set size in bytes, or None if unavailable."""
    statm = pathlib.Path("/proc/self/statm")
    if statm.exists():
        resident_pages = int(statm.read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    return None


def reset_peak_rss():
    """Reset the RSS high-water mark. Returns True if supported (Linux)."""
    try:
        pathlib.Path("/proc/self/clear_refs").write_text("5")
    except OSError:
        return False
    return True


def peak_rss():
    """RSS high-water mark in bytes."""
    status = pathlib.Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes
    return maxrss if sys.platform == "darwin" else maxrss * 1024


def rss_now():
    """Current RSS where available, else the high-water mark."""
    rss = current_rss()
    return rss if rss is not None else peak_rss()


# ---------------------------------------------------------------------------
# Benchmark definition
# ---------------------------------------------------------------------------

class Benchmark:
    """A kernel to time.

    Subclasses set ``name`` and override the methods below. ``state`` is a
    dict shared between them.
    """

    name = None

    def add_arguments(self, parser):
        """Add benchmark specific command line arguments."""

    def setup(self, mesh, args, state):
        """Create everything except the particles (spaces, functions)."""

    def create_particles(self, mesh, args, state):
        """Create the particles. Return them, or None if the kernel
        creates them itself."""

    def prepare(self, mesh, args, state):
        """Untimed work before each kernel call."""

    def run(self, mesh, args, state):
        """The timed kernel."""
        raise NotImplementedError

    def config(self, args, state):
        """Benchmark specific configuration to record."""
        return {}


def default_np_per_cell(args, default):
    return args.ppc if args.ppc is not None else default


def fill_mesh(mesh, np_per_cell):
    """Particles filling the mesh, with gdim position components."""
    x, cells = pyleopart.mesh_fill(mesh._cpp_object, np_per_cell, seed=1)
    return pyleopart.Particles(np.ascontiguousarray(x), cells)


def cellular_flow(x, gdim):
    """Divergence free velocity with no flow through the boundary of the
    unit square or cube, so advected particles stay in the domain."""
    u = np.zeros((gdim, x.shape[1]))
    u[0] = np.sin(np.pi * x[0]) * np.cos(np.pi * x[1])
    u[1] = -np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
    return u


def velocity_function(mesh, degree):
    """The cellular flow interpolated into a vector CG space."""
    gdim = mesh.geometry.dim
    V = dolfinx.fem.functionspace(mesh, ("CG", degree, (gdim,)))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: cellular_flow(x, gdim))
    u.x.scatter_forward()
    return u


def num_particles_global(comm, ptcls):
    return comm.allreduce(len(ptcls.active_pidxs()), op=MPI.SUM)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def default_size():
    return "small" if os.environ.get("GITHUB_ACTIONS") else "medium"


def parse_args(bench, argv):
    parser = argparse.ArgumentParser(description=f"Benchmark: {bench.name}")
    parser.add_argument("--cell-type", choices=CELL_TYPES, default="triangle")
    parser.add_argument(
        "--size", choices=SIZES, default=default_size(),
        help="Preset mesh size (default: small in CI, else medium)")
    parser.add_argument(
        "--n", type=int, default=None,
        help="Cells in each direction; overrides --size")
    parser.add_argument("--ppc", type=int, default=None,
                        help="Particles per cell")
    parser.add_argument("--degree", type=int, default=1,
                        help="Polynomial degree of the function space")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=pathlib.Path, default=None,
                        help="JSON file to write (default: stdout)")
    bench.add_arguments(parser)
    return parser.parse_args(argv)


def create_mesh(comm, cell_type, n):
    if dolfinx.mesh.cell_dim(cell_type) == 2:
        return dolfinx.mesh.create_unit_square(
            comm, n, n, cell_type=cell_type)
    return dolfinx.mesh.create_unit_cube(
        comm, n, n, n, cell_type=cell_type)


def git_commit():
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=pathlib.Path(__file__).parent, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def collect_timers(comm):
    timers = {}
    for name in LEOPART_TIMERS:
        try:
            count, elapsed = dolfinx.common.timing(name)
        except RuntimeError:
            count, elapsed = 0, None
        seconds = elapsed.total_seconds() if elapsed is not None else 0.0
        count = comm.allreduce(count, op=MPI.MAX)
        seconds = comm.allreduce(seconds, op=MPI.MAX)
        if count > 0:
            timers[name] = {"count": count, "total_s": seconds,
                            "mean_s": seconds / count}
    return timers


def measure(comm, mesh, args, bench, state):
    """Run the kernel once, timed. Returns (seconds, peak bytes), where
    seconds is the slowest rank's time and bytes the summed peaks."""
    bench.prepare(mesh, args, state)
    gc.collect()
    comm.Barrier()
    peak_resettable = reset_peak_rss()
    rss0 = rss_now()
    t0 = time.perf_counter()
    bench.run(mesh, args, state)
    elapsed = time.perf_counter() - t0
    peak = max(peak_rss() - rss0, 0)
    elapsed = comm.allreduce(elapsed, op=MPI.MAX)
    peak = comm.allreduce(peak, op=MPI.SUM)
    exact = comm.allreduce(peak_resettable, op=MPI.LAND)
    return elapsed, peak, exact


def main(bench, argv=None):
    args = parse_args(bench, sys.argv[1:] if argv is None else argv)
    comm = MPI.COMM_WORLD
    cell_type = CELL_TYPES[args.cell_type]
    tdim = dolfinx.mesh.cell_dim(cell_type)
    n = args.n if args.n is not None else SIZES[args.size][tdim]

    mesh = create_mesh(comm, cell_type, n)
    num_cells = mesh.topology.index_map(tdim).size_global
    state = {"n": n}
    bench.setup(mesh, args, state)

    gc.collect()
    rss_before = rss_now()
    ptcls = bench.create_particles(mesh, args, state)
    gc.collect()
    storage = comm.allreduce(max(rss_now() - rss_before, 0), op=MPI.SUM)
    state["ptcls"] = ptcls

    # Warm up, then time
    bench.prepare(mesh, args, state)
    bench.run(mesh, args, state)
    times, peaks, exact = [], [], True
    for _ in range(args.repeats):
        elapsed, peak, peak_exact = measure(comm, mesh, args, bench, state)
        times.append(elapsed)
        peaks.append(peak)
        exact = exact and peak_exact

    n_particles = num_particles_global(comm, state["ptcls"])
    median = statistics.median(times)
    record = {
        "benchmark": bench.name,
        "config": {
            "cell_type": args.cell_type,
            "n": n,
            "num_cells": num_cells,
            "num_ranks": comm.size,
            "degree": args.degree,
            **bench.config(args, state),
        },
        "num_particles": n_particles,
        "time_s": {"median": median, "min": min(times), "max": max(times),
                   "repeats": times},
        "particles_per_s": n_particles / median if median > 0 else None,
        "memory": {
            "storage_bytes_per_particle":
                storage / n_particles if ptcls is not None else None,
            "kernel_peak_bytes_per_particle": max(peaks) / n_particles,
            "kernel_peak_exact": exact,
        },
        "timers": collect_timers(comm),
        "environment": {
            "commit": git_commit(),
            "leopart": importlib.metadata.version("leopart"),
            "dolfinx": dolfinx.__version__,
            "python": platform.python_version(),
            "machine": platform.machine(),
            "system": platform.system(),
        },
    }

    if comm.rank == 0:
        text = json.dumps(record, indent=2)
        if args.output is None:
            print(text)
        else:
            args.output.write_text(text + "\n")
