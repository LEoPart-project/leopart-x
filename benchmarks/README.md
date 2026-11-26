# LEoPart-x benchmarks

Timing and memory benchmarks for the main kernels. They exist to measure
the changes planned in [`ROADMAP.md`](../ROADMAP.md): each change should
show its gain, or at least no loss, here.

| Script | Kernel |
|---|---|
| `bench_generate.py` | `mesh_fill` plus constructing `Particles` |
| `bench_interpolate.py` | `transfer_to_particles` from a vector CG function |
| `bench_project.py` | `transfer_to_function` onto a scalar DG function |
| `bench_relocate.py` | moving particles, then `relocate_bbox` (`--motion small` or `reflect`) |
| `bench_rk.py` | one explicit Runge–Kutta step (`--rk-order 1` to `4`) |

## Running

Build and install LEoPart-x first (see the top-level README), then:

```bash
python3 benchmarks/run.py --num-ranks 1 2 --output results.json
```

`run.py` runs every benchmark on triangles and tetrahedra, each in its
own process, prints a summary table and writes all results to one JSON
file. Use `--benchmarks`, `--cell-types`, `--size`, `--degree` and
`--repeats` to choose what runs. A single benchmark can also be run
directly, with more options:

```bash
mpirun -np 4 python3 benchmarks/bench_relocate.py --cell-type hexahedron \
    --n 48 --ppc 20 --motion reflect --num-fields 5
```

Mesh sizes are presets of cells per direction of the unit square or cube:

| `--size` | 2D | 3D | Use |
|---|---|---|---|
| `small` | 64 | 16 | CI; seconds |
| `medium` | 256 | 32 | default outside CI |
| `large` | 1024 | 64 | workstation or cluster |

## What is measured

Each record holds the configuration, the environment (commit, DOLFINx and
LEoPart versions) and:

- **Time:** the median, minimum and maximum over `--repeats` calls, after
  one warm-up call. Each call's time is that of the slowest rank.
- **Throughput:** particles per second at the median time.
- **`storage_bytes_per_particle`:** growth in resident memory (RSS) when
  the particles are created, summed over ranks, per particle.
- **`kernel_peak_bytes_per_particle`:** the largest growth in RSS during
  one kernel call, summed over ranks, per particle. This shows temporary
  memory, such as per-particle work arrays. For `generate` it includes
  the new particles themselves.
- **Timers:** the `dolfinx::common::Timer` sections recorded inside
  LEoPart-x, as the slowest rank's mean time per call.

Memory is read from RSS because that is the only measure that sees C++
allocations. Two consequences:

- The peak is exact on Linux, where it is reset before each call
  (`kernel_peak_exact` in the record). Elsewhere it is only growth beyond
  earlier peaks, so may read low.
- RSS counts whole pages, so memory per particle is only meaningful with
  many particles. `run.py` makes glibc return large freed blocks to the
  system at once (`MALLOC_MMAP_THRESHOLD_`), so that memory freed by
  earlier work does not hide a kernel's temporary allocations.

## Comparing runs

```bash
python3 benchmarks/compare.py main.json branch.json
```

matches runs by configuration and prints a Markdown table of time and
memory changes, ready to paste into a pull request. Compare runs made on
the same machine. With `--max-memory-increase 0.1` it exits with an error
if memory per particle grows by more than 10% (ignoring changes smaller
than `--memory-slack`, 8 bytes per particle by default). Time can be
gated the same way with `--max-time-increase`, though timings on shared
machines vary too much to make that useful in CI.

## In CI

- **Pull requests** run every benchmark once on triangles, on 1 and 2
  ranks, so that a change cannot break the scripts unnoticed.
- **Nightly** runs the `small` suite on 1 and 2 ranks and uploads the
  results as the `benchmark-results` artifact. If
  `benchmarks/baseline.json` exists, it also fails when memory per
  particle has grown by more than 10% against it. To create or update the
  baseline, download a nightly artifact and commit it as
  `benchmarks/baseline.json`, so that the baseline comes from the same
  kind of machine as the runs it is compared with.
