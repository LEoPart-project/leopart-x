# Copyright (c) 2026 Nathan Sime
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

# Run each demo as a separate process from within its own directory, so
# that relative input files (e.g. demo_image/leopard.jpg) are found.
#
# This module deliberately does not import mpi4py: an MPI initialised
# process cannot reliably launch further MPI jobs with mpirun.

import os
import pathlib
import subprocess
import sys

import pytest

demo_dir = pathlib.Path(__file__).resolve().parent
demos = sorted(demo_dir.rglob("demo_*.py"))


@pytest.mark.parametrize("num_proc", [1, 2])
@pytest.mark.parametrize(
    "demo", demos, ids=[str(d.relative_to(demo_dir)) for d in demos])
def test_demo(demo, num_proc):
    cmd = [sys.executable, demo.name]
    if num_proc > 1:
        cmd = ["mpirun", "-np", str(num_proc), *cmd]

    # Non-interactive matplotlib backend so plt.show() does not block
    env = {**os.environ, "MPLBACKEND": "Agg"}
    subprocess.run(cmd, cwd=demo.parent, env=env, check=True)
