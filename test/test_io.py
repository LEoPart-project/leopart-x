# Copyright (c) 2023 Nathan Sime
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pathlib

from mpi4py import MPI

import adios2
import numpy as np

import dolfinx
import leopart.cpp as pyleopart
import leopart.io


def create(datadir):
    datadir.mkdir()

def test_xdmf_output(tempdir):
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, [[-1.0, -1.0], [1.0, 1.0]], [4, 4],
        cell_type=dolfinx.mesh.CellType.triangle)
    x, p2c = pyleopart.mesh_fill(mesh._cpp_object, 5, seed=1)
    x = np.c_[x, np.zeros_like(x[:, 0])]
    ptcls = pyleopart.Particles(x, p2c)

    filename = pathlib.Path(tempdir) / "example.xdmf"
    fi = leopart.io.XDMFParticlesFile(
        MPI.COMM_WORLD, filename, adios2.Mode.Write)

    shapes = (1, 2, 3)
    field_names = [f"field{shape}" for shape in shapes]
    for shape, field_name in zip(shapes, field_names):
        ptcls.add_field(field_name, [shape])

    for t in [0.0, 1.0]:
        fi.write_particles(ptcls, t, field_names)
