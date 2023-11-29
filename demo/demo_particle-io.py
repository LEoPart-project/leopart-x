# Copyright: (c) 2023 Nathan Sime
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import adios2
import numpy as np

import dolfinx
import leopart.cpp as pyleopart
import leopart.io

mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD, [[-1.0, -1.0], [1.0, 1.0]], [32, 32],
    cell_type=dolfinx.mesh.CellType.triangle)
x, p2c = pyleopart.mesh_fill(mesh._cpp_object, 25)
x = np.c_[x, np.zeros_like(x[:,0])]
ptcls = pyleopart.Particles(x, p2c)

xp = ptcls.x().data()

ptcls.add_field("rank", [1])
ptcls.field("rank").data().T[:] = mesh.comm.rank

ptcls.add_field("scalar", [1])
ptcls.field("scalar").data().T[:] = \
    np.sin(np.pi*xp[:,0]) * np.sin(np.pi*xp[:,1])

ptcls.add_field("vector", [2])
ptcls.field("vector").data()[:] = np.stack((
    np.sin(xp[:,0]), np.sin(xp[:,1]))).T

fi = leopart.io.XDMFParticlesFile(
    MPI.COMM_WORLD, "example.xdmf", adios2.Mode.Write)


def u_f(x):
    return np.stack((x[1] * (1 - x[0] ** 2), -x[0] * (1 - x[1] ** 2)))


V = dolfinx.fem.FunctionSpace(mesh, ("CG", 2, (2,)))
u = dolfinx.fem.Function(V)
u.interpolate(u_f)

ptcls.add_field("u", [2])

fields_to_write = ["rank", "scalar", "vector"]
dt = 0.25
t = 0.0
tableau = pyleopart.tableaus.order2.explicit_midpoint()
tableau.check_and_create_fields(ptcls)
for step in range(25):
    t += dt
    pyleopart.rk(mesh._cpp_object, ptcls, tableau,
                 lambda t: u._cpp_object, t, dt)
    fi.write_particles(ptcls, t, fields_to_write)
