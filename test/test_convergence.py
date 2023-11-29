# Copyright (c) 2023 Nathan Sime
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

import dolfinx
import leopart.cpp as pyleopart
import ufl

tableaus = [
    pyleopart.tableaus.order2.generic_alpha(0.5),
    pyleopart.tableaus.order2.explicit_midpoint(),
    pyleopart.tableaus.order2.heun(),
    pyleopart.tableaus.order2.ralston(),
    pyleopart.tableaus.order3.generic_alpha(0.5),
    pyleopart.tableaus.order3.heun(),
    pyleopart.tableaus.order3.wray(),
    pyleopart.tableaus.order3.ralston(),
    pyleopart.tableaus.order3.ssp(),
    pyleopart.tableaus.order4.classic(),
    pyleopart.tableaus.order4.kutta1901(),
    pyleopart.tableaus.order4.ralston(),
]


def create_mesh(cell_type, dtype, n):
    mesh_fn = {
        2: dolfinx.mesh.create_rectangle,
        3: dolfinx.mesh.create_box
    }

    cell_dim = dolfinx.mesh.cell_dim(cell_type)
    return mesh_fn[cell_dim](MPI.COMM_WORLD, [[-1.0, -1.0], [1.0, 1.0]],
                             [n] * cell_dim, cell_type=cell_type, dtype=dtype)


@pytest.mark.parametrize("tableau", tableaus)
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("cell_type", [dolfinx.mesh.CellType.triangle])
def test_l2_project_convergence(tableau, dtype, cell_type):
    # Advect particles through velocity field until t = t_max / 2 at
    # which time the velocity field is reversed until t = t_max, returning
    # particles to their original positions.
    seed = 1

    def u_f(x):
        return np.stack((x[1] * (1 - x[0] ** 2), -x[0] * (1 - x[1] ** 2)))

    def phi0_f(x):
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])

    t_max = 2.0 * np.pi
    n_steps_vals = np.array([20, 40], dtype=int)
    l2_errors = np.zeros_like(n_steps_vals, dtype=np.double)

    mesh = create_mesh(cell_type, dtype, 8)
    dt_vals = t_max / n_steps_vals
    for run_num, (dt, n_steps) in enumerate(zip(dt_vals, n_steps_vals)):
        xp, p2cell = pyleopart.mesh_fill(mesh._cpp_object, 15, seed)
        xp = np.c_[xp, np.zeros_like(xp[:, 0])]
        ptcls = pyleopart.Particles(xp, p2cell)
        tableau.check_and_create_fields(ptcls)

        V = dolfinx.fem.FunctionSpace(
            mesh, ("CG", max(tableau.order - 1, 1), (mesh.geometry.dim,)))
        u = dolfinx.fem.Function(V)

        u.interpolate(u_f)
        u.x.scatter_forward()

        DG0 = dolfinx.fem.FunctionSpace(mesh, ("DG", tableau.order - 1))
        phi0 = dolfinx.fem.Function(DG0)

        phi0.interpolate(phi0_f)

        ptcls.add_field("phi", [1])
        pyleopart.transfer_to_particles(
            ptcls, ptcls.field("phi"), phi0._cpp_object)

        t = 0.0
        for j in range(n_steps):
            t += dt
            if j == (n_steps // 2):
                u.x.array[:] *= -1
                u.x.scatter_forward()

            pyleopart.rk(mesh._cpp_object, ptcls, tableau,
                         lambda t: u._cpp_object, t, dt)

        phi1 = dolfinx.fem.Function(DG0)
        pyleopart.transfer_to_function(phi1._cpp_object, ptcls,
                                       ptcls.field("phi"))

        l2_error = mesh.comm.allreduce(
            dolfinx.fem.assemble_scalar(
                dolfinx.fem.form((phi0 - phi1) ** 2 * ufl.dx)), op=MPI.SUM)
        l2_error = l2_error ** 0.5

        l2_errors[run_num] = l2_error

    rates = np.log(l2_errors[1:] / l2_errors[:-1]) \
            / np.log(dt_vals[1:] / dt_vals[:-1])

    assert np.all(rates > tableau.order - 0.1)
