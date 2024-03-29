# Copyright (c) 2023 Nathan Sime
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

import dolfinx
import leopart.cpp as pyleopart

tableaus = [
    pyleopart.tableaus.order1.forward_euler(),
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


@pytest.mark.parametrize("tableau", tableaus)
def test_advect_exact_space(tableau):
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, [[-1.0, -1.0], [1.0, 1.0]], [8, 8])

    V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1, (mesh.geometry.dim,)))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: np.stack((x[1], -x[0])))

    xp = np.array([
                   [0.1, 0.0, 0.0],
                   [0.0, 0.2, 0.0],
                   [0.1215, -0.35, 0.0],
                   [-0.4, 0.5, 0.0],
                   [np.pi / 11.0, -np.pi / 10.0, 0.0]
                   ], dtype=np.double)

    # All particles traverse single rotation in same time frame
    r = np.linalg.norm(xp, axis=1)
    d = 2 * np.pi * r
    speed = np.linalg.norm(xp, axis=1)
    t_max = d / speed
    assert np.all(np.isclose(t_max, t_max[0]))
    t_max = t_max[0]

    n_steps_vals = np.array([40, 60, 80], dtype=int)
    l2_errors = np.zeros((xp.shape[0], n_steps_vals.shape[0]), dtype=np.double)
    dt_vals = t_max / n_steps_vals
    for run_num, (dt, n_steps) in enumerate(zip(dt_vals, n_steps_vals)):
        xp_arr = xp if mesh.comm.rank == 0 else np.array([], dtype=np.double)
        ptcls = pyleopart.Particles(xp_arr, [0] * xp_arr.shape[0])
        ptcls.add_field("idx", [1])
        if mesh.comm.rank == 0:
            ptcls.field("idx").data().T[:] = np.arange(xp.shape[0],
                                                       dtype=np.double)
        ptcls.relocate_bbox(mesh._cpp_object, np.arange(xp_arr.shape[0]))
        tableau.check_and_create_fields(ptcls)

        t = 0.0
        for j in range(n_steps):
            t += dt
            pyleopart.rk(mesh._cpp_object, ptcls, tableau,
                         lambda t: u._cpp_object, t, dt)

        active = np.where(np.array(ptcls.particle_to_cell()) != -1)[0]
        idxs = np.array([int(idx) for idx in ptcls.field("idx").data()])[active]
        l2_err = np.linalg.norm(ptcls.x().data()[active] - xp[idxs], axis=1)
        idxs = np.concatenate(mesh.comm.allgather(idxs))
        l2_err = np.concatenate(mesh.comm.allgather(l2_err))
        l2_errors[idxs, run_num] = l2_err

    rates = np.log(l2_errors[:, 1:] / l2_errors[:, :-1]) \
            / np.log(dt_vals[1:] / dt_vals[:-1])
    TOL = 1e-1
    assert np.all(rates > tableau.order - TOL)


@pytest.mark.parametrize("gdim", [2, 3])
@pytest.mark.parametrize("tableau", tableaus)
def test_tableau_check_and_create_fields(gdim, tableau):
    if gdim == 2:
        mesh = dolfinx.mesh.create_unit_square(
            MPI.COMM_WORLD, 8, 8)
    else:
        mesh = dolfinx.mesh.create_unit_cube(
            MPI.COMM_WORLD, 4, 4, 4)

    xp, p2c = pyleopart.mesh_fill(mesh._cpp_object, 1, seed=1)
    if gdim == 2:
        xp = np.c_[xp, np.zeros_like(xp[:, 0])]
    ptcls = pyleopart.Particles(xp, p2c)

    tableau.check_and_create_fields(ptcls)

    assert ptcls.field_exists(tableau.field_name_xn())
    xdim = ptcls.x().value_shape
    assert ptcls.field(tableau.field_name_xn()).value_shape == xdim
    for s in range(tableau.order):
        assert ptcls.field_exists(tableau.field_name_substep(s))
        assert ptcls.field(tableau.field_name_substep(s)).value_shape == xdim
