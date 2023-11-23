# Copyright: (c) 2023 Nathan Sime
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
import leopart.cpp as pyleopart
from mpi4py import MPI
import dolfinx
import ufl


def pprint(*msg, rank=None):
    if rank is not None and MPI.COMM_WORLD.rank != rank:
        return
    print(f"[{MPI.COMM_WORLD.rank}]: {' '.join(map(str, msg))}", flush=True)


mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD, [[-1.0, -1.0], [1.0, 1.0]], [32, 32],
    cell_type=dolfinx.mesh.CellType.triangle)


def u_f(x):
    return np.stack((x[1] * (1 - x[0] ** 2), -x[0] * (1 - x[1] ** 2)))


def phi0_f(x):
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


tableau = pyleopart.tableaus.order2.explicit_midpoint()
t_max = 2.0 * np.pi
n_steps_vals = np.array([20, 40], dtype=int)
l2_errors = np.zeros_like(n_steps_vals, dtype=np.double)

dt_vals = t_max / n_steps_vals
for run_num, (dt, n_steps) in enumerate(zip(dt_vals, n_steps_vals)):
    import time
    then = time.time()
    xp, p2cell = pyleopart.mesh_fill(mesh._cpp_object, 15)
    xp = np.c_[xp, np.zeros_like(xp[:,0])]
    pprint(f"num paticles: {xp.shape[0]}")
    ptcls = pyleopart.Particles(xp, p2cell)

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

    ptcls.add_field("xn", [3])
    for i in range(tableau.order):
        ptcls.add_field(f"k{i}", [3])

    t = 0.0
    for j in range(n_steps):
        t += dt
        pprint(f"Time step {j}, t = {t:.3e}", rank=0)
        if j == (n_steps // 2):
            pprint(f"Swap velocity direction at t = {t:.3e}", rank=0)
            u.x.array[:] *= -1
            u.x.scatter_forward()

        pyleopart.rk(mesh._cpp_object, ptcls, tableau,
                     lambda t: u._cpp_object, t, dt)

    phi1 = dolfinx.fem.Function(DG0)
    pyleopart.transfer_to_function(phi1._cpp_object, ptcls, ptcls.field("phi"))

    l2_error = mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(
            dolfinx.fem.form((phi0 - phi1)**2 * ufl.dx)), op=MPI.SUM)
    l2_error = l2_error ** 0.5

    l2_errors[run_num] = l2_error

rates = np.log(l2_errors[1:] / l2_errors[:-1]) \
        / np.log(dt_vals[1:] / dt_vals[:-1])

pprint("L2 errors: ", l2_errors, rank=0)
pprint("L2 convergence rates: ", rates, rank=0)
