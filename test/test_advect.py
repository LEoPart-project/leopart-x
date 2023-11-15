import pytest
import numpy as np
import leopart.cpp as pyleopart
from mpi4py import MPI
import dolfinx


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
        MPI.COMM_WORLD, [[-1.0, -1.0], [1.0, 1.0]], [1, 1])

    x0 = np.array([0.5, 0.0, 0.0], dtype=np.double)
    xp = np.array([x0], dtype=np.double)
    ptcls = pyleopart.Particles(xp, [0])

    V = dolfinx.fem.FunctionSpace(mesh, ("CG", 2, (mesh.geometry.dim,)))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: np.stack((x[1], -x[0])))
    ptcls.add_field("u", [mesh.geometry.dim])

    r = np.sqrt(x0[0] ** 2 + x0[1] ** 2)
    d = 2 * np.pi * r
    speed = np.sqrt(x0[0] ** 2 + x0[1] ** 2)
    t_max = d / speed

    ptcls.add_field("xn", [3])
    for i in range(tableau.order):
        ptcls.add_field(f"k{i}", [3])

    n_steps_vals = np.array([10, 20, 40, 80], dtype=int)
    l2_errors = np.zeros_like(n_steps_vals, dtype=np.double)
    dt_vals = t_max / n_steps_vals
    for run_num, (dt, n_steps) in enumerate(zip(dt_vals, n_steps_vals)):
        ptcls.field("x").data()[:] = x0
        ptcls.relocate_bbox(mesh._cpp_object, [0])
        t = 0.0
        for j in range(n_steps):
            t += dt
            pyleopart.rk(mesh._cpp_object, ptcls, tableau,
                         lambda t: u._cpp_object, t, dt)
        l2_errors[run_num] = np.linalg.norm(ptcls.field("x").data() - x0)

    rates = np.log(l2_errors[1:] / l2_errors[:-1]) \
            / np.log(dt_vals[1:] / dt_vals[:-1])
    TOL = 1e-1
    assert np.all(rates > tableau.order - TOL)
