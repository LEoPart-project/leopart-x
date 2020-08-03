import dolfinx
from mpi4py import MPI
import numpy as np
import pyleopart
import pytest

# Test back-and-forth projection on 2D mesh
@pytest.mark.parametrize("is_vec, k", [(True, 2), (True, 3), (False, 2), (False, 3)])
def test_interpolate_project_dg(is_vec, k):
    npart = 20
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 5, 5)
    ncells = mesh.topology.index_map(2).size_local
    x, c = pyleopart.mesh_fill(mesh, ncells * npart)
    p = pyleopart.Particles(x, c)

    Q = (
        dolfinx.function.VectorFunctionSpace(mesh, ("DG", k))
        if is_vec
        else dolfinx.function.FunctionSpace(mesh, ("DG", k))
    )
    pbasis = pyleopart.get_particle_contributions(p, Q._cpp_object)

    if is_vec:

        def sq_val(x):
            return [x[0] ** k, x[1] ** k]

    else:

        def sq_val(x):
            return x[0] ** k

    u = dolfinx.function.Function(Q)
    u.interpolate(sq_val)

    p.add_field("v", [2]) if is_vec else p.add_field("v", [1])
    v = p.field("v")

    # Transfer from Function "u" to field "v"
    pyleopart.transfer_to_particles(p, v, u._cpp_object, pbasis)

    # Transfer from field "v" back to Function "u"
    pyleopart.transfer_to_function(u._cpp_object, p, v, pbasis)

    p.add_field("w", [2]) if is_vec else p.add_field("w", [1])
    w = p.field("w")
    # Transfer from Function "u" to field "w"
    pyleopart.transfer_to_particles(p, w, u._cpp_object, pbasis)

    # Compare fields "w" and "x"(squared)
    for pidx in range(len(x)):
        expected = (
            p.field("x").data(pidx) ** k if is_vec else p.field("x").data(pidx)[0] ** k
        )
        assert np.isclose(p.field("w").data(pidx), expected).all()
