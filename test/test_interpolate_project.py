import dolfinx
from mpi4py import MPI
import numpy as np
import pyleopart
import pytest

from dolfinx.fem.assemble import assemble_scalar
from ufl import dx, dot

# Test back-and-forth projection on 2D mesh: scalar case
@pytest.mark.parametrize("k", [2, 3])
def test_interpolate_project_dg_scalar(k):
    npart = 20
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 5, 5)
    ncells = mesh.topology.index_map(2).size_local
    x, c = pyleopart.mesh_fill(mesh, ncells * npart)
    p = pyleopart.Particles(x, c)

    Q = dolfinx.function.FunctionSpace(mesh, ("DG", k))

    pbasis = pyleopart.get_particle_contributions(p, Q._cpp_object)

    def sq_val(x):
        return x[0] ** k

    u = dolfinx.function.Function(Q)
    u.interpolate(sq_val)

    p.add_field("v", [1])
    v = p.field("v")

    # Transfer from Function "u" to field "v"
    pyleopart.transfer_to_particles(p, v, u._cpp_object, pbasis)

    # Transfer from field "v" back to Function "u"
    pyleopart.transfer_to_function(u._cpp_object, p, v, pbasis)

    p.add_field("w", [1])
    w = p.field("w")
    # Transfer from Function "u" to field "w"
    pyleopart.transfer_to_particles(p, w, u._cpp_object, pbasis)

    # Compare fields "w" and "x"(squared)
    for pidx in range(len(x)):
        expected = p.field("x").data(pidx)[0] ** k
        assert np.isclose(p.field("w").data(pidx), expected).all()


# Test back-and-forth projection on 2D mesh: vector valued case
@pytest.mark.parametrize("k", [2, 3])
def test_interpolate_project_dg_vector(k):
    npart = 20
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 5, 5)
    ncells = mesh.topology.index_map(2).size_local
    x, c = pyleopart.mesh_fill(mesh, ncells * npart)
    p = pyleopart.Particles(x, c)

    Q = dolfinx.function.VectorFunctionSpace(mesh, ("DG", k))
    pbasis = pyleopart.get_particle_contributions(p, Q._cpp_object)

    def sq_val(x):
        return [x[0] ** k, x[1] ** k]

    u = dolfinx.function.Function(Q)
    u.interpolate(sq_val)

    p.add_field("v", [2])
    v = p.field("v")

    # Transfer from Function "u" to field "v"
    pyleopart.transfer_to_particles(p, v, u._cpp_object, pbasis)

    # Transfer from field "v" back to Function "u"
    pyleopart.transfer_to_function(u._cpp_object, p, v, pbasis)

    p.add_field("w", [2])
    w = p.field("w")
    # Transfer from Function "u" to field "w"
    pyleopart.transfer_to_particles(p, w, u._cpp_object, pbasis)

    # Compare fields "w" and "x"(squared)
    for pidx in range(len(x)):
        expected = p.field("x").data(pidx) ** k
        assert np.isclose(p.field("w").data(pidx), expected).all()

# Test back-and-forth projection on 2D mesh: vector valued case
@pytest.mark.parametrize("k", [2, 3])
def test_l2_project(k):
    npart = 20
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 5, 5)
    ncells = mesh.topology.index_map(2).size_local
    x, c = pyleopart.mesh_fill(mesh, ncells * npart)
    p = pyleopart.Particles(x, c)
    p.add_field("v", [2])
    v = p.field("v")

    Q = dolfinx.function.VectorFunctionSpace(mesh, ("DG", k))
    pbasis = pyleopart.get_particle_contributions(p, Q._cpp_object)

    def sq_val(x):
       return [x[0] ** k, x[1] ** k]

    u = dolfinx.function.Function(Q)
    u.interpolate(sq_val)

    # Transfer from Function "u" to (particle) field "v"
    pyleopart.transfer_to_particles(p, v, u._cpp_object, pbasis)

    #Init and conduct l2projection
    v = dolfinx.function.Function(Q)
    l2project = pyleopart.L2Project(p, v._cpp_object, "v")
    l2project.solve()

    l2_error = mesh.mpi_comm().allreduce(assemble_scalar(dot(u - v, u - v) * dx), op=MPI.SUM)
    assert l2_error < 1e-15