import dolfinx
from mpi4py import MPI
import numpy as np
import leopart.cpp as pyleopart
import pytest
import ufl


def create_mesh(cell_type, dtype, n):
    mesh_fn = {
        2: dolfinx.mesh.create_unit_square,
        3: dolfinx.mesh.create_unit_cube
    }

    cell_dim = dolfinx.mesh.cell_dim(cell_type)
    return mesh_fn[cell_dim](MPI.COMM_WORLD, *[n]*cell_dim,
                             cell_type=cell_type, dtype=dtype)


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("cell_type", [dolfinx.mesh.CellType.triangle,
                                       dolfinx.mesh.CellType.tetrahedron,
                                       dolfinx.mesh.CellType.quadrilateral,
                                       dolfinx.mesh.CellType.hexahedron])
@pytest.mark.parametrize("shape", [(1,), (2,)])
def test_transfer_to_particles(k, dtype, cell_type, shape):
    mesh = create_mesh(cell_type, dtype, 4)
    Q = dolfinx.fem.FunctionSpace(mesh, ("DG", k, shape))

    npart = Q.dofmap.dof_layout.num_entity_closure_dofs(mesh.topology.dim)
    x, c = pyleopart.mesh_fill(mesh._cpp_object, npart)
    if mesh.geometry.dim == 2:
        x = np.c_[x, np.zeros_like(x[:,0])]
    p = pyleopart.Particles(x, c)

    # Function is exactly represented in FE space
    def sq_val(x):
        return np.stack([x[i] ** k for i in range(shape[0])])

    u = dolfinx.fem.Function(Q)
    u.interpolate(sq_val)
    u.x.scatter_forward()

    p.add_field("v", shape)
    v = p.field("v")

    # Transfer from function to particles
    pyleopart.transfer_to_particles(p, v, u._cpp_object)

    expected = sq_val(p.field("x").data().T).T
    assert np.all(
        np.isclose(v.data(), expected,
                   rtol=np.finfo(dtype).eps*1e2, atol=np.finfo(dtype).eps))


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("cell_type", [dolfinx.mesh.CellType.triangle,
                                       dolfinx.mesh.CellType.tetrahedron,
                                       dolfinx.mesh.CellType.quadrilateral,
                                       dolfinx.mesh.CellType.hexahedron])
@pytest.mark.parametrize("shape", [(1,), (2,)])
def test_transfer_to_function(k, dtype, cell_type, shape):
    mesh = create_mesh(cell_type, dtype, 4)
    Q = dolfinx.fem.FunctionSpace(mesh, ("DG", k, shape))

    npart = Q.dofmap.dof_layout.num_entity_closure_dofs(mesh.topology.dim)
    x, c = pyleopart.mesh_fill(mesh._cpp_object, npart)
    if mesh.geometry.dim == 2:
        x = np.c_[x, np.zeros_like(x[:,0])]
    p = pyleopart.Particles(x, c)

    # Function is exactly represented in FE space
    def sq_val(x):
        return np.stack([x[i] ** k for i in range(shape[0])])

    uh_exact = dolfinx.fem.Function(Q)
    uh_exact.interpolate(sq_val)
    uh_exact.x.scatter_forward()

    p.add_field("v", shape)
    p.field("v").data()[:] = sq_val(p.field("x").data().T).T

    # Transfer from particles to function
    u = dolfinx.fem.Function(Q)
    pyleopart.transfer_to_function(u._cpp_object, p, p.field("v"))

    l2error = mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(
            (u - uh_exact)**2 * ufl.dx)), op=MPI.SUM)
    assert l2error < 1e-12


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("cell_type", [dolfinx.mesh.CellType.triangle,
                                       dolfinx.mesh.CellType.tetrahedron,
                                       dolfinx.mesh.CellType.quadrilateral,
                                       dolfinx.mesh.CellType.hexahedron])
def test_constrained_transfer_to_function(k, dtype, cell_type):
    # Mesh mush align with discontinuity
    nele = 4
    x0 = 0.5
    assert (nele * x0) % 2.0 == 0.0

    mesh = create_mesh(cell_type, dtype, 4)
    Q = dolfinx.fem.FunctionSpace(mesh, ("DG", k))

    x, c = pyleopart.generate_at_dof_coords(Q._cpp_object)
    p = pyleopart.Particles(x, c)

    # Function is exactly represented in FE space
    def unconstrained_func(x):
        return x[0]

    # Constrained function is also exactly represented in FE space
    def constrained_func(x):
        return np.where(x[0] < x0, unconstrained_func(x), x0)

    # Interpolate the constrained function exactly
    uh_exact = dolfinx.fem.Function(Q)
    uh_exact.x.array[:] = x0
    cells = dolfinx.mesh.locate_entities(
        mesh, mesh.topology.dim, lambda x: x[0] < x0 + np.finfo(dtype).eps)
    uh_exact.interpolate(constrained_func, cells=cells)
    uh_exact.x.scatter_forward()

    # Set the particle data to the *un*constrained function
    p.add_field("v", [1])
    p.field("v").data()[:] = unconstrained_func(
        p.field("x").data().T).reshape((-1, 1))

    # Transfer the *un*constrained particle data to the FE function using
    # constrained solve
    u = dolfinx.fem.Function(Q)
    pyleopart.transfer_to_function_constrained(
        u._cpp_object, p, p.field("v"), 0.0, 0.5)

    l2error = mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(
            (u - uh_exact)**2 * ufl.dx)), op=MPI.SUM)
    assert l2error < np.finfo(dtype).eps
