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
def test_transfer_to_particles(k, dtype, cell_type):
    mesh = create_mesh(cell_type, dtype, 4)
    Q = dolfinx.fem.FunctionSpace(mesh, ("DG", k))

    npart = Q.dofmap.dof_layout.num_entity_closure_dofs(mesh.topology.dim)
    x, c = pyleopart.mesh_fill(mesh._cpp_object, npart)
    if mesh.geometry.dim == 2:
        x = np.c_[x, np.zeros_like(x[:,0])]
    p = pyleopart.Particles(x, c)

    # Function is exactly represented in FE space
    def sq_val(x):
        return x[0] ** k

    u = dolfinx.fem.Function(Q)
    u.interpolate(sq_val)
    u.x.scatter_forward()

    p.add_field("v", [1])
    v = p.field("v")

    # Transfer from function to particles
    pyleopart.transfer_to_particles(p, v, u._cpp_object)

    expected = sq_val(p.field("x").data().T)
    assert np.all(
        np.isclose(p.field("v").data().ravel(), expected,
                   rtol=np.finfo(dtype).eps*1e2, atol=np.finfo(dtype).eps))


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("cell_type", [dolfinx.mesh.CellType.triangle,
                                       dolfinx.mesh.CellType.tetrahedron,
                                       dolfinx.mesh.CellType.quadrilateral,
                                       dolfinx.mesh.CellType.hexahedron])
def test_transfer_to_function(k, dtype, cell_type):
    mesh = create_mesh(cell_type, dtype, 4)
    Q = dolfinx.fem.FunctionSpace(mesh, ("DG", k))

    npart = Q.dofmap.dof_layout.num_entity_closure_dofs(mesh.topology.dim)
    x, c = pyleopart.mesh_fill(mesh._cpp_object, npart)
    if mesh.geometry.dim == 2:
        x = np.c_[x, np.zeros_like(x[:,0])]
    p = pyleopart.Particles(x, c)

    # Function is exactly represented in FE space
    def sq_val(x):
        return x[0] ** k

    uh_exact = dolfinx.fem.Function(Q)
    uh_exact.interpolate(sq_val)
    uh_exact.x.scatter_forward()

    p.add_field("v", [1])
    p.field("v").data()[:] = sq_val(p.field("x").data().T).reshape((-1, 1))

    # Transfer from particles to function
    u = dolfinx.fem.Function(Q)
    pyleopart.transfer_to_function(u._cpp_object, p, p.field("v"))

    l2error = mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(
            (u - uh_exact)**2 * ufl.dx)), op=MPI.SUM)
    assert l2error < np.finfo(dtype).eps


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

    npart = Q.dofmap.dof_layout.num_entity_closure_dofs(mesh.topology.dim)
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


# # Test back-and-forth projection on 2D mesh: vector valued case
# @pytest.mark.parametrize("k", [2, 3])
# def test_interpolate_project_dg_vector(k):
#     npart = 20
#     mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 5, 5)
#     ncells = mesh.topology.index_map(2).size_local
#     x, c = pyleopart.mesh_fill(mesh, ncells * npart)
#     p = pyleopart.Particles(x, c)
#
#     Q = dolfinx.function.VectorFunctionSpace(mesh, ("DG", k))
#     pbasis = pyleopart.get_particle_contributions(p, Q._cpp_object)
#
#     def sq_val(x):
#         return [x[0] ** k, x[1] ** k]
#
#     u = dolfinx.function.Function(Q)
#     u.interpolate(sq_val)
#
#     p.add_field("v", [2])
#     v = p.field("v")
#
#     # Transfer from Function "u" to field "v"
#     pyleopart.transfer_to_particles(p, v, u._cpp_object, pbasis)
#
#     # Transfer from field "v" back to Function "u"
#     pyleopart.transfer_to_function(u._cpp_object, p, v, pbasis)
#
#     p.add_field("w", [2])
#     w = p.field("w")
#     # Transfer from Function "u" to field "w"
#     pyleopart.transfer_to_particles(p, w, u._cpp_object, pbasis)
#
#     # Compare fields "w" and "x"(squared)
#     for pidx in range(len(x)):
#         expected = p.field("x").data(pidx) ** k
#         assert np.isclose(p.field("w").data(pidx), expected).all()

# @pytest.mark.parametrize("k", [2])
# def test_l2_project(k):
#     # Test back-and-forth projection on 2D mesh: vector valued case
#     npart = 10
#     mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 5, 5)
#     x, c = pyleopart.mesh_fill(mesh._cpp_object, npart)
#     x = np.c_[x, np.zeros_like(x[:,0])]
#     p = pyleopart.Particles(x, c)
#     p.add_field("v", [2])
#     vp = p.field("v")
#
#     Q = dolfinx.fem.VectorFunctionSpace(mesh, ("DG", k))
#
#     def sq_val(x):
#        return [x[0] ** k, x[1] ** k]
#
#     u = dolfinx.fem.Function(Q)
#     u.interpolate(sq_val)
#
#     # Transfer from Function "u" to (particle) field "v"
#     pyleopart.transfer_to_particles(p, vp, u._cpp_object)
#
#     #Init and conduct l2projection
#     v = dolfinx.fem.Function(Q)
#     pyleopart.transfer_to_function(v._cpp_object, p, vp)
#
#     l2_error = mesh.comm.allreduce(
#         dolfinx.fem.assemble_scalar(dolfinx.fem.form(dot(u - v, u - v) * dx)),
#         op=MPI.SUM)
#     assert l2_error < 1e-15
#
# # # @pytest.mark.parametrize("polynomial_order, lb, ub", [(1, -3.0, -1.0), (2, -3.0, -1.0)])
# @pytest.mark.parametrize("k, lb, ub", [(2, -3.0, -1.0), (3, -3.0, -1.0)])
# def test_l2_project_bounded(k, lb, ub):
#     npart = 20
#     mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 20, 20)
#     ncells = mesh.topology.index_map(2).size_local
#     x, c = pyleopart.mesh_fill(mesh, ncells * npart)
#     p = pyleopart.Particles(x, c)
#     p.add_field("v", [1])
#     vp = p.field("v")
#
#     Q = dolfinx.function.FunctionSpace(mesh, ("DG", k))
#     pbasis = pyleopart.get_particle_contributions(p, Q._cpp_object)
#
#     class SlottedDisk:
#         def __init__(self, radius, center, width, depth, lb=0.0, ub=1.0, **kwargs):
#             self.r = radius
#             self.width = width
#             self.depth = depth
#             self.center = center
#             self.lb = lb
#             self.ub = ub
#
#         def eval(self, x):
#             values = np.empty((x.shape[1], ))
#             xc = self.center[0]
#             yc = self.center[1]
#
#             # The mask is True within region of slotted disk and False elsewhere
#             mask = (((x[0] - xc) ** 2 + (x[1] - yc) ** 2 <= self.r ** 2)) & np.invert(
#                  ((x[0] > (xc - self.width)) & (x[0] <= (xc + self.width)) & (x[1] >= yc + self.depth))
#             )
#             return mask * self.ub + np.invert(mask) * self.lb
#
#     # Add disturbance to lower and upper bound, to make sure the bounded projection return the
#     # original bounds
#     slotted_disk = SlottedDisk(
#         radius=0.15, center=[0.5, 0.5], width=0.05, depth=0.0, degree=3, lb=lb-0.1, ub=ub+0.1)
#     u = dolfinx.function.Function(Q)
#     u.interpolate(slotted_disk.eval)
#
#     # Transfer from Function "u" to (particle) field "v"
#     pyleopart.transfer_to_particles(p, vp, u._cpp_object, pbasis)
#
#     #Init and conduct l2projection
#     vh = dolfinx.function.Function(Q)
#     l2project = pyleopart.L2Project(p, vh._cpp_object, "v")
#     l2project.solve(lb, ub)
#
#     # Assert if it stays within bounds
#     assert np.min(vh.x.array()) < ub + 1e-12
#     assert np.max(vh.x.array()) > lb - 1e-12