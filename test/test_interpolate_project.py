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

    Q = dolfinx.FunctionSpace(mesh, ("DG", k))

    pbasis = pyleopart.get_particle_contributions(p, Q._cpp_object)

    def sq_val(x):
        return x[0] ** k

    u = dolfinx.Function(Q)
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
        print(f"Expected {expected}")
        print(f"Value at pidx {p.field('w').data(pidx)}")
        assert np.isclose(p.field("w").data(pidx), expected).all()


# # Test back-and-forth projection on 2D mesh: vector valued case
# @pytest.mark.parametrize("k", [2, 3])
# def test_interpolate_project_dg_vector(k):
#     npart = 20
#     mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 5, 5)
#     ncells = mesh.topology.index_map(2).size_local
#     x, c = pyleopart.mesh_fill(mesh, ncells * npart)
#     p = pyleopart.Particles(x, c)

#     Q = dolfinx.VectorFunctionSpace(mesh, ("DG", k))
#     pbasis = pyleopart.get_particle_contributions(p, Q._cpp_object)

#     def sq_val(x):
#         return [x[0] ** k, x[1] ** k]

#     u = dolfinx.Function(Q)
#     u.interpolate(sq_val)

#     p.add_field("v", [2])
#     v = p.field("v")

#     # Transfer from Function "u" to field "v"
#     pyleopart.transfer_to_particles(p, v, u._cpp_object, pbasis)

#     # Transfer from field "v" back to Function "u"
#     pyleopart.transfer_to_function(u._cpp_object, p, v, pbasis)

#     p.add_field("w", [2])
#     w = p.field("w")
#     # Transfer from Function "u" to field "w"
#     pyleopart.transfer_to_particles(p, w, u._cpp_object, pbasis)

#     # Compare fields "w" and "x"(squared)
#     for pidx in range(len(x)):
#         expected = p.field("x").data(pidx) ** k
#         assert np.isclose(p.field("w").data(pidx), expected).all()

# # Test back-and-forth projection on 2D mesh: vector valued case
# @pytest.mark.parametrize("k", [2, 3])
# def test_l2_project(k):
#     npart = 20
#     mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 5, 5)
#     ncells = mesh.topology.index_map(2).size_local
#     x, c = pyleopart.mesh_fill(mesh, ncells * npart)
#     p = pyleopart.Particles(x, c)
#     p.add_field("v", [2])
#     vp = p.field("v")

#     Q = dolfinx.VectorFunctionSpace(mesh, ("DG", k))
#     pbasis = pyleopart.get_particle_contributions(p, Q._cpp_object)

#     def sq_val(x):
#        return [x[0] ** k, x[1] ** k]

#     u = dolfinx.Function(Q)
#     u.interpolate(sq_val)

#     # Transfer from Function "u" to (particle) field "v"
#     pyleopart.transfer_to_particles(p, vp, u._cpp_object, pbasis)

#     #Init and conduct l2projection
#     v = dolfinx.Function(Q)
#     l2project = pyleopart.L2Project(p, v._cpp_object, "v")
#     l2project.solve()

#     l2_error = mesh.mpi_comm().allreduce(assemble_scalar(dot(u - v, u - v) * dx), op=MPI.SUM)
#     assert l2_error < 1e-15

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

#     Q = dolfinx.FunctionSpace(mesh, ("DG", k))
#     pbasis = pyleopart.get_particle_contributions(p, Q._cpp_object)
        
#     class SlottedDisk:
#         def __init__(self, radius, center, width, depth, lb=0.0, ub=1.0, **kwargs):
#             self.r = radius
#             self.width = width
#             self.depth = depth
#             self.center = center
#             self.lb = lb
#             self.ub = ub

#         def eval(self, x):
#             values = np.empty((x.shape[1], ))
#             xc = self.center[0]
#             yc = self.center[1]

#             # The mask is True within region of slotted disk and False elsewhere
#             mask = (((x[0] - xc) ** 2 + (x[1] - yc) ** 2 <= self.r ** 2)) & np.invert( 
#                  ((x[0] > (xc - self.width)) & (x[0] <= (xc + self.width)) & (x[1] >= yc + self.depth))
#             )
#             return mask * self.ub + np.invert(mask) * self.lb

#     # Add disturbance to lower and upper bound, to make sure the bounded projection return the 
#     # original bounds
#     slotted_disk = SlottedDisk(
#         radius=0.15, center=[0.5, 0.5], width=0.05, depth=0.0, degree=3, lb=lb-0.1, ub=ub+0.1)
#     u = dolfinx.Function(Q)
#     u.interpolate(slotted_disk.eval)

#     # Transfer from Function "u" to (particle) field "v"
#     pyleopart.transfer_to_particles(p, vp, u._cpp_object, pbasis)

#     #Init and conduct l2projection
#     vh = dolfinx.Function(Q)
#     l2project = pyleopart.L2Project(p, vh._cpp_object, "v")
#     l2project.solve(lb, ub)
    
#     # Assert if it stays within bounds
#     assert np.min(vh.x.array()) < ub + 1e-12
#     assert np.max(vh.x.array()) > lb - 1e-12