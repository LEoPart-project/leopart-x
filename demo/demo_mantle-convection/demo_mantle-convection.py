# Copyright (c) 2023 Nathan Sime
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI
from petsc4py import PETSc

import adios2
import numpy as np

import dolfinx
import dolfinx.fem.petsc
import leopart.cpp as pyleopart
import leopart.io
import ufl


def pprint(*msg, rank=None):
    if rank is not None and MPI.COMM_WORLD.rank != rank:
        return
    print(f"[{MPI.COMM_WORLD.rank}]: {' '.join(map(str, msg))}", flush=True)


lmbda, H = 0.9142, 1.0
mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD, [[0.0, 0.0], [lmbda, H]], [40, 40],
    cell_type=dolfinx.mesh.CellType.triangle,
    diagonal=dolfinx.mesh.DiagonalType.left_right)

# Parameters
p = 2
A = 0.02
db = 0.2
S = db * (1 - db)
tableau = pyleopart.tableaus.order2.explicit_midpoint()
t_max = 2000.0
num_t_steps_max = 200

# Chemistry space
PHI = dolfinx.fem.FunctionSpace(mesh, ("DG", 0))
phi = dolfinx.fem.Function(PHI)
phi.interpolate(lambda x: np.where(x[1] > db, 1.0, 0.0))

# Perturb the mesh and initial condition
meshx = mesh.geometry.x
meshx[:, 1] += meshx[:, 1] * (H - meshx[:, 1]) / S \
              * A * np.cos(np.pi * lmbda * meshx[:, 0])

# Generate particles and interpolate the composition
xp, p2cell = pyleopart.mesh_fill(mesh._cpp_object, 30)
xp = np.c_[xp, np.zeros_like(xp[:, 0])]
pprint(f"num paticles: {xp.shape[0]}")
ptcls = pyleopart.Particles(xp, p2cell)
tableau.check_and_create_fields(ptcls)
ptcls.add_field("phi", [1])
pyleopart.transfer_to_particles(ptcls, ptcls.field("phi"), phi._cpp_object)

# Viscosity model
eta_top = dolfinx.fem.Constant(mesh, 1.0)
eta_bottom = dolfinx.fem.Constant(mesh, 0.1)
mu = eta_bottom + phi * (eta_top - eta_bottom)

# Standard Taylor Hood mixed element
Ve = ufl.VectorElement("CG", mesh.ufl_cell(), p)
Qe = ufl.FiniteElement("CG", mesh.ufl_cell(), p - 1)
We = ufl.MixedElement([Ve, Qe])
W = dolfinx.fem.FunctionSpace(mesh, We)
u, p_ = ufl.TrialFunctions(W)
v, q = ufl.TestFunctions(W)
Uh = dolfinx.fem.Function(W)

# Bilinear formulation
a = (
        ufl.inner(2 * mu * ufl.sym(ufl.grad(u)), ufl.grad(v)) * ufl.dx
        - p_ * ufl.div(v) * ufl.dx
        - q * ufl.div(u) * ufl.dx
)

Rb = dolfinx.fem.Constant(mesh, 1.0)
f = Rb * phi * ufl.as_vector((0, -1))
L = ufl.inner(f, v) * ufl.dx

# Create BCs: free slip on left and right, zero flow top and bottom
facets_top_bot = dolfinx.mesh.locate_entities_boundary(
    mesh, dim=mesh.topology.dim - 1,
    marker=lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], H))
facets_left_right = dolfinx.mesh.locate_entities_boundary(
    mesh, dim=mesh.topology.dim - 1,
    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], lmbda))

V_x = W.sub(0).sub(0).collapse()
zero = dolfinx.fem.Function(V_x[0])
dofs_lr = dolfinx.fem.locate_dofs_topological(
    (W.sub(0).sub(0), V_x[0]), mesh.topology.dim - 1, facets_left_right)
zero_x_bc = dolfinx.fem.dirichletbc(zero, dofs_lr, W.sub(0).sub(0))

W0 = W.sub(0).collapse()
zero = dolfinx.fem.Function(W0[0])
dofs_tb = dolfinx.fem.locate_dofs_topological(
    (W.sub(0), W0[0]), mesh.topology.dim - 1, facets_top_bot)
zero_y_bc = dolfinx.fem.dirichletbc(zero, dofs_tb, W.sub(0))

# Pin pressure DoF in bottom left corner for solvable system
Q = W.sub(1).collapse()
zero_p = dolfinx.fem.Function(Q[0])
dofs_p = dolfinx.fem.locate_dofs_geometrical(
    (W.sub(1), Q[0]),
    lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
zero_p_bc = dolfinx.fem.dirichletbc(zero_p, dofs_p, W.sub(1))

problem = dolfinx.fem.petsc.LinearProblem(
    a, L, u=Uh, bcs=[zero_x_bc, zero_y_bc, zero_p_bc],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                   "pc_factor_mat_solver_type": "mumps"})

# Velocity as function of time to be used in Runge-Kutta integration
uh = Uh.sub(0)
def velocity(t):
    pyleopart.transfer_to_function(phi._cpp_object, ptcls, ptcls.field("phi"))
    problem.solve()
    return uh._cpp_object

# h measured used in CFL criterion estimation
h_measure = dolfinx.cpp.mesh.h(
    mesh._cpp_object, mesh.topology.dim - 1, np.arange(
        mesh.topology.index_map(mesh.topology.dim - 1).size_local,
        dtype=np.int32))
hmin = mesh.comm.allreduce(h_measure.min(), op=MPI.MIN)

# Output files
ptcl_file = leopart.io.XDMFParticlesFile(
    mesh.comm, "particles.xdmf", adios2.Mode.Write)
phi_file = dolfinx.io.XDMFFile(mesh.comm, "phi.xdmf", "w")

# Space for estimating speed for CFL criterion
Vspd = dolfinx.fem.FunctionSpace(mesh, ("DG", 0))
uspd = dolfinx.fem.Function(Vspd)
uspd_expr = dolfinx.fem.Expression(
    ufl.sqrt(ufl.inner(uh, uh)), Vspd.element.interpolation_points())

# Main time loop
t = 0.0
ptcl_file.write_particles(ptcls, t, ["phi"])
phi_file.write_mesh(mesh)
phi_file.write_function(phi, t=t)
problem.solve()  # Get initial speed estimate
for j in range(num_t_steps_max):
    uspd.interpolate(uspd_expr)
    uspd.vector.assemble()
    max_u_vec = uspd.vector.norm(PETSc.NormType.INF)
    dt = (c_cfl := 1.0) * hmin / max_u_vec
    t += dt
    pprint(f"Time step {j}, dt = {dt:.3e}, t = {t:.3e}", rank=0)

    pyleopart.rk(mesh._cpp_object, ptcls, tableau,
                 velocity, t, dt)

    pyleopart.transfer_to_function(phi._cpp_object, ptcls, ptcls.field("phi"))
    ptcl_file.write_particles(ptcls, t, ["phi"])
    phi_file.write_function(phi, t=t)

    if t > t_max:
        break
