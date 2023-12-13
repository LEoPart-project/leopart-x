# Copyright (c) 2023 Nathan Sime
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI
from petsc4py import PETSc

import adios2
import numpy as np
import enum

import dolfinx
import dolfinx.fem.petsc
import leopart.cpp as pyleopart
import leopart.io
import ufl


# This demo reproduces the Rayeigh-Taylor instability benchmark exhibited in,
# for example, van Keken et al. (1997)
# https://agupubs.onlinelibrary.wiley.com/doi/10.1029/97JB01353
#
# Here we show the importance of a divergence free velocity field approximation.
# Provided is are implementations of the Stokes system discretised by the
# Taylor-Hood (TH) element and the C^0-interior penalty Galerkin (IPG) method.
# Note that the C^0-IPG scheme does and the TH method does *not* provide an
# exactly pointwise divergence free velocity approximation, respectively.


def pprint(*msg, rank=None):
    if rank is not None and MPI.COMM_WORLD.rank != rank:
        return
    print(f"[{MPI.COMM_WORLD.rank}]: {' '.join(map(str, msg))}", flush=True)


# Parameters
lmbda, H = 0.9142, 1.0  # Domain dimensions
p = 2                   # velocity field polynomial order
A = 0.02                # Initial perturbation magnitude
db = 0.2                # Initial height of light layer
tableau = pyleopart.tableaus.order3.generic_alpha(0.5)
t_max = 86.0
num_t_steps_max = 1500

# Geometry
mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD, [[0.0, 0.0], [lmbda, H]], [40, 40],
    cell_type=dolfinx.mesh.CellType.triangle,
    diagonal=dolfinx.mesh.DiagonalType.left_right)

# Chemistry space. A value of 1 corresponds to dense material, and 0 light
# material
chem_space = dolfinx.fem.FunctionSpace(mesh, ("DG", 0))
C = dolfinx.fem.Function(chem_space)
C.interpolate(lambda x: np.where(x[1] > db, 1.0, 0.0))
C.x.scatter_forward()

# Perturb the mesh and initial condition
meshx = mesh.geometry.x
S = db * (1.0 - db)
meshx[:, 1] += meshx[:, 1] * (H - meshx[:, 1]) / S \
              * A * np.cos(np.pi * lmbda * meshx[:, 0])

# Generate particles and interpolate the composition
xp, p2cell = pyleopart.mesh_fill(mesh._cpp_object, 25)
xp = np.c_[xp, np.zeros_like(xp[:, 0])]
pprint(f"num paticles: {xp.shape[0]}")
ptcls = pyleopart.Particles(xp, p2cell)
tableau.check_and_create_fields(ptcls)
chem_label = "C"
ptcls.add_field(chem_label, [1])
pyleopart.transfer_to_particles(ptcls, ptcls.field(chem_label), C._cpp_object)

# Viscosity model and buoyancy terms
eta_dense = dolfinx.fem.Constant(mesh, 1.0)
eta_light = dolfinx.fem.Constant(mesh, 0.1)
mu = eta_light + C * (eta_dense - eta_light)
Rb = dolfinx.fem.Constant(mesh, 1.0)
f = Rb * C * ufl.as_vector((0, -1))

class Scheme(enum.Enum):
    c0sipg = enum.auto()
    taylor_hood = enum.auto()
scheme = Scheme.taylor_hood

if scheme is Scheme.taylor_hood:
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

    problem = dolfinx.fem.petsc.LinearProblem(
        a, L, u=Uh, bcs=[zero_x_bc, zero_y_bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"})

    # Velocity as function of time to be used in Runge-Kutta integration
    uh = Uh.sub(0)
    def velocity(t):
        pyleopart.transfer_to_function(
            C._cpp_object, ptcls, ptcls.field(chem_label))
        C.x.scatter_forward()
        problem.solve()
        return uh._cpp_object

elif scheme is Scheme.c0sipg:
    if mesh.ufl_cell() != ufl.triangle:
        err_msg = (
            "Non-affine cells require careful interpolation to appropriate "
            "velocity spaces")
        raise NotImplementedError(err_msg)
    # Stream function space
    PSI = dolfinx.fem.FunctionSpace(mesh, ("CG", p + 1))
    psi = ufl.TestFunction(PSI)
    phi = ufl.TrialFunction(PSI)
    n = ufl.FacetNormal(mesh)
    penalty_constant = dolfinx.fem.Constant(mesh, 20.0)

    # Homogeneous BCs imposed on the stream function
    facets = dolfinx.mesh.locate_entities_boundary(
        mesh, dim=mesh.topology.dim - 1,
        marker=lambda x: np.ones_like(x[0], dtype=np.int8))
    dofs = dolfinx.fem.locate_dofs_topological(
        PSI, mesh.topology.dim - 1, facets)
    zero_bc = dolfinx.fem.dirichletbc(0.0, dofs, PSI)

    # Weak imposition of zero flow on the top and bottom
    facets_top_bot = dolfinx.mesh.locate_entities_boundary(
        mesh, dim=mesh.topology.dim - 1,
        marker=lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], H))
    mt_top_bot = dolfinx.mesh.meshtags(
        mesh, mesh.topology.dim - 1, facets_top_bot,
        np.full_like(facets_top_bot, 1, dtype=np.int32))
    ds_0 = ufl.Measure("ds", subdomain_data=mt_top_bot)(1)
    zero_u = dolfinx.fem.Constant(mesh, (0.0, 0.0))

    # Strainrate
    def eps(u):
        return ufl.sym(ufl.grad(u))

    # Stress tensor
    def sigma(u):
        return 2 * mu * ufl.sym(ufl.grad(u))

    # Rank 4 and rank 2 tensor multiplication
    def G_mult(G, tau):
        m, d = tau.ufl_shape
        return ufl.as_matrix([[ufl.inner(G[i, k, :, :], tau) for k in range(d)]
                              for i in range(m)])

    # Homogeneity of the stress tesnor
    G = mu * ufl.as_tensor([[
        [[2, 0],
         [0, 0]],
        [[0, 1],
         [1, 0]]],
        [[[0, 1],
          [1, 0]],
         [[0, 0],
          [0, 2]]]])

    def tensor_jump(u, n):
        return ufl.outer(u, n)("+") + ufl.outer(u, n)("-")

    # Formulate standard SIPG problem
    h = ufl.CellVolume(mesh) / ufl.FacetArea(mesh)
    degree_penalty = dolfinx.fem.Constant(mesh, float((p + 1) ** 2))
    beta = penalty_constant * degree_penalty / h

    def Bh(u, v):
        domain = ufl.inner(sigma(u), eps(v)) * ufl.dx
        interior = (
           - ufl.inner(tensor_jump(u, n), ufl.avg(sigma(v)))
           - ufl.inner(tensor_jump(v, n), ufl.avg(sigma(u)))
           + ufl.inner(beta("+") * G_mult(ufl.avg(G), tensor_jump(u, n)),
                       tensor_jump(v, n))) * ufl.dS
        exterior = (
           - ufl.inner(ufl.outer(u, n), sigma(v))
           - ufl.inner(ufl.outer(v, n), sigma(u))
           + ufl.inner(beta * G_mult(G, ufl.outer(u, n)), ufl.outer(v, n))
                   ) * ds_0
        return domain + interior + exterior

    def lh(v):
        domain = ufl.inner(f, v) * ufl.dx
        exterior = (
           - ufl.inner(ufl.outer(zero_u, n), sigma(v))
           + ufl.inner(beta * G_mult(G, ufl.outer(zero_u, n)), ufl.outer(v, n))
                   ) * ds_0
        return domain + exterior

    # Linear problem and solver
    phi_h = dolfinx.fem.Function(PSI)
    problem = dolfinx.fem.petsc.LinearProblem(
        Bh(ufl.curl(phi), ufl.curl(psi)), lh(ufl.curl(psi)),
        u=phi_h, bcs=[zero_bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"})

    # Velocity function space into which we interpolate curl(phi)
    uh_spc = dolfinx.fem.FunctionSpace(mesh, ("DG", p, (2,)))
    uh_expr = dolfinx.fem.Expression(
        ufl.curl(phi_h), uh_spc.element.interpolation_points())
    uh = dolfinx.fem.Function(uh_spc)

    # Velocity as function of time to be used in Runge-Kutta integration
    def velocity(t):
        pyleopart.transfer_to_function(
            C._cpp_object, ptcls, ptcls.field(chem_label))
        C.x.scatter_forward()
        problem.solve()
        uh.interpolate(uh_expr)
        uh.x.scatter_forward()
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
chem_file = dolfinx.io.XDMFFile(mesh.comm, "C.xdmf", "w")

# Space for estimating speed for CFL criterion
Vspd = dolfinx.fem.FunctionSpace(mesh, ("DG", 0))
uspd = dolfinx.fem.Function(Vspd)
uspd_expr = dolfinx.fem.Expression(
    ufl.sqrt(ufl.inner(uh, uh)), Vspd.element.interpolation_points())

urms_form = dolfinx.fem.form(ufl.inner(uh, uh) * ufl.dx)
def compute_urms():
    return mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(urms_form), op=MPI.SUM) ** 0.5

# Main time loop
t = 0.0
ptcl_file.write_particles(ptcls, t, [chem_label])
chem_file.write_mesh(mesh)
chem_file.write_function(C, t=t)
velocity(0.0)  # Initial velocity estimate

# Record u_rms as a function of time
t_vals = [0.0]
urms_vals = []
urms_vals.append(compute_urms())
for j in range(num_t_steps_max):
    # Estimate dt based on CFL criterion
    uspd.interpolate(uspd_expr)
    uspd.vector.assemble()
    max_u_vec = uspd.vector.norm(PETSc.NormType.INF)
    dt = (c_cfl := 1.0) * hmin / max_u_vec
    pprint(f"Time step {j}, dt = {dt:.3e}, t = {t:.3e}", rank=0)

    # Enact explicit Runge-Kutta integration
    pyleopart.rk(mesh._cpp_object, ptcls, tableau,
                 velocity, t, dt)
    t += dt

    # Check for particle deficient cells and output data
    deficient_cells = pyleopart.find_deficient_cells(C._cpp_object, ptcls)
    if len(deficient_cells) > 0:
        pprint(f"Particle deficient cells found: {deficient_cells}",
               rank=mesh.comm.rank)
    pyleopart.transfer_to_function(
        C._cpp_object, ptcls, ptcls.field(chem_label))
    C.x.scatter_forward()
    ptcl_file.write_particles(ptcls, t, [chem_label])
    chem_file.write_function(C, t=t)
    urms_vals.append(compute_urms())
    t_vals.append(t)

    if t > t_max:
        break

if mesh.comm.rank == 0:
    # Plot u_rms data
    import matplotlib.pyplot as plt
    plt.plot(t_vals, urms_vals)
    plt.show()
