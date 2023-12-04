# Copyright (c) 2023 Nathan Sime
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI
from petsc4py import PETSc

import imageio
import matplotlib.pyplot as plt
import numpy as np

import dolfinx
import dolfinx.nls.petsc
import leopart.cpp as pyleopart
import ufl

# Initial mesh matches the resolution of the input image
mesh = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, 150, 150, cell_type=dolfinx.mesh.CellType.quadrilateral)

# Load the data on rank 0
if mesh.comm.rank == 0:
    # "Leopard" by Mark Kent (flamesworddragon) is licensed under CC BY-SA 2.0.
    data = imageio.v2.imread("leopard.jpg")
    img_size = (data.shape[0], data.shape[1])
    L, H = data.shape[:2]
    x, y = np.meshgrid(np.linspace(0.0, 1.0, L), np.linspace(0.0, 1.0, H))
    xp = np.c_[x.ravel(), y.ravel(), np.zeros_like(x.ravel())]
else:
    img_size = None
    xp = np.zeros(0, dtype=np.double)
    data = np.zeros((0, 0, 3), dtype=np.double)

p2c = [0] * xp.shape[0]
ptcls = pyleopart.Particles(xp, p2c)
ptcls.add_field("data", [3])
ptcls.add_field("r", [1])
ptcls.add_field("c", [1])
if mesh.comm.rank == 0:
    ptcls.field("data").data()[:] = data.reshape((-1, 3))
    indices = np.indices(img_size)
    ptcls.field("r").data().T[:] = indices[0].ravel()
    ptcls.field("c").data().T[:] = indices[1].ravel()

# Distribute data across processes
ptcls.relocate_bbox(mesh._cpp_object, np.arange(len(p2c)))

# Compute grayscale of data and add artificial noise
gray_label = "gray"
noise_label = "noise"
ptcls.add_field(gray_label, [1])
ptcls.add_field(noise_label, [1])

ptcl_data = ptcls.field("data").data()
grayscale = np.dot(ptcl_data, np.array([0.299, 0.587, 0.114])) / 255.0
noise = np.random.default_rng(1).normal(0.0, 0.05, grayscale.shape)
ptcls.field(gray_label).data().T[:] = grayscale
ptcls.field(noise_label).data().T[:] = np.clip(grayscale + noise, 0.0, 1.0)

# DG0 continuum for image data
DG0 = dolfinx.fem.FunctionSpace(mesh, ("DG", 0))
u_img = dolfinx.fem.Function(DG0)

# FE space and problem for data diffusion
V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
uh = dolfinx.fem.Function(V)
um = dolfinx.fem.Function(V)
uth = 0.5 * (uh + um)
v = ufl.TestFunction(V)

# FE formulation of Perona Malik model
sigma_d0 = dolfinx.fem.Constant(mesh, 0.0)
pm_d = ufl.exp(
    - ufl.inner(ufl.grad(uth), ufl.grad(uth)) / (2 * sigma_d0**2))
pmd_expr = dolfinx.fem.Expression(pm_d, DG0.element.interpolation_points())
pmd_dg0 = dolfinx.fem.Function(DG0)

dt = dolfinx.fem.Constant(mesh, 0.0)
F = ufl.inner((uh - um) / dt, v) * ufl.dx
F += ufl.inner(pm_d * ufl.grad(uth), ufl.grad(v)) * ufl.dx

# Nonlinear solver and parameters
problem = dolfinx.fem.petsc.NonlinearProblem(F, uh)
solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
solver.max_it = 10
solver.rtol = 1e-5

ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

# Utility functions for storing snapshot data
def create_timestamp(t):
    return f"t={t:.2e}"


def record_snapshot(name):
    uname = "u_" + name
    ptcls.add_field(uname, [1])
    pyleopart.transfer_to_particles(ptcls, ptcls.field(uname), uh._cpp_object)

    pmd_name = "pmd_" + name
    pmd_dg0.interpolate(pmd_expr)
    ptcls.add_field(pmd_name, [1])
    pyleopart.transfer_to_particles(
        ptcls, ptcls.field(pmd_name), pmd_dg0._cpp_object)


# Adaptive time stepping parameters
dt_scl = 0.97  # Scale factor in (0.0, 1.0]
du_max = 0.1   # Maximum permitted change between steps
dt_min = 1e-6  # Minimum time step
dt_max = 1.0   # Maximum time step
euler_o = 1.0  # Euler method order

# Initial parameters
dt.value = 1e-5
t = 0.0
max_steps = 5
sigma_d0.value = 1e1

# Data snapshot label storage
t_snapshots = []
snapshot_interval = 2

# Transfer discrete initial data to FE continuum
pyleopart.transfer_to_function(
    u_img._cpp_object, ptcls, ptcls.field(noise_label))
u_img.x.scatter_forward()
uh.interpolate(u_img)
uh.x.scatter_forward()

# Main loop
for n in range(max_steps):
    if n > 0:
        # Adapt time step
        diff = uh.vector - um.vector
        diff.abs()
        du = diff.max()[1]
        dt_new = dt_scl * dt.value * (du_max / (euler_o * du))
        dt_new = min(max(dt_new, dt_min), dt_max)
        dt.value = dt_new
        PETSc.Sys.Print(
            f"Adapting time step: step={n}, dt={dt.value:.3e}, t={t:.3e}")

    # Solve system
    t += dt.value
    um.x.array[:] = uh.x.array
    um.x.scatter_forward()
    solver.solve(uh)
    if n % snapshot_interval == 0 or n == max_steps - 1:
        record_snapshot(create_timestamp(t))
        t_snapshots += [t]

# Gather data on process 0
r = mesh.comm.gather(ptcls.field("r").data())
c = mesh.comm.gather(ptcls.field("c").data())
if mesh.comm.rank == 0:
    r, c = (np.asarray(np.concatenate(x), dtype=int) for x in (r, c))


def gather_img_rank0(field_name, dtype):
    value_shape = ptcls.field(field_name).value_shape
    data = mesh.comm.gather(ptcls.field(field_name).data())
    img = None
    if mesh.comm.rank == 0:
        data = np.concatenate(data)
        img = np.zeros((*img_size, value_shape[0]), dtype=dtype)
        img[r.ravel(), c.ravel(), ...] = data
    return img


img_gray = gather_img_rank0(gray_label, dtype=np.double)
img_gray_noise = gather_img_rank0(noise_label, dtype=np.double)
img_tstep_gray = [gather_img_rank0("u_" + create_timestamp(t), dtype=np.double)
                  for t in t_snapshots]
img_tstep_pmd = [gather_img_rank0("pmd_" + create_timestamp(t), dtype=np.double)
                  for t in t_snapshots]

# Plot results
if mesh.comm.rank == 0:
    n_figs = len(t_snapshots) + 1
    fig, axs = plt.subplots(2, n_figs, figsize=(16, 2 * 16 / n_figs))
    axs[0, 0].imshow(img_gray, "gray")
    axs[0, 0].set_title("Original grayscale")
    axs[1, 0].imshow(img_gray_noise, "gray")
    axs[1, 0].set_title("Noisy grayscale")

    for i, t in enumerate(t_snapshots):
        axs[0, i + 1].imshow(img_tstep_gray[i], "gray")
        axs[0, i + 1].set_title(f"u({create_timestamp(t)})")

        axs[1, i + 1].imshow(img_tstep_pmd[i], "gray")
        axs[1, i + 1].set_title(rf"$A(\nabla u, {create_timestamp(t)})$")

    for ax in axs.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()
