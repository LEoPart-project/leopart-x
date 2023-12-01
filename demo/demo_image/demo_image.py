import imageio
import numpy as np
import dolfinx
from mpi4py import MPI
import leopart.cpp as pyleopart
import febug
import matplotlib.pyplot as plt
import ufl

# dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
mesh = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, 150, 150, cell_type=dolfinx.mesh.CellType.quadrilateral)

if mesh.comm.rank == 0:
    # Image: https://www.flickr.com/photos/61027726@N06/5558338829
    # https://openverse.org/image/7abac9f4-fadb-4de2-a182-680d006e4404?q=leopard
    # "Snow Leopard" by Sideshow_Matt is licensed under CC BY 2.0.
    data = imageio.v2.imread("leopard2.jpg")
    L, H = data.shape[:2]
    x, y = np.meshgrid(np.linspace(0.0, 1.0, L), np.linspace(0.0, 1.0, H))
    xp = np.c_[x.ravel(), y.ravel(), np.zeros_like(x.ravel())]
else:
    xp = np.zeros(0, dtype=np.double)
    data = np.zeros((0, 0, 3), dtype=np.double)

p2c = [0] * xp.shape[0]
ptcls = pyleopart.Particles(xp, p2c)

ptcls.add_field("L", [1])
if mesh.comm.rank == 0:
    greyscale = np.dot(data[..., :3], [0.299, 0.587, 0.114]).ravel() / 255.0
    noise = np.random.default_rng().random(greyscale.shape) * 0.5
    ptcls.field("L").data().T[:] = greyscale + noise
ptcls.relocate_bbox(mesh._cpp_object, np.arange(len(p2c)))

DG0 = dolfinx.fem.FunctionSpace(mesh, ("DG", 0))
u_img = dolfinx.fem.Function(DG0)

V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
u, v = dolfinx.fem.Function(V), ufl.TestFunction(V)
um = dolfinx.fem.Function(V)
uth = 0.5 * (u + um)


sigma_d0 = dolfinx.fem.Constant(mesh, 0.0)
pm_d0 = dolfinx.fem.Constant(mesh, 0.0)
pm_d = pm_d0 * ufl.exp(
    - ufl.inner(ufl.grad(uth), ufl.grad(uth)) / (2 * sigma_d0**2))

dt = dolfinx.fem.Constant(mesh, 0.0)
F = ufl.inner((u - um) / dt, v) * ufl.dx
F += ufl.inner(pm_d * ufl.grad(uth), ufl.grad(v)) * ufl.dx

import dolfinx.fem.petsc
import dolfinx.nls.petsc

problem = dolfinx.fem.petsc.NonlinearProblem(F, u)
solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
solver.max_it = 10
solver.rtol = 1e-5
# solver.error_on_nonconvergence = False

from petsc4py import PETSc
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

pyleopart.transfer_to_function(u_img._cpp_object, ptcls, ptcls.field("L"))

u.interpolate(u_img)
pm_d0.value = 1e-4
sigma_d0.value = 1.0

data_t0 = ptcls.field("L").data().reshape(data[:,:,0].shape).copy()
dt.value = 0.1
for i in range(200):
    if i % 100 == 0:
        print(f"i = {i}")
    um.interpolate(u)
    solver.solve(u)

pyleopart.transfer_to_particles(ptcls, ptcls.field("L"), u._cpp_object)
data_new = ptcls.field("L").data().reshape(data[:,:,0].shape).copy()

fig, axs = plt.subplots(1, 3)
axs[0].imshow(data)
axs[1].imshow(data_t0, "gray")
axs[2].imshow(data_new, "gray")
plt.show()
quit()

febug.plot_function(uh).show()