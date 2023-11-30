import imageio
import numpy as np
import dolfinx
from mpi4py import MPI
import leopart.cpp as pyleopart
import febug
import matplotlib.pyplot as plt
import ufl

mesh = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, 150, 150, cell_type=dolfinx.mesh.CellType.quadrilateral)

if mesh.comm.rank == 0:
    # Image: https://www.flickr.com/photos/61027726@N06/5558338829
    # https://openverse.org/image/7abac9f4-fadb-4de2-a182-680d006e4404?q=leopard
    # "Snow Leopard" by Sideshow_Matt is licensed under CC BY 2.0.
    data = imageio.v2.imread("leopard.jpg")
    L, H = data.shape[:2]
    x, y = np.meshgrid(np.linspace(0.0, 1.0, L), np.linspace(0.0, 1.0, H))
    xp = np.c_[x.ravel(), y.ravel(), np.zeros_like(x.ravel())]
else:
    xp = np.zeros(0, dtype=np.double)
    data = np.zeros((0, 0, 3), dtype=np.double)

p2c = [0] * xp.shape[0]
ptcls = pyleopart.Particles(xp, p2c)

for idx, color in enumerate(("r", "g", "b")):
    ptcls.add_field(color, [1])
    ptcls.field(color).data().T[:] = data[:,:,idx].ravel()

ptcls.relocate_bbox(mesh._cpp_object, np.arange(len(p2c)))


DG0 = dolfinx.fem.FunctionSpace(mesh, ("DG", 0))
u_img = dolfinx.fem.Function(DG0)

V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

k = dolfinx.fem.Constant(mesh, 1e-3)
a = ufl.inner(k * ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(u, v) * ufl.dx
L = ufl.inner(u_img, v) * ufl.dx

facets = dolfinx.mesh.locate_entities_boundary(
    mesh, mesh.topology.dim - 1,
    lambda x: np.full_like(x[0], 1, dtype=np.int8))
dofs = dolfinx.fem.locate_dofs_topological(
    V, mesh.topology.dim - 1, facets)
u_bc = dolfinx.fem.Function(V)
u_bc.interpolate(u_img)
bc = dolfinx.fem.dirichletbc(u_bc, dofs)

import dolfinx.fem.petsc
problem = dolfinx.fem.petsc.LinearProblem(
    a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

for idx, (color, k_value) in enumerate(
        (("r", 1e-9), ("g", 1e-4), ("b", 1e-4))):
    pyleopart.transfer_to_function(u_img._cpp_object, ptcls, ptcls.field(color))
    k.value = k_value
    uh = problem.solve()

    pyleopart.transfer_to_particles(ptcls, ptcls.field(color), uh._cpp_object)
    data[:,:,idx] = ptcls.field(color).data().reshape(data[:,:, idx].shape)

plt.imshow(data)
plt.show()
quit()

febug.plot_function(uh).show()