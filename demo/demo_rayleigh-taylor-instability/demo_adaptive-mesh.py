# Copyright (c) 2023 Nathan Sime
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

import enum

from mpi4py import MPI
from petsc4py import PETSc

import adios2
import numpy as np

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
tableau = pyleopart.tableaus.order2.generic_alpha(0.5)
t_max = 1500.0
num_t_steps_max = 500
nref = 1

# Geometry
mesh0 = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD, [[0.0, 0.0], [lmbda, H]], [20, 20],
    cell_type=dolfinx.mesh.CellType.triangle,
    diagonal=dolfinx.mesh.DiagonalType.left_right)

# Chemistry space. A value of 1 corresponds to dense material, and 0 light
# material
cells_bottom = dolfinx.mesh.locate_entities(
    mesh0, mesh0.topology.dim, lambda x: x[1] <= db)
cells_top = dolfinx.mesh.locate_entities(
    mesh0, mesh0.topology.dim, lambda x: x[1] >= db)

# Perturb the mesh and initial condition
meshx = mesh0.geometry.x
S = db * (1.0 - db)
meshx[:, 1] += meshx[:, 1] * (H - meshx[:, 1]) / S \
              * A * np.cos(np.pi * lmbda * meshx[:, 0])

# Generate particles and interpolate the composition
# xp, p2cell = pyleopart.mesh_fill(mesh0._cpp_object, 50)
# xp = np.c_[xp, np.zeros_like(xp[:, 0])]
if mesh0.comm.rank == 0:
    n_dir = 256
    xp_pts = [np.linspace(0.0, lmbda, n_dir),
              np.concatenate((
                np.linspace(0.0, H, n_dir),
                np.linspace(db - 0.025, db + 0.025, 64)
              )),
              [0.0]]
    xp = np.array(np.meshgrid(*xp_pts), dtype=np.double).T.reshape((-1, 3))
    p2cell = np.array([0]*xp.shape[0], dtype=np.int32)
else:
    xp = np.array([], dtype=np.double)
    p2cell = np.array([], dtype=np.int32)


pprint(f"num paticles: {xp.shape[0]}")
ptcls = pyleopart.Particles(xp, p2cell)
ptcls.relocate_bbox(mesh0._cpp_object, ptcls.active_pidxs())
tableau.check_and_create_fields(ptcls)
chem_label = "C"
ptcls.add_field(chem_label, [1])

chem_space = dolfinx.fem.FunctionSpace(mesh0, ("DG", 0))
C = dolfinx.fem.Function(chem_space)
C.interpolate(lambda x: np.ones_like(x[0], dtype=np.double), cells_top)
C.interpolate(lambda x: np.zeros_like(x[0], dtype=np.double), cells_bottom)
pyleopart.transfer_to_particles(ptcls, ptcls.field(chem_label), C._cpp_object)

def refine_mesh(mesh):
    ptcls.relocate_bbox(mesh._cpp_object, ptcls.active_pidxs())
    c2p = ptcls.cell_to_particle()
    C_data = np.asarray(ptcls.field("C").data(), dtype=np.int32)
    cell_markers = []
    for c in range(mesh.topology.index_map(mesh.topology.dim).size_local):
        C_vals = np.unique(C_data[np.array(c2p[c], dtype=int)])
        if C_vals.shape[0] > 1:
            cell_markers.append(c)

    mesh.topology.create_entities(1)
    edges_to_ref = dolfinx.mesh.compute_incident_entities(
        mesh.topology, cell_markers, mesh.topology.dim, 1)
    new_mesh, _, _ = dolfinx.cpp.refinement.refine_plaza(
        mesh._cpp_object, edges_to_ref, True,
        dolfinx.mesh.RefinementOption.none)
    new_mesh = dolfinx.mesh.Mesh(
        new_mesh, ufl.Mesh(mesh._ufl_domain.ufl_coordinate_element()))

    return new_mesh

import scipy.spatial
def balance_ptcls(ptcls, mesh, np_min, np_max):
    c2p = ptcls.cell_to_particle()
    np_per_cell = list(map(len, c2p))
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    for c in range(num_cells):
        np_this_cell = np_per_cell[c]
        for j in range(np_max, np_this_cell):
            ptcls.delete_particle(c, np.random.randint(np_this_cell))
            np_this_cell -= 1

    cells_to_populate = []
    np_per_cell_vec = []
    for c in range(num_cells):
        if len(c2p[c]) < np_min:
            cells_to_populate += [c]
            np_per_cell_vec += [np_min - len(c2p[c])]
    xp_new, p2c_new = pyleopart.mesh_fill(
        mesh._cpp_object, np_per_cell_vec, cells_to_populate)
    xp_new = np.c_[xp_new, np.zeros_like(xp_new[:, 0])]

    active_pidxs = ptcls.active_pidxs()
    active_pts = ptcls.x().data()[active_pidxs]
    kdtree = scipy.spatial.cKDTree(active_pts)
    closest_pidxs = active_pidxs[kdtree.query(xp_new)[1]]

    for xp, c, closest in zip(xp_new, p2c_new, closest_pidxs):
        new_pidx = ptcls.add_particle(xp, c)
        ptcls.field("C").data()[new_pidx] = ptcls.field("C").data()[closest]

    ptcl_file = leopart.io.XDMFParticlesFile(
        mesh0.comm, "debug.xdmf", adios2.Mode.Write)
    ptcl_file.write_particles(ptcls, 0.0, ["C"])

class Problem:

    def __init__(self, comm):
        self._solver = PETSc.KSP().create(comm)
        self._solver.setType(PETSc.KSP.Type.PREONLY)
        self._solver.getPC().setType(PETSc.PC.Type.LU)
        self._solver.getPC().setFactorSolverType("mumps")

    def velocity(self, t, do_refine=True):
        new_mesh = mesh0
        if do_refine:
            for i in range(nref):
                new_mesh = refine_mesh(new_mesh)
        ptcls.relocate_bbox(new_mesh._cpp_object, ptcls.active_pidxs())
        mesh = new_mesh

        chem_space = dolfinx.fem.FunctionSpace(mesh, ("DG", 0))
        C = dolfinx.fem.Function(chem_space)
        ptcls.relocate_bbox(mesh._cpp_object, ptcls.active_pidxs())

        deficient_cells = pyleopart.find_deficient_cells(C._cpp_object, ptcls)
        if len(deficient_cells) > 0:
            pprint(f"Particle deficient cells found: {deficient_cells}",
                   rank=mesh.comm.rank)

            import febug
            plotter = febug.plot_mesh(mesh)
            import pyvista
            pd = pyvista.PolyData(ptcls.x().data()[ptcls.active_pidxs()])
            plotter.add_mesh(pd)
            plotter.show()
        pyleopart.transfer_to_function(
            C._cpp_object, ptcls, ptcls.field(chem_label))
        C.x.scatter_forward()

        # import febug
        # febug.plot_function(C).show()

        # Viscosity model and buoyancy terms
        eta_dense = dolfinx.fem.Constant(mesh, 1.0)
        eta_light = dolfinx.fem.Constant(mesh, 0.01)
        mu = eta_light + C * (eta_dense - eta_light)
        Rb = dolfinx.fem.Constant(mesh, 1.0)
        f = Rb * C * ufl.as_vector((0, -1))

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

        # problem = dolfinx.fem.petsc.LinearProblem(
        #     a, L, u=Uh, bcs=[zero_x_bc, zero_y_bc],
        #     petsc_options={"ksp_type": "preonly", "pc_type": "lu",
        #                    "pc_factor_mat_solver_type": "mumps"})
        #
        # # Velocity as function of time to be used in Runge-Kutta integration
        # uh = Uh.sub(0)
        # problem.solve()

        a = dolfinx.fem.form(a)
        A = dolfinx.fem.petsc.create_matrix(a)
        L = dolfinx.fem.form(L)
        b = dolfinx.fem.petsc.create_vector(L)
        x = dolfinx.la.create_petsc_vector_wrap(Uh.x)

        # Assemble lhs
        bcs = [zero_x_bc, zero_y_bc]
        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix_mat(A, a, bcs=bcs)
        A.assemble()

        # Assemble rhs
        with b.localForm() as b_loc:
            b_loc.set(0)
        dolfinx.fem.petsc.assemble_vector(b, L)

        # Apply boundary conditions to the rhs
        dolfinx.fem.apply_lifting(b, [a], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.set_bc(b, bcs)

        # Solve linear system and update ghost values in the solution
        self._solver.getPC().reset()
        self._solver.setOperators(A)
        self._solver.solve(b, x)
        Uh.x.scatter_forward()

        uh = Uh.sub(0)
        self.last_mesh = mesh
        self.last_vel = uh
        # del problem
        return uh._cpp_object

    def relocator(self, ptcls):
        ptcls.relocate_bbox(self.last_mesh._cpp_object, ptcls.active_pidxs())

    def estimate_dt(self, h, c_cfl):
        # Space for estimating speed for CFL criterion
        Vspd = dolfinx.fem.FunctionSpace(self.last_mesh, ("DG", 0))
        uspd = dolfinx.fem.Function(Vspd)

        uh = self.last_vel
        uspd_expr = dolfinx.fem.Expression(
            ufl.sqrt(ufl.inner(uh, uh)), Vspd.element.interpolation_points())
        uspd.interpolate(uspd_expr)
        uspd.vector.assemble()
        max_u_vec = uspd.vector.norm(PETSc.NormType.INF)
        dt = c_cfl * h / max_u_vec
        return dt

# h measured used in CFL criterion estimation
h_measure = dolfinx.cpp.mesh.h(
    mesh0._cpp_object, mesh0.topology.dim, np.arange(
        mesh0.topology.index_map(mesh0.topology.dim).size_local,
        dtype=np.int32))
hmin = mesh0.comm.allreduce(h_measure.min(), op=MPI.MIN)

# Output files
ptcl_file = leopart.io.XDMFParticlesFile(
    mesh0.comm, "particles.xdmf", adios2.Mode.Write)


# Main time loop
t = 0.0
ptcl_file.write_particles(ptcls, t, [chem_label])
problem = Problem(mesh0.comm)
problem.velocity(0.0)  # Initial velocity estimate

# Record u_rms as a function of time
t_vals = [0.0]
# urms_vals = []
# urms_vals.append(compute_urms())
for j in range(num_t_steps_max):
    # Estimate dt based on CFL criterion
    dt = problem.estimate_dt(hmin, 1.0)
    print(dt)
    pprint(f"Time step {j}, dt = {dt:.3e}, t = {t:.3e}", rank=0)

    # Enact explicit Runge-Kutta integration
    new_mesh = mesh0
    do_refine = True
    if j > 1:
        if do_refine:
            for i in range(nref):
                new_mesh = refine_mesh(new_mesh)
        ptcls.relocate_bbox(new_mesh._cpp_object, ptcls.active_pidxs())
        balance_ptcls(ptcls, new_mesh, np_min=25, np_max=30)
        ptcls.relocate_bbox(mesh0._cpp_object, ptcls.active_pidxs())
    pyleopart.rk(ptcls, tableau, problem.velocity, problem.relocator, t, dt)
    t += dt

    # Output data
    ptcl_file.write_particles(ptcls, t, [chem_label])
    chem_space = dolfinx.fem.FunctionSpace(problem.last_mesh, ("DG", 0))
    C = dolfinx.fem.Function(chem_space)
    pyleopart.transfer_to_function(
        C._cpp_object, ptcls, ptcls.field(chem_label))
    C.x.scatter_forward()
    chem_file = dolfinx.io.XDMFFile(
        problem.last_mesh.comm, f"C_{j:04d}.xdmf", "w")
    chem_file.write_mesh(problem.last_mesh)
    chem_file.write_function(C, t=t)

    # urms_vals.append(compute_urms())
    t_vals.append(t)

    if t > t_max:
        break

# if mesh0.comm.rank == 0:
#     # Plot u_rms data
#     import matplotlib.pyplot as plt
#     plt.plot(t_vals, urms_vals)
#     plt.show()
