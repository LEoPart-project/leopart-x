# Copyright (c) 2023 Nathan Sime
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

import enum

from mpi4py import MPI
from petsc4py import PETSc

import adios2
import numpy as np
import scipy.spatial

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
p = 1                   # velocity field polynomial order
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
if mesh0.comm.rank == 0:
    nparts_per_dir = 256
    xp_pts = [np.linspace(0.0, lmbda, nparts_per_dir),
              np.linspace(0.0, H, nparts_per_dir),
              [0.0]]
    xp = np.array(np.meshgrid(*xp_pts), dtype=np.double).T.reshape((-1, 3))
    p2cell = np.array([0] * xp.shape[0], dtype=np.int32)
else:
    xp = np.array([], dtype=np.double)
    p2cell = np.array([], dtype=np.int32)

# Create particles and associate chemistry data with each particle
pprint(f"num paticles: {xp.shape[0]}")
ptcls = pyleopart.Particles(xp, p2cell)
ptcls.relocate_bbox(mesh0._cpp_object, ptcls.active_pidxs())
tableau.check_and_create_fields(ptcls)
chem_label = "C"
ptcls.add_field(chem_label, [1])

# Create function space for continuum representation of chemistry data
chem_space = dolfinx.fem.FunctionSpace(mesh0, ("DG", 0))
C = dolfinx.fem.Function(chem_space)
C.interpolate(lambda x: np.ones_like(x[0], dtype=np.double), cells_top)
C.interpolate(lambda x: np.zeros_like(x[0], dtype=np.double), cells_bottom)

# Initialise particle chemistry data
pyleopart.transfer_to_particles(ptcls, ptcls.field(chem_label), C._cpp_object)


def refine_mesh(mesh):
    """
    Refine the mesh based on the particles' chemistry data. If a cell contains
    particles with non-uniform chemistry data, refine that cell. This is a
    basic scheme for refining the mesh at the two species' interface.
    """
    # Determine if the chemistry data in each cell is uniform
    ptcls.relocate_bbox(mesh._cpp_object, ptcls.active_pidxs())
    c2p = ptcls.cell_to_particle()
    C_data = np.asarray(ptcls.field("C").data(), dtype=np.int32)
    cell_markers = []
    for c in range(mesh.topology.index_map(mesh.topology.dim).size_local):
        C_vals = np.unique(C_data[np.array(c2p[c], dtype=int)])
        if C_vals.shape[0] > 1:
            cell_markers.append(c)

    # Refine those cells which have non-uniform chemistry data
    mesh.topology.create_entities(1)
    edges_to_ref = dolfinx.mesh.compute_incident_entities(
        mesh.topology, cell_markers, mesh.topology.dim, 1)
    new_mesh, _, _ = dolfinx.cpp.refinement.refine_plaza(
        mesh._cpp_object, edges_to_ref, True,
        dolfinx.mesh.RefinementOption.none)
    return dolfinx.mesh.Mesh(
        new_mesh, ufl.Mesh(mesh._ufl_domain.ufl_coordinate_element()))


rng = np.random.default_rng()
def balance_ptcls(ptcls, mesh, np_min, np_max):
    """
    To encourage efficiency and appropriate resolution we add and remove
    particles in each cell.

    Args:
        ptcls: Particles
        mesh: Mesh on which to balance the particles
        np_min: Minimum number of particles per cell
        np_max: Maximum number of particles per cell
    """
    # Remove particles from overpopulated cells
    c2p = ptcls.cell_to_particle()
    np_per_cell = list(map(len, c2p))
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    for c in range(num_cells):
        np_this_cell = np_per_cell[c]
        for j in range(np_max, np_this_cell):
            ptcls.delete_particle(c, rng.integers(np_this_cell))
            np_this_cell -= 1

    # Determine the cells to populate with additional particles
    cells_to_populate = []
    np_per_cell_vec = []
    for c in range(num_cells):
        if len(c2p[c]) < np_min:
            cells_to_populate += [c]
            np_per_cell_vec += [np_min - len(c2p[c])]
    xp_new, p2c_new = pyleopart.mesh_fill(
        mesh._cpp_object, np_per_cell_vec, cells_to_populate)
    xp_new = np.c_[xp_new, np.zeros_like(xp_new[:, 0])]

    # Add new particles where their data is assigned to be identical to the
    # data of the closest existing particle
    active_pidxs = ptcls.active_pidxs()
    active_pts = ptcls.x().data()[active_pidxs]
    kdtree = scipy.spatial.cKDTree(active_pts)
    closest_pidxs = active_pidxs[kdtree.query(xp_new)[1]]

    for xp, c, closest in zip(xp_new, p2c_new, closest_pidxs):
        new_pidx = ptcls.add_particle(xp, c)
        ptcls.field("C").data()[new_pidx] = ptcls.field("C").data()[closest]


class Problem:

    def __init__(self, comm):
        """
        Utility class for computing the velocity approximation to be used
        in the particles' Runge-Kutta advection scheme.

        Args:
            comm: MPI communicator
        """
        self._solver = PETSc.KSP().create(comm)
        self._solver.setType(PETSc.KSP.Type.PREONLY)
        self._solver.getPC().setType(PETSc.PC.Type.LU)
        self._solver.getPC().setFactorSolverType("mumps")

    def velocity(self, t: float, do_refine: bool = True):
        """
        Velocity computation function to be used in Ruge-Kutta advection

        Args:
            t: Time step
            do_refine: Refine the mesh if available

        Returns:
        The velocity approximation DOLFINx function
        """
        # Compute the new refined mesh and relocate all particles
        new_mesh = mesh0
        if do_refine:
            for i in range(nref):
                new_mesh = refine_mesh(new_mesh)
        ptcls.relocate_bbox(new_mesh._cpp_object, ptcls.active_pidxs())
        mesh = new_mesh

        # Project the chemistry data into a continuum approximation
        chem_space = dolfinx.fem.FunctionSpace(mesh, ("DG", 0))
        C = dolfinx.fem.Function(chem_space)
        ptcls.relocate_bbox(mesh._cpp_object, ptcls.active_pidxs())

        # Check for cells which are deficient of particle data required for
        # the projection
        deficient_cells = pyleopart.find_deficient_cells(C._cpp_object, ptcls)
        if len(deficient_cells) > 0:
            pprint(f"Particle deficient cells found: {deficient_cells}",
                   rank=mesh.comm.rank)

        # Do the projection
        pyleopart.transfer_to_function(
            C._cpp_object, ptcls, ptcls.field(chem_label))
        C.x.scatter_forward()

        # Fromulate the Stokes system for the velocity approximation
        # Viscosity model and buoyancy terms
        eta_dense = dolfinx.fem.Constant(mesh, 1.0)
        eta_light = dolfinx.fem.Constant(mesh, 0.01)
        mu = eta_light + C * (eta_dense - eta_light)
        Rb = dolfinx.fem.Constant(mesh, 1.0)
        f = Rb * C * ufl.as_vector((0, -1))

        # Two numerical schemes are provided:
        # c0sipg: divergence free
        # Taylor-Hood: non divergence free
        class Scheme(enum.Enum):
            c0sipg = enum.auto()
            taylor_hood = enum.auto()

        scheme = Scheme.c0sipg

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
                marker=lambda x:
                    np.isclose(x[0], 0.0) | np.isclose(x[0], lmbda))

            V_x = W.sub(0).sub(0).collapse()
            zero = dolfinx.fem.Function(V_x[0])
            dofs_lr = dolfinx.fem.locate_dofs_topological(
                (W.sub(0).sub(0), V_x[0]), mesh.topology.dim - 1,
                facets_left_right)
            zero_x_bc = dolfinx.fem.dirichletbc(zero, dofs_lr, W.sub(0).sub(0))

            W0 = W.sub(0).collapse()
            zero = dolfinx.fem.Function(W0[0])
            dofs_tb = dolfinx.fem.locate_dofs_topological(
                (W.sub(0), W0[0]), mesh.topology.dim - 1, facets_top_bot)
            zero_y_bc = dolfinx.fem.dirichletbc(zero, dofs_tb, W.sub(0))

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
            b.ghostUpdate(
                addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            dolfinx.fem.set_bc(b, bcs)

            # Solve linear system and update ghost values in the solution
            self._solver.getPC().reset()
            self._solver.setOperators(A)
            self._solver.solve(b, x)
            Uh.x.scatter_forward()
            uh = Uh.sub(0)
        elif scheme is Scheme.c0sipg:
            if mesh.ufl_cell() != ufl.triangle:
                err_msg = (
                    "Non-affine cells require careful interpolation to "
                    "appropriate velocity spaces")
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
                return ufl.as_matrix(
                    [[ufl.inner(G[i, k, :, :], tau) for k in range(d)]
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
                    + ufl.inner(
                        beta("+") * G_mult(ufl.avg(G), tensor_jump(u, n)),
                        tensor_jump(v, n))) * ufl.dS
                exterior = (
                    - ufl.inner(ufl.outer(u, n), sigma(v))
                    - ufl.inner(ufl.outer(v, n), sigma(u))
                    + ufl.inner(beta * G_mult(G, ufl.outer(u, n)),
                                ufl.outer(v, n))) * ds_0
                return domain + interior + exterior

            def lh(v):
                domain = ufl.inner(f, v) * ufl.dx
                exterior = (
                    - ufl.inner(ufl.outer(zero_u, n), sigma(v))
                    + ufl.inner(beta * G_mult(G, ufl.outer(zero_u, n)),
                               ufl.outer(v, n))) * ds_0
                return domain + exterior

            # Linear problem and solver
            phi_h = dolfinx.fem.Function(PSI)

            a = dolfinx.fem.form(Bh(ufl.curl(phi), ufl.curl(psi)))
            A = dolfinx.fem.petsc.create_matrix(a)
            L = dolfinx.fem.form(lh(ufl.curl(psi)))
            b = dolfinx.fem.petsc.create_vector(L)
            x = phi_h.vector

            bcs = [zero_bc]
            A.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix_mat(A, a, bcs=bcs)
            A.assemble()

            # Assemble rhs
            with b.localForm() as b_loc:
                b_loc.set(0)
            dolfinx.fem.petsc.assemble_vector(b, L)

            # Apply boundary conditions to the rhs
            dolfinx.fem.apply_lifting(b, [a], bcs=[bcs])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                          mode=PETSc.ScatterMode.REVERSE)
            dolfinx.fem.set_bc(b, bcs)

            # Solve linear system and update ghost values in the solution
            self._solver.getPC().reset()
            self._solver.setOperators(A)
            self._solver.solve(b, x)
            phi_h.x.scatter_forward()

            # Velocity function space into which we interpolate curl(phi)
            uh_spc = dolfinx.fem.FunctionSpace(mesh, ("DG", p, (2,)))
            uh_expr = dolfinx.fem.Expression(
                ufl.curl(phi_h), uh_spc.element.interpolation_points())
            uh = dolfinx.fem.Function(uh_spc)
            uh.interpolate(uh_expr)

        self.last_mesh = mesh
        self.last_vel = uh
        return uh._cpp_object

    def relocator(self, ptcls):
        """
        When using a Runge-Kutta scheme where the mesh changes between steps
        we provide a custom particle relocation method for callback.

        Args:
            ptcls: Particles to relocate
        """
        ptcls.relocate_bbox(self.last_mesh._cpp_object, ptcls.active_pidxs())

    def estimate_dt(self, h, c_cfl):
        """
        The time step is estimated from the Courant-Friedrichs-Lewy condition.

        Args:
            h: Mesh cell size
            c_cfl: CFL number

        Returns:
        Time step approximation
        """
        # Space for estimating speed for CFL criterion
        Vspd = dolfinx.fem.FunctionSpace(self.last_mesh, ("DG", 0))
        uspd = dolfinx.fem.Function(Vspd)

        uh = self.last_vel
        uspd_expr = dolfinx.fem.Expression(
            ufl.sqrt(ufl.inner(uh, uh)), Vspd.element.interpolation_points())
        uspd.interpolate(uspd_expr)
        uspd.vector.assemble()
        max_u_vec = uspd.vector.norm(PETSc.NormType.INF)
        return c_cfl * h / max_u_vec

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

for j in range(num_t_steps_max):
    # Estimate dt based on CFL criterion
    dt = problem.estimate_dt(hmin, 1.0)
    pprint(f"Time step {j}, dt = {dt:.3e}, t = {t:.3e}", rank=0)

    # Enact explicit Runge-Kutta integration
    new_mesh = mesh0
    do_refine = True
    if j > 1:
        if do_refine:
            for i in range(nref):
                new_mesh = refine_mesh(new_mesh)
        ptcls.relocate_bbox(new_mesh._cpp_object, ptcls.active_pidxs())
        balance_ptcls(ptcls, new_mesh, np_min=50, np_max=55)
        ptcls.relocate_bbox(mesh0._cpp_object, ptcls.active_pidxs())
    pyleopart.rk(ptcls, tableau, problem.velocity, problem.relocator, t, dt)
    t += dt

    # Output data
    ptcl_file.write_particles(ptcls, t, [chem_label])
    chem_space = dolfinx.fem.FunctionSpace(problem.last_mesh, ("DG", 0))
    C = dolfinx.fem.Function(chem_space)

    # Check for particle deficient cells prior to l2 projection
    deficient_cells = pyleopart.find_deficient_cells(C._cpp_object, ptcls)
    if len(deficient_cells) > 0:
        pprint(
            f"Deficient cells prior to transfer to function: {deficient_cells}",
            rank=problem.last_mesh.comm.rank)
    pyleopart.transfer_to_function(
        C._cpp_object, ptcls, ptcls.field(chem_label))

    C.x.scatter_forward()
    chem_file = dolfinx.io.XDMFFile(
        problem.last_mesh.comm, f"chem/C_{j:04d}.xdmf", "w")
    chem_file.write_mesh(problem.last_mesh)
    chem_file.write_function(C, t=t)

    if t > t_max:
        break
