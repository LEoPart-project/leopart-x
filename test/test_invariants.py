# Copyright (c) 2026 Nathan Sime
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Invariants that must hold however particles are stored.

These tests check only what a user can observe: which particles exist,
where they are, which cell holds them and what data they carry. They are
the safety net for changing the particle storage (ROADMAP.md, Phase 1),
so they must not depend on storage order or particle indices.

The helpers ``active_indices``, ``particle_cells``,
``cell_particle_lists`` and ``delete_cell_particles`` are the only code
here that touches the storage interface. Update them, not the tests, when
that interface changes.
"""

import itertools

from mpi4py import MPI

import numpy as np
import pytest

import dolfinx
import leopart.cpp as pyleopart

CELL_TYPES = [
    dolfinx.mesh.CellType.triangle,
    dolfinx.mesh.CellType.quadrilateral,
    dolfinx.mesh.CellType.tetrahedron,
    dolfinx.mesh.CellType.hexahedron,
]

# Number of cells in each direction, chosen per topological dimension so
# that each test runs in a few seconds and still spans several ranks
NUM_CELLS = {2: 8, 3: 4}

# Squared distance within which dolfinx::geometry::compute_colliding_cells
# treats a point as inside a cell
DOLFINX_COLLISION_TOL2 = 1e-12


def create_mesh(cell_type, n=None):
    tdim = dolfinx.mesh.cell_dim(cell_type)
    if n is None:
        n = NUM_CELLS[tdim]
    if tdim == 2:
        return dolfinx.mesh.create_unit_square(
            MPI.COMM_WORLD, n, n, cell_type=cell_type)
    return dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD, n, n, n, cell_type=cell_type)


def assert_on_all_ranks(comm, local_ok, msg):
    """Assert a condition checked on each rank, failing on every rank
    together. If only one rank failed, the others would carry on into the
    next collective call and the test run would hang."""
    ok = comm.allgather(bool(local_ok))
    failed = [rank for rank, rank_ok in enumerate(ok) if not rank_ok]
    assert not failed, f"{msg} (failed on ranks {failed})"


def num_local_cells(mesh):
    return mesh.topology.index_map(mesh.topology.dim).size_local


def pad_to_3d(x):
    if x.shape[1] == 3:
        return x
    return np.c_[x, np.zeros((x.shape[0], 3 - x.shape[1]))]


def active_indices(ptcls):
    """Process local indices of the particles that exist."""
    return np.arange(ptcls.num_particles())


def particle_cells(ptcls, pidxs):
    """Process local cell containing each of the given particles."""
    return np.asarray(ptcls.particle_to_cell(), dtype=np.int32)[pidxs]


def cell_particle_lists(ptcls):
    """Process local indices of the particles in each cell, as recorded
    by the cell offsets."""
    offsets = ptcls.cell_offsets()
    return [np.arange(offsets[c], offsets[c + 1])
            for c in range(len(offsets) - 1)]


def delete_cell_particles(ptcls, cell):
    """Delete every particle in a cell."""
    offsets = ptcls.cell_offsets()
    ptcls.delete_particles(np.arange(offsets[cell], offsets[cell + 1]))


def payload(gid):
    """Data a particle carries, recomputable from its global ID."""
    return np.c_[np.sin(gid), np.cos(3.0 * gid)]


def create_tagged_particles(mesh, np_per_cell):
    """Fill the mesh with particles, each tagged with a global ID, its
    initial position and a payload derived from the ID.

    Returns the particles and the global number of particles.
    """
    x, cells = pyleopart.mesh_fill(mesh._cpp_object, np_per_cell, seed=1)
    x = pad_to_3d(x)
    ptcls = pyleopart.Particles(x, cells)

    comm = mesh.comm
    n_local = x.shape[0]
    # exscan is collective; its result is undefined (None) on rank 0
    offset = comm.exscan(n_local)
    if comm.rank == 0:
        offset = 0
    gid = np.arange(offset, offset + n_local, dtype=np.float64)

    ptcls.add_field("gid", [1])
    ptcls.add_field("x0", [3])
    ptcls.add_field("w", [2])
    ptcls.field("gid").data()[:, 0] = gid
    ptcls.field("x0").data()[:] = x
    ptcls.field("w").data()[:] = payload(gid)

    return ptcls, comm.allreduce(n_local, op=MPI.SUM)


def small_step(x, gdim):
    """Move each particle by up to half a cell. The displacement vanishes
    on the boundary of the unit square or cube, and the map is monotone
    in each component, so particles stay inside the domain."""
    a = 0.5 / NUM_CELLS[gdim]
    y = x.copy()
    for i in range(gdim):
        j = (i + 1) % gdim
        y[:, i] += (a * 4.0 * x[:, i] * (1.0 - x[:, i])
                    * np.sin(2.0 * np.pi * x[:, j]))
    return y


def reflect(x, gdim):
    """Reflect particles through the centre of the domain, sending most
    of them across the mesh and to another rank."""
    y = x.copy()
    y[:, :gdim] = 1.0 - x[:, :gdim]
    return y


MOTIONS = {"small_step": small_step, "reflect": reflect}


@pytest.mark.parametrize("cell_type", CELL_TYPES)
def test_generated_particles_lie_in_their_cells(cell_type):
    mesh = create_mesh(cell_type)
    np_per_cell = 3
    x, cells = pyleopart.mesh_fill(mesh._cpp_object, np_per_cell, seed=1)
    cells = np.asarray(cells, dtype=np.int32)

    comm = mesh.comm
    n_cells = num_local_cells(mesh)
    assert_on_all_ranks(
        comm, x.shape == (n_cells * np_per_cell, mesh.geometry.dim),
        "Wrong number of generated particles")
    assert_on_all_ranks(
        comm, np.array_equal(np.bincount(cells, minlength=n_cells),
                             np.full(n_cells, np_per_cell)),
        "Particles not spread evenly over cells")

    h = 1.0 / NUM_CELLS[mesh.topology.dim]
    dist2 = dolfinx.geometry.squared_distance(
        mesh, mesh.topology.dim, cells, pad_to_3d(x))
    assert_on_all_ranks(comm, np.all(dist2 < (1e-10 * h) ** 2),
                        "Generated particle outside its cell")


def check_tagged_particles(mesh, ptcls, n_global, expected_x):
    """Check particles made by create_tagged_particles after relocation
    onto ``mesh``. ``expected_x`` maps each particle's initial position
    to where it should now be."""
    comm = mesh.comm
    pidxs = active_indices(ptcls)
    gid = ptcls.field("gid").data()[pidxs, 0]
    x = ptcls.x().data()[pidxs]
    x0 = ptcls.field("x0").data()[pidxs]
    w = ptcls.field("w").data()[pidxs]

    # Every particle exists exactly once across all ranks
    all_gid = np.sort(np.concatenate(comm.allgather(gid)))
    assert np.array_equal(all_gid, np.arange(n_global, dtype=np.float64))

    # Each particle kept its position and its data
    assert_on_all_ranks(
        comm, np.allclose(x, expected_x(x0), rtol=0.0, atol=1e-14),
        "Particle position changed by relocation")
    assert_on_all_ranks(
        comm, np.allclose(w, payload(gid), rtol=0.0, atol=1e-14),
        "Particle field data lost or mixed up by relocation")

    # Each particle is assigned to a local cell that contains it, up to
    # the tolerance of DOLFINx's compute_colliding_cells, which serial
    # relocation uses. A particle near a facet can be assigned to the
    # neighbouring cell it is up to 1e-6 outside.
    n_cells = num_local_cells(mesh)
    cells = particle_cells(ptcls, pidxs)
    in_range = np.all((cells >= 0) & (cells < n_cells))
    assert_on_all_ranks(comm, in_range,
                        "Particle assigned to a non-local cell")
    dist2 = dolfinx.geometry.squared_distance(
        mesh, mesh.topology.dim, cells, x)
    assert_on_all_ranks(comm, np.all(dist2 < DOLFINX_COLLISION_TOL2),
                        "Particle outside its assigned cell")

    # The cell to particle map lists exactly the particles in each cell,
    # and nothing for cells the mesh does not have
    lists = cell_particle_lists(ptcls)
    consistent = all(
        np.array_equal(np.sort(lists[c]) if c < len(lists) else [],
                       np.sort(pidxs[cells == c]))
        for c in range(n_cells))
    assert_on_all_ranks(comm, consistent,
                        "Cell to particle map disagrees with particle cells")
    stale = sum(len(ps) for ps in lists[n_cells:])
    assert_on_all_ranks(comm, stale == 0,
                        f"{stale} particles listed in cells beyond the mesh")


@pytest.mark.parametrize("cell_type", CELL_TYPES)
@pytest.mark.parametrize("motion", MOTIONS.keys())
@pytest.mark.parametrize("n_steps", [1, 3])
def test_relocation_conserves_particles(cell_type, motion, n_steps):
    mesh = create_mesh(cell_type)
    gdim = mesh.geometry.dim
    move = MOTIONS[motion]

    ptcls, n_global = create_tagged_particles(mesh, np_per_cell=4)

    for _ in range(n_steps):
        pidxs = active_indices(ptcls)
        x = ptcls.x().data()
        x[pidxs] = move(x[pidxs], gdim)
        ptcls.relocate_bbox(mesh._cpp_object, pidxs)

    def expected_x(x0):
        for _ in range(n_steps):
            x0 = move(x0, gdim)
        return x0

    check_tagged_particles(mesh, ptcls, n_global, expected_x)


@pytest.mark.parametrize("cell_type", CELL_TYPES)
@pytest.mark.parametrize("refinement", [0.5, 2.0])
def test_relocation_onto_another_mesh(cell_type, refinement):
    # As when a mesh is adapted: the new mesh has a different number of
    # cells and a different partition
    mesh = create_mesh(cell_type)
    ptcls, n_global = create_tagged_particles(mesh, np_per_cell=4)

    n_new = round(NUM_CELLS[mesh.topology.dim] * refinement)
    new_mesh = create_mesh(cell_type, n_new)
    ptcls.relocate_bbox(new_mesh._cpp_object, active_indices(ptcls))

    check_tagged_particles(new_mesh, ptcls, n_global, lambda x0: x0)


@pytest.mark.parametrize("cell_type", CELL_TYPES)
def test_new_particles_start_with_zero_data(cell_type):
    # Particles added to refill a cell must not inherit the data of the
    # deleted particles whose storage they may take over
    mesh = create_mesh(cell_type)
    comm = mesh.comm
    np_per_cell = 4
    ptcls, _ = create_tagged_particles(mesh, np_per_cell)
    ptcls.field("w").data()[:] = 7.0

    emptied_cell = 0
    delete_cell_particles(ptcls, emptied_cell)
    ptcls.generate_minimum_particles_per_cell(mesh._cpp_object, np_per_cell)

    pidxs = active_indices(ptcls)
    cells = particle_cells(ptcls, pidxs)
    w = ptcls.field("w").data()[pidxs]
    new = cells == emptied_cell
    assert_on_all_ranks(
        comm, np.count_nonzero(new) == np_per_cell,
        "Emptied cell not refilled")
    # Exact comparisons: the data is zeroed or left untouched, not computed
    assert_on_all_ranks(
        comm, not np.any(w[new]),
        "New particles inherited data from deleted particles")
    assert_on_all_ranks(
        comm, np.array_equal(w[~new], np.full_like(w[~new], 7.0)),
        "Existing particles' data changed")


def polynomial(x, gdim, k, n_components):
    """A polynomial of total degree k with every monomial present. It lies
    in DG_k on simplices (P_k) and on tensor-product cells (Q_k)."""
    values = np.zeros((n_components, x.shape[1]))
    for comp in range(n_components):
        for exps in itertools.product(range(k + 1), repeat=gdim):
            if sum(exps) > k:
                continue
            coeff = 1.0 + 0.5 * comp + 0.25 * sum(
                (i + 1) * e for i, e in enumerate(exps))
            monomial = np.ones(x.shape[1])
            for i, e in enumerate(exps):
                monomial *= x[i] ** e
            values[comp] += coeff * monomial
    return values


@pytest.mark.parametrize("cell_type", CELL_TYPES)
@pytest.mark.parametrize("k", [0, 1, 2])
@pytest.mark.parametrize("shape", [(1,), (2,)])
def test_projection_reproduces_polynomials(cell_type, k, shape):
    mesh = create_mesh(cell_type)
    gdim = mesh.geometry.dim
    Q = dolfinx.fem.functionspace(mesh, ("DG", k, shape))

    def f(x):
        return polynomial(x, gdim, k, shape[0])

    u_exact = dolfinx.fem.Function(Q)
    u_exact.interpolate(f)

    # Twice as many particles as DoFs per cell, so that the local least
    # squares problem is overdetermined
    np_per_cell = 2 * Q.dofmap.dof_layout.num_dofs
    x, cells = pyleopart.mesh_fill(mesh._cpp_object, np_per_cell, seed=1)
    ptcls = pyleopart.Particles(pad_to_3d(x), cells)
    ptcls.add_field("v", list(shape))
    ptcls.field("v").data()[:] = f(ptcls.x().data().T).T

    u = dolfinx.fem.Function(Q)
    pyleopart.transfer_to_function(u._cpp_object, ptcls, ptcls.field("v"))
    u.x.scatter_forward()

    scale = max(1.0, np.max(np.abs(u_exact.x.array)))
    assert_on_all_ranks(
        mesh.comm,
        np.allclose(u.x.array, u_exact.x.array, rtol=0.0,
                    atol=1e-10 * scale),
        f"Projection does not reproduce a degree {k} polynomial")


@pytest.mark.parametrize("cell_type", CELL_TYPES)
@pytest.mark.parametrize("k", [1, 2])
def test_transfer_independent_of_particle_order(cell_type, k):
    mesh = create_mesh(cell_type)
    Q = dolfinx.fem.functionspace(mesh, ("DG", k))

    np_per_cell = 2 * Q.dofmap.dof_layout.num_dofs
    x, cells = pyleopart.mesh_fill(mesh._cpp_object, np_per_cell, seed=1)
    x = pad_to_3d(x)
    cells = np.asarray(cells, dtype=np.int32)

    # Same particles, given in a different order. Storage is sorted by
    # cell, so this changes the order of the particles within each cell.
    perm = np.random.default_rng(seed=0).permutation(x.shape[0])
    ptcls_a = pyleopart.Particles(x, cells)
    ptcls_b = pyleopart.Particles(np.ascontiguousarray(x[perm]), cells[perm])

    # Data not representable in the space, so that the projection is a
    # genuine least squares fit
    def g(x):
        return np.sin(3.0 * x[0]) * np.cos(2.0 * x[1]) + np.exp(x[2])

    u = {}
    for name, ptcls in (("a", ptcls_a), ("b", ptcls_b)):
        ptcls.add_field("v", [1])
        ptcls.field("v").data()[:, 0] = g(ptcls.x().data().T)
        u[name] = dolfinx.fem.Function(Q)
        pyleopart.transfer_to_function(
            u[name]._cpp_object, ptcls, ptcls.field("v"))
    assert_on_all_ranks(
        mesh.comm,
        np.allclose(u["a"].x.array, u["b"].x.array, rtol=1e-9, atol=1e-11),
        "Projection depends on particle storage order")

    # Interpolating back gives each particle the same value, whatever
    # its position in storage
    for ptcls in (ptcls_a, ptcls_b):
        ptcls.add_field("vh", [1])
        pyleopart.transfer_to_particles(
            ptcls, ptcls.field("vh"), u["a"]._cpp_object)
    # Match the particles of a and b by position
    order_a = np.lexsort(ptcls_a.x().data().T)
    order_b = np.lexsort(ptcls_b.x().data().T)
    assert np.array_equal(ptcls_a.x().data()[order_a],
                          ptcls_b.x().data()[order_b])
    vh_a = ptcls_a.field("vh").data()[order_a, 0]
    vh_b = ptcls_b.field("vh").data()[order_b, 0]
    assert_on_all_ranks(
        mesh.comm, np.allclose(vh_a, vh_b, rtol=1e-12, atol=1e-14),
        "Interpolation depends on particle storage order")
