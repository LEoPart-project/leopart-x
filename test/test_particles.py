# Copyright (c) 2023 Nathan Sime
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

import gc

from mpi4py import MPI

import numpy as np
import pytest

import dolfinx
import leopart.cpp as pyleopart


def check_sorted_and_compact(p):
    """Particles are stored in increasing cell order with no deleted
    slots, and each cell's particles are the index range given by the
    cell offsets."""
    p2c = p.particle_to_cell()
    offsets = p.cell_offsets()
    assert offsets[0] == 0
    assert offsets[-1] == len(p2c) == p.num_particles()
    assert len(offsets) == p.num_cells() + 1
    assert np.all(np.diff(offsets) >= 0)
    cells = np.arange(p.num_cells(), dtype=np.int32)
    assert np.array_equal(p2c, np.repeat(cells, np.diff(offsets)))


def tagged_particles(cells):
    """Particles in the given cells, with positions and a field "tag"
    recording the order in which they were given."""
    cells = np.asarray(cells, dtype=np.int32)
    x = np.arange(cells.size * 3, dtype=np.float64).reshape((-1, 3))
    p = pyleopart.Particles(x, cells)
    p.add_field("tag", [1])
    p.field("tag").data()[:, 0] = p.x().data()[:, 0] / 3
    return p


def tags(p):
    return p.field("tag").data()[:, 0].astype(int)


def test_empty_create():
    x = np.array([], dtype=np.float64)
    p = pyleopart.Particles(x, [])
    assert p.num_particles() == 0
    assert p.num_cells() == 0
    assert np.array_equal(p.cell_offsets(), [0])
    assert len(p.particle_to_cell()) == 0


def test_simple_create():
    x = np.array([[1, 2, 3]], dtype=np.float64)
    p = pyleopart.Particles(x, [0])
    assert np.array_equal(p.cell_offsets(), [0, 1])
    assert np.all(p.x().data(0) == [1, 2, 3])


def test_create_sorts_by_cell():
    # Stable: particles in the same cell keep the order given. Particles
    # with a negative cell are dropped.
    cells = np.array([2, 0, 1, -1, 0, 2, 1], dtype=np.int32)
    x = np.arange(cells.size * 3, dtype=np.float64).reshape((-1, 3))
    p = pyleopart.Particles(x, cells)
    check_sorted_and_compact(p)
    assert np.array_equal(p.cell_offsets(), [0, 2, 4, 6])
    assert np.array_equal(p.x().data(), x[[1, 4, 2, 6, 0, 5]])


def test_add_field():
    n = 20
    x = np.random.default_rng().random((n, 3))
    p = pyleopart.Particles(x, [0] * n)
    assert np.array_equal(p.cell_offsets(), [0, n])
    p.add_field("w", [3])
    assert p.field("w").value_shape == [3]
    assert not np.any(p.field("w").data())
    p.add_field("u", [3, 2])
    u = p.field("u")
    assert u.value_shape == [3, 2]
    assert u.value_size == 6
    with pytest.raises(IndexError):
        p.field("r")


def test_add_particles():
    # One particle in each of cells 0, ..., 19
    n = 20
    p = tagged_particles(np.arange(n))
    p.field("tag").data()[:, 0] += 1

    x_new = np.random.default_rng().random((3, 3))
    new = p.add_particles(x_new, [12, 3, 12])
    check_sorted_and_compact(p)
    assert p.num_particles() == n + 3

    # New particles follow the existing particles of their cell, in the
    # order given, and are returned in the order given
    assert new.dtype == np.int64
    assert np.array_equal(new, [14, 4, 15])
    assert np.array_equal(p.x().data()[new], x_new)
    assert np.array_equal(p.particle_to_cell()[new], [12, 3, 12])

    # New particles' fields are zero, and existing particles keep theirs
    assert np.array_equal(tags(p)[new], [0, 0, 0])
    old = np.setdiff1d(np.arange(n + 3), new)
    assert np.array_equal(tags(p)[old], np.arange(1, n + 1))

    # Adding to a cell beyond the last extends the cell offsets
    new = p.add_particles(np.zeros((1, 3)), [n + 4])
    check_sorted_and_compact(p)
    assert p.num_cells() == n + 5
    assert np.array_equal(new, [n + 3])

    with pytest.raises(ValueError, match="non-negative"):
        p.add_particles(np.zeros((1, 3)), [-1])
    with pytest.raises(ValueError, match="shape"):
        p.add_particles(np.zeros((2, 3)), [0])
    check_sorted_and_compact(p)
    assert p.num_particles() == n + 4


def test_delete_particles():
    p = tagged_particles([2, 0, 1, 0, 2, 1])
    assert np.array_equal(tags(p), [1, 3, 2, 5, 0, 4])

    # Delete the second particle of cell 0 and the first of cell 2.
    # The rest keep their order and data.
    p.delete_particles([1, 4])
    check_sorted_and_compact(p)
    assert np.array_equal(p.cell_offsets(), [0, 1, 3, 4])
    assert np.array_equal(tags(p), [1, 2, 5, 4])
    assert np.array_equal(p.x().data()[:, 0], 3 * tags(p))

    with pytest.raises(IndexError):
        p.delete_particles([4])

    # Emptying a cell keeps it, with no particles
    p.delete_particles([0])
    check_sorted_and_compact(p)
    assert np.array_equal(p.cell_offsets(), [0, 0, 2, 3])


def test_views_share_storage():
    p = tagged_particles([1, 0, 1])

    # Field data and positions are writable views of the particles' data
    p.field("tag").data()[:] = 7.0
    assert np.array_equal(p.field("tag").data(), np.full((3, 1), 7.0))
    p.x().data()[0] = -1.0
    assert np.array_equal(p.x().data(0), np.full(3, -1.0))

    # The cell maps are read-only views
    offsets = p.cell_offsets()
    p2c = p.particle_to_cell()
    assert offsets.dtype == np.int64
    assert p2c.dtype == np.int32
    assert np.shares_memory(offsets, p.cell_offsets())
    assert np.shares_memory(p2c, p.particle_to_cell())
    with pytest.raises(ValueError, match="read-only"):
        offsets[0] = 1
    with pytest.raises(ValueError, match="read-only"):
        p2c[0] = 1

    # A view keeps the particles alive
    del p
    gc.collect()
    assert np.array_equal(offsets, [0, 1, 3])
    assert np.array_equal(p2c, [0, 1, 1])


@pytest.mark.parametrize("cell_type", [dolfinx.mesh.CellType.triangle,
                                       dolfinx.mesh.CellType.tetrahedron])
def test_relocation_sorts_by_cell(cell_type):
    comm = MPI.COMM_WORLD
    if cell_type == dolfinx.mesh.CellType.triangle:
        mesh = dolfinx.mesh.create_unit_square(comm, 6, 6, cell_type)
    else:
        mesh = dolfinx.mesh.create_unit_cube(comm, 3, 3, 3, cell_type)

    x, _ = pyleopart.mesh_fill(mesh._cpp_object, 3, seed=1)
    x = np.c_[x, np.zeros((x.shape[0], 3 - x.shape[1]))]

    # Store the particles in reverse order, all in cell 0, so relocation
    # moves nearly all of them, and tag each with its position to check
    # its data follows it
    x = np.ascontiguousarray(x[::-1])
    p = pyleopart.Particles(x, np.zeros(x.shape[0], dtype=np.int32))
    p.add_field("tag", [3])
    p.field("tag").data()[:] = p.x().data()

    # Delete one
    if p.num_particles() > 0:
        p.delete_particles([0])
    n_global = comm.allreduce(p.num_particles(), op=MPI.SUM)

    p.relocate_bbox(mesh._cpp_object)

    check_sorted_and_compact(p)
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    assert p.num_cells() == num_cells
    assert comm.allreduce(p.num_particles(), op=MPI.SUM) == n_global
    assert np.array_equal(p.field("tag").data(), p.x().data())
