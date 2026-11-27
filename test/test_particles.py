# Copyright (c) 2023 Nathan Sime
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

import dolfinx
import leopart.cpp as pyleopart


def check_sorted_and_compact(p):
    """Particles are stored in increasing cell order with no deleted
    slots, and each cell's particles are a contiguous index range."""
    p2c = np.asarray(p.particle_to_cell())
    assert np.all(p2c >= 0)
    assert np.all(np.diff(p2c) >= 0)
    offset = 0
    for c, ps in enumerate(p.cell_to_particle()):
        n = len(ps)
        assert np.array_equal(ps, np.arange(offset, offset + n))
        assert np.all(p2c[offset:offset + n] == c)
        offset += n
    assert offset == len(p2c)


def test_empty_create():
    x = np.array([], dtype=np.float64)
    p = pyleopart.Particles(x, [])
    assert len(p.cell_to_particle()) == 1
    assert len(p.cell_to_particle()[0]) == 0
    assert len(p.particle_to_cell()) == 0


def test_simple_create():
    x = np.array([[1, 2, 3]], dtype=np.float64)
    p = pyleopart.Particles(x, [0])
    c2p = p.cell_to_particle()
    assert c2p[0][0] == 0
    assert np.all(p.x().data(0) == [1, 2, 3])


def test_add_field():
    n = 20
    x = np.random.default_rng().random((n, 3))
    p = pyleopart.Particles(x, [0] * n)
    c2p = p.cell_to_particle()
    assert len(c2p[0]) == n
    p.add_field("w", [3])
    assert p.field("w").value_shape == [3]
    p.add_field("u", [3, 2])
    u = p.field("u")
    assert u.value_shape == [3, 2]
    assert u.value_size == 6
    with pytest.raises(IndexError):
        p.field("r")


def test_add_delete_particles():
    n = 20
    dim = 3
    x = np.arange(n * dim, dtype=np.float64).reshape((-1, dim))
    x2c = np.arange(x.shape[0], dtype=np.int32)
    p = pyleopart.Particles(x, x2c)

    # Add to cell 12
    x = np.random.default_rng().random(3)
    new_pidx = p.add_particle(x, 12)
    assert len(p.particle_to_cell()) == n + 1
    assert new_pidx == n
    c2p = p.cell_to_particle()
    assert len(c2p[12]) == 2
    assert np.all(c2p[12] == np.array([12, new_pidx]))

    # Delete from cell 0
    p.delete_particle(0, 0)
    c2p = p.cell_to_particle()
    assert len(p.particle_to_cell()) == n + 1
    assert len(c2p[0]) == 0

    # Add to cell 1
    new_pidx = p.add_particle(x, 1)
    assert len(p.particle_to_cell()) == n + 1
    assert new_pidx == 0
    c2p = p.cell_to_particle()
    assert c2p[1][1] == 0
    assert np.all(c2p[1] == np.array([1, new_pidx]))


def test_sort_by_cell():
    cells = np.array([2, 0, 1, 0, 2, 1], dtype=np.int32)
    x = np.arange(cells.size * 3, dtype=np.float64).reshape((-1, 3))
    p = pyleopart.Particles(x, cells)
    p.add_field("tag", [1])
    p.field("tag").data()[:, 0] = np.arange(cells.size)

    # Leave a hole: delete particle 3, the second particle in cell 0
    p.delete_particle(0, 1)
    p.sort_by_cell()
    check_sorted_and_compact(p)

    # Stable: within a cell, particles keep their previous order.
    # Positions moved with their particles.
    tag = p.field("tag").data()[:, 0].astype(int)
    assert np.array_equal(tag, [1, 2, 5, 0, 4])
    assert np.array_equal(p.x().data(), x[tag])

    # A particle added after sorting reuses no slot, and sorting again
    # moves it into its cell's range
    p.add_particle(np.zeros(3), 0)
    p.sort_by_cell()
    check_sorted_and_compact(p)
    assert np.array_equal(p.field("tag").data()[:, 0], [1, 0, 2, 5, 0, 4])


@pytest.mark.parametrize("cell_type", [dolfinx.mesh.CellType.triangle,
                                       dolfinx.mesh.CellType.tetrahedron])
def test_relocation_sorts_by_cell(cell_type):
    comm = MPI.COMM_WORLD
    if cell_type == dolfinx.mesh.CellType.triangle:
        mesh = dolfinx.mesh.create_unit_square(comm, 6, 6, cell_type)
    else:
        mesh = dolfinx.mesh.create_unit_cube(comm, 3, 3, 3, cell_type)

    x, cells = pyleopart.mesh_fill(mesh._cpp_object, 3, seed=1)
    x = np.c_[x, np.zeros((x.shape[0], 3 - x.shape[1]))]

    # Store particles in reverse cell order, so storage starts unsorted,
    # and tag each with its position to check its data follows it
    x = np.ascontiguousarray(x[::-1])
    cells = np.asarray(cells, dtype=np.int32)[::-1]
    p = pyleopart.Particles(x, cells)
    p.add_field("tag", [3])
    p.field("tag").data()[:] = x

    # Leave a hole
    p.delete_particle(int(cells[0]), 0)
    n_global = comm.allreduce(x.shape[0] - 1, op=MPI.SUM)

    p.relocate_bbox(mesh._cpp_object, p.active_pidxs())

    check_sorted_and_compact(p)
    assert comm.allreduce(len(p.particle_to_cell()), op=MPI.SUM) == n_global
    assert np.array_equal(p.field("tag").data(), p.x().data())
