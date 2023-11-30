# Copyright (c) 2023 Nathan Sime
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
import pytest

import leopart.cpp as pyleopart


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
