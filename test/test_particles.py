
import numpy as np
import pyleopart
import pytest


def test_simple_create():
    x = np.array([[1, 2, 3]], dtype=np.float64)
    p = pyleopart.Particles(x, [0])
    cp = p.cell_particles()
    assert(cp[0][0] == 0)
    assert((p.field("x").data(0) == [1, 2, 3]).all())


def test_add_field():
    n = 20
    x = np.random.rand(n, 3)
    p = pyleopart.Particles(x, [0]*n)
    cp = p.cell_particles()
    assert(len(cp[0]) == n)
    p.add_field("w", [3])
    assert(p.field("w").value_shape == [3])
    p.add_field("u", [3, 2])   
    u = p.field("u")
    assert(u.value_shape == [3, 2])
    assert(u.value_size == 6)
    with pytest.raises(IndexError):
        r = p.field("r")


def test_add_delete_particles():
    n = 20
    x = np.random.rand(n, 3)
    p = pyleopart.Particles(x, list(range(n)))
    x = np.random.rand(3)
    
    # Add to cell 12
    p.add_particle(x, 12)
    cp = p.cell_particles()
    assert(len(cp[12]) == 2)

    # Delete from cell 1
    p.delete_particle(0, 0)
    cp = p.cell_particles()
    assert(len(cp[0]) == 0)

    # Add to cell 0
    p.add_particle(x, 1)
    cp = p.cell_particles()
    assert(cp[1][1] == 0)