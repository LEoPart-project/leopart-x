
import numpy as np
import pyleopart


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
    assert(p.field("u").value_shape == [3, 2])