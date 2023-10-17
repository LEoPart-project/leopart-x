import pytest

import dolfinx
from mpi4py import MPI
import numpy as np
import pyleopart


def create_mesh(cell_type, dtype, n):
    if cell_type in (dolfinx.mesh.CellType.triangle,):
        mesh = dolfinx.mesh.create_unit_square(
            MPI.COMM_WORLD, n, n, cell_type=cell_type, dtype=dtype)
        return mesh
    elif cell_type in (dolfinx.mesh.CellType.tetrahedron,):
        mesh = dolfinx.mesh.create_unit_cube(
            MPI.COMM_WORLD, n, n, n, cell_type=cell_type, dtype=dtype)
        return mesh


@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("cell_type", [dolfinx.mesh.CellType.triangle,
                                       dolfinx.mesh.CellType.tetrahedron])
def test_simple_mesh_fill(dtype, cell_type):
    num_p = 10
    mesh = create_mesh(cell_type, dtype, n=2)
    x, c = pyleopart.mesh_fill(mesh._cpp_object, num_p)
    p = pyleopart.Particles(x, c)
    assert p.field("x").data().shape[1] == mesh.geometry.dim
    assert np.all(p.field("x").data() == x)
