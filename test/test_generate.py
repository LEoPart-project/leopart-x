from mpi4py import MPI

import numpy as np
import pytest

import dolfinx
import leopart.cpp as pyleopart


def create_mesh(cell_type, dtype, n):
    mesh_fn = {
        2: dolfinx.mesh.create_unit_square,
        3: dolfinx.mesh.create_unit_cube
    }

    cell_dim = dolfinx.mesh.cell_dim(cell_type)
    return mesh_fn[cell_dim](MPI.COMM_WORLD, *[n]*cell_dim,
                             cell_type=cell_type, dtype=dtype)


@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("cell_type", [dolfinx.mesh.CellType.triangle,
                                       dolfinx.mesh.CellType.tetrahedron,
                                       dolfinx.mesh.CellType.quadrilateral,
                                       dolfinx.mesh.CellType.hexahedron])
def test_simple_mesh_fill(dtype, cell_type):
    num_p = 10
    mesh = create_mesh(cell_type, dtype, n=2)
    x, c = pyleopart.mesh_fill(mesh._cpp_object, num_p)
    p = pyleopart.Particles(x, c)
    assert p.x().data().shape[1] == mesh.geometry.dim
    assert np.all(p.x().data() == x)
