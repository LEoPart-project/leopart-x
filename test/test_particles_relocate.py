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
                                       dolfinx.mesh.CellType.quadrilateral,
                                       dolfinx.mesh.CellType.tetrahedron,
                                       dolfinx.mesh.CellType.hexahedron])
def test_relocate_single_particle_per_process(cell_type, dtype):
    mesh = create_mesh(cell_type, dtype, 3)
    interval = np.array([0.0, 1.0])
    if mesh.geometry.dim == 2:
        xp_pts = [*[interval] * 2, [0.0]]
    else:
        xp_pts = [interval] * 3
    xp = np.array(np.meshgrid(*xp_pts), dtype=np.double).T.reshape((-1, 3))

    bad_cells = np.zeros(xp.shape[0], dtype=np.int32)
    ptcls = pyleopart.Particles(xp, bad_cells)

    ptcls.relocate_bbox(mesh._cpp_object, np.arange(xp.shape[0]))

    num_expected = 4 if mesh.geometry.dim == 2 else 8
    num_expected *= mesh.comm.size
    num_valid_p = np.sum(np.array(ptcls.particle_to_cell()) > -1)
    global_num_valid_p = mesh.comm.allreduce(num_valid_p, op=MPI.SUM)
    assert global_num_valid_p == num_expected