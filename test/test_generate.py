# Copyright (c) 2023 Nathan Sime
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

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
    return mesh_fn[cell_dim](MPI.COMM_WORLD, *[n] * cell_dim,
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


def _reference_coords_per_cell(mesh, x, c):
    cmap = mesh.geometry.cmaps[0]
    x_dofmap = mesh.geometry.dofmaps[0]
    gdim = mesh.geometry.dim
    X_per_cell = []
    for cell in np.unique(c):
        cell_geom = mesh.geometry.x[x_dofmap[cell], :gdim]
        X_per_cell.append(cmap.pull_back(x[c == cell, :gdim], cell_geom))
    return X_per_cell


@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("cell_type", [dolfinx.mesh.CellType.triangle,
                                       dolfinx.mesh.CellType.tetrahedron,
                                       dolfinx.mesh.CellType.quadrilateral,
                                       dolfinx.mesh.CellType.hexahedron])
def test_mesh_fill_seed_varies_across_cells(dtype, cell_type):
    num_p = 5
    mesh = create_mesh(cell_type, dtype, n=4)
    x, c = pyleopart.mesh_fill(mesh._cpp_object, num_p, seed=1)
    X_per_cell = _reference_coords_per_cell(mesh, x, c)
    assert len(X_per_cell) > 1
    X0 = X_per_cell[0]
    assert not all(np.allclose(X, X0) for X in X_per_cell[1:])


@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("cell_type", [dolfinx.mesh.CellType.triangle,
                                       dolfinx.mesh.CellType.hexahedron])
def test_mesh_fill_seed_reproducible(dtype, cell_type):
    num_p = 5
    mesh = create_mesh(cell_type, dtype, n=2)
    x0, c0 = pyleopart.mesh_fill(mesh._cpp_object, num_p, seed=42)
    x1, c1 = pyleopart.mesh_fill(mesh._cpp_object, num_p, seed=42)
    assert np.array_equal(x0, x1)
    assert np.array_equal(c0, c1)


def _particles_per_cell(x, c):
    return {cell: x[c == cell] for cell in np.unique(c)}


@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("cell_type", [dolfinx.mesh.CellType.triangle,
                                       dolfinx.mesh.CellType.tetrahedron,
                                       dolfinx.mesh.CellType.quadrilateral,
                                       dolfinx.mesh.CellType.hexahedron])
def test_mesh_fill_seed_partition_independent(dtype, cell_type):
    num_p = 5
    seed = 7
    cell_dim = dolfinx.mesh.cell_dim(cell_type)
    mesh_fn = {2: dolfinx.mesh.create_unit_square,
               3: dolfinx.mesh.create_unit_cube}[cell_dim]
    mesh_dist = mesh_fn(MPI.COMM_WORLD, *[3] * cell_dim,
                        cell_type=cell_type, dtype=dtype)
    mesh_serial = mesh_fn(MPI.COMM_SELF, *[3] * cell_dim,
                          cell_type=cell_type, dtype=dtype)

    x_d, c_d = pyleopart.mesh_fill(mesh_dist._cpp_object, num_p, seed=seed)
    x_s, c_s = pyleopart.mesh_fill(mesh_serial._cpp_object, num_p, seed=seed)

    oci_d = mesh_dist.topology.original_cell_index
    oci_s = mesh_serial.topology.original_cell_index
    serial_by_original = {
        oci_s[cell]: xp for cell, xp in _particles_per_cell(x_s, c_s).items()}

    for cell, xp in _particles_per_cell(x_d, c_d).items():
        assert np.allclose(xp, serial_by_original[oci_d[cell]])


@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("cell_type", [dolfinx.mesh.CellType.triangle,
                                       dolfinx.mesh.CellType.hexahedron])
def test_mesh_fill_seed_subset_matches_full(dtype, cell_type):
    num_p = 5
    seed = 3
    mesh = create_mesh(cell_type, dtype, n=3)
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    subset = np.arange(num_cells, dtype=np.int32)[::-2]

    x_full, c_full = pyleopart.mesh_fill(mesh._cpp_object, num_p, seed=seed)
    x_sub, c_sub = pyleopart.mesh_fill(
        mesh._cpp_object, [num_p] * len(subset), subset, seed=seed)

    full = _particles_per_cell(x_full, c_full)
    for cell, xp in _particles_per_cell(x_sub, c_sub).items():
        assert np.allclose(xp, full[cell])
