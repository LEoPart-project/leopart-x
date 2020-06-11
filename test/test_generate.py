
import dolfinx
from mpi4py import MPI
import numpy as np
import pyleopart


def test_simple_mesh_fill():
    np = 20
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 5, 5)
    ncells = mesh.topology.index_map(2).size_local
    x, c = pyleopart.mesh_fill(mesh, ncells*np)
    x0 = x[0,:]
    p = pyleopart.Particles(x, c)
    assert((p.field("x").data(0)==x0).all())

