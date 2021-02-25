import dolfinx
from mpi4py import MPI
import numpy as np
import pyleopart

npart = 20
mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 3, 3)
ncells = mesh.topology.index_map(2).size_local
x, c = pyleopart.mesh_fill(mesh, ncells * npart)
p = pyleopart.Particles(x, c)

print(dir(pyleopart))
print(dir(p))
# advect = pyleopart.Advect(10)
advect = pyleopart.Advect(p, mesh)