from dolfinx import *
import dolfinx.geometry
from mpi4py import MPI
from pyleopart import *
import numpy as np

x = np.random.rand(500,3)
x[:,2] = 0

mesh = UnitSquareMesh(MPI.COMM_WORLD, 4, 4)
tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.topology.dim)
cells = []
for p in x:
    cell = dolfinx.geometry.compute_colliding_cells(tree, mesh, p)[0]
    cells.append(cell)
    
p = particles(x, cells)
p.add_field("w", [2]);
p.add_field("v", [2]);

print(p)

Q = VectorFunctionSpace(mesh, ("DG", 2))

arr = get_particle_contributions(p, Q._cpp_object)
print (arr[0])

def sq_val(x):
    return [x[0]**2, x[1]**2]

u = Function(Q)
u.interpolate(sq_val)

# Transfer from Function "u" to field "v"
transfer_to_particles(p, u._cpp_object, 2, arr)

# Transfer from field "v" back to Function "u"
transfer_to_function(u._cpp_object, p, 2, arr)

# Transfer from Function "u" to field "w"
transfer_to_particles(p, u._cpp_object, 1, arr)

# Examine fields "v" and "x" (position)
print(p.data(0, 1), p.data(0, 0)**2);