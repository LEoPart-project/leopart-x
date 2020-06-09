from dolfinx import *
import dolfinx.geometry
from mpi4py import MPI
from pyleopart import *
import numpy as np

mesh = UnitSquareMesh(MPI.COMM_WORLD, 4, 4)
ncell = mesh.topology.index_map(2).size_local
print("ncells = ", ncell)
ppc = 10
x, cells = mesh_fill(mesh, ncell*ppc)
print(len(x))
    
p = Particles(x, cells)
p.add_field("w", [2])
p.add_field("v", [2])

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

# Examine fields "w" and "x" (position)
print(p.field("w").data(0), p.field("x").data(0)**2)