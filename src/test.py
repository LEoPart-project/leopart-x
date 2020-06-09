
import pyleopart
import numpy as np

x = np.array([[0,0,0],
              [1,0,0], 
              [1,2,3]], dtype=np.float64)
c = [0, 2, 0]

p = pyleopart.Particles(x, c)

for i in range(3):
    print(p.field("x").data(i))
print(p.cell_particles()[0])

print(p.field("x").value_shape)

for i in p.cell_particles()[0]:
    print(i, p.field("x").data(i))
