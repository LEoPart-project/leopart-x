
import pyleopart
import numpy as np

x = np.array([[0,0,0],
              [1,0,0], 
              [1,2,3]], dtype=np.float64)
c = [0, 2, 0]

p = pyleopart.particles(x, c)

print(p.data(1, 0))
print(p.cell_particles(0))

for i in p.cell_particles(0):
    print(i, p.data(i, 0))