#include "Particles.h"
#include <dolfinx.h>

using namespace leopart;

Particles::Particles(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>& x,
                     const std::vector<int>& cells)
{
  // Find max cell index, and create cell->particle map
  auto max_cell_it = std::max_element(cells.begin(), cells.end());
  if (max_cell_it == cells.end())
    throw std::runtime_error("Error in cells data");
  const int max_cell = *max_cell_it;
  _cell_particles.resize(max_cell + 1);
  for (std::size_t p = 0; p < cells.size(); ++p)
    _cell_particles[cells[p]].push_back(p);

  Field fx("x", {{(int)x.cols()}}, x.rows());
  for (int i = 0; i < x.rows(); ++i)
    fx.data(i) = x.row(i);
  _fields.push_back(fx);
}

void Particles::add_field(std::string name, const std::vector<int>& shape)
{
  int np = 0;
  for (std::vector<int>& q : _cell_particles)
    np += q.size();

  for (const Field& f : _fields)
    if (name == f.name)
      throw std::runtime_error("Field name \"" + name + "\" already in use");

  Field f(name, shape, np);
  _fields.push_back(f);
}