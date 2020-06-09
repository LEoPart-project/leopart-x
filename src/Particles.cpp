#include "Particles.h"
#include <cassert>
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

int Particles::add_particle(const Eigen::VectorXd& x, int cell)
{
  assert(x.size() == _fields[0].shape()[0]);
  int pidx;
  if (_free_list.empty())
  {
    // Need to create a new particle, and extend associated fields
    // Get new particle index from size of "x" field (which must exist)
    pidx = _fields[0].size();
    // Resize all fields
    for (Field& f : _fields)
      f.resize(f.size() + 1);
  }
  else
  {
    pidx = _free_list.back();
    _free_list.pop_back();
    _cell_particles[cell].push_back(pidx);
  }
  // Set "x"
  _fields[0].data(pidx) = x;
  return pidx;
}

void Particles::delete_particle(int cell, int p)
{
  assert(cell < _cell_particles.size());
  std::vector<int>& cp = _cell_particles[cell];
  assert(p < cp.size());
  int pidx = cp[p];
  cp.erase(cp.begin() + p);
  _free_list.push_back(pidx);
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