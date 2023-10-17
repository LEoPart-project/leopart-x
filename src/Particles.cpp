// Copyright: (c) 2020 Chris Richardson and Jakob Maljaars
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <cassert>
#include <dolfinx.h>

#include "Particles.h"

using namespace leopart;

Particles::Particles(const std::span<double>& x,
                     const std::vector<std::int32_t>& cells,
                     const std::size_t gdim)
{
  // Find max cell index, and create cell->particle map
  auto max_cell_it = std::max_element(cells.begin(), cells.end());
  if (max_cell_it == cells.end())
    throw std::runtime_error("Error in cells data");
  const std::int32_t max_cell = *max_cell_it;
  _cell_particles.resize(max_cell + 1);
  for (std::size_t p = 0; p < cells.size(); ++p)
    _cell_particles[cells[p]].push_back(p);

  const std::size_t rows = x.size() / gdim;
  Field fx("x", {gdim}, rows);
  for (std::size_t i = 0; i < rows; ++i)
  {
    std::span<double> x_row = x.subspan(i*gdim, gdim);
    std::copy(x_row.begin(), x_row.end(), fx.data(i).begin());
  }
  _fields.push_back(fx);
}
//------------------------------------------------------------------------
std::size_t Particles::add_particle(
  const std::span<double>& x, std::int32_t cell)
{
  assert(cell < _cell_particles.size());
  assert(x.size() == _fields[0].value_shape()[0]);
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
  }

  _cell_particles[cell].push_back(pidx);
  _fields[0].data(pidx) = x;
  return pidx;
}
//------------------------------------------------------------------------
void Particles::delete_particle(std::int32_t cell, std::size_t p)
{
  assert(cell < _cell_particles.size());
  std::vector<std::size_t>& cp = _cell_particles[cell];
  assert(p < cp.size());
  std::size_t pidx = cp[p];
  cp.erase(cp.begin() + p);
  _free_list.push_back(pidx);
}
//------------------------------------------------------------------------
void Particles::add_field(
  std::string name, const std::vector<std::size_t>& shape)
{
  for (const Field& f : _fields)
    if (name == f.name)
      throw std::runtime_error("Field name \"" + name + "\" already in use");

  // Give the field the same number of entries as "x" (which must exist)
  Field f(name, shape, _fields[0].size());
  _fields.push_back(f);
}