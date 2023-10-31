// Copyright: (c) 2020-2023 Chris Richardson, Jakob Maljaars and Nathan Sime
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <cassert>
#include <dolfinx.h>

#include "Particles.h"

using namespace leopart;

template <std::floating_point T>
Particles<T>::Particles(const std::vector<T>& x,
                        const std::vector<std::int32_t>& cells,
                        const std::size_t gdim) : _particle_to_cell(cells)
{
  // Find max cell index, and create cell->particle map
  auto max_cell_it = std::max_element(cells.begin(), cells.end());
  if (max_cell_it == cells.end())
    throw std::runtime_error("Error in cells data");
  const std::int32_t max_cell = *max_cell_it;
  _cell_to_particle.resize(max_cell + 1);
  for (std::size_t p = 0; p < cells.size(); ++p)
    _cell_to_particle[cells[p]].push_back(p);

  const std::size_t rows = x.size() / gdim;
  Field<T> fx(_posname, {gdim}, rows);
  std::copy(x.cbegin(), x.cend(), fx.data().begin());
  _fields.emplace(std::make_pair(_posname, std::move(fx)));
}
//------------------------------------------------------------------------
template <std::floating_point T>
std::size_t Particles<T>::add_particle(
  const std::span<T>& x, std::int32_t cell)
{
  assert(cell < _cell_to_particle.size());
  assert(x.size() == _fields.at(_posname).value_shape()[0]);
  std::size_t pidx;
  if (_free_list.empty())
  {
    // Need to create a new particle, and extend associated fields
    // Get new particle index from size of _posname field
    // (which must exist)
    pidx = _fields.at(_posname).size();
    // Resize all fields
    for (auto& [f_name, f] : _fields)
      f.resize(f.size() + 1);
    _particle_to_cell.resize(_particle_to_cell.size() + 1);
  }
  else
  {
    pidx = _free_list.back();
    _free_list.pop_back();
  }

  _cell_to_particle[cell].push_back(pidx);
  _particle_to_cell[pidx] = cell;
  _fields.at(_posname).data(pidx) = x;
  return pidx;
}
//------------------------------------------------------------------------
template <std::floating_point T>
void Particles<T>::delete_particle(std::int32_t cell, std::size_t p_local)
{
  // delete cell to particle entry
  assert(cell < _cell_to_particle.size());
  std::vector<std::size_t>& cp = _cell_to_particle[cell];
  assert(p_local < cp.size());
  std::size_t pidx = cp[p_local];
  cp.erase(cp.begin() + p_local);

  assert(pidx < _particle_to_cell.size());
  _particle_to_cell[pidx] = INVALID_CELL;

  _free_list.push_back(pidx);
}
//------------------------------------------------------------------------
template <std::floating_point T>
void Particles<T>::add_field(
  std::string name, const std::vector<std::size_t>& shape)
{
  if (_fields.find(name) != _fields.end())
    throw std::runtime_error("Field name \"" + name + "\" already in use");

  // Give the field the same number of entries as _posname
  // (which must exist)
  Field<T> f(name, shape, _fields.at(_posname).size());
  _fields.emplace(std::make_pair(name, std::move(f)));
}
//------------------------------------------------------------------------
template class Particles<double>;
//------------------------------------------------------------------------