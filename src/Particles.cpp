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
  std::int32_t max_cell = 0;
  if (!cells.empty())
  {
    auto max_cell_it = std::max_element(cells.begin(), cells.end());
    max_cell = *max_cell_it;
  }
  _cell_to_particle.resize(max_cell + 1);
  for (std::size_t p = 0; p < cells.size(); ++p)
    _cell_to_particle[cells[p]].push_back(p);

  // Create position data field
  const std::size_t rows = x.size() / gdim;
  Field<T> fx(_posname, {gdim}, rows);
  std::copy(x.cbegin(), x.cend(), fx.data().begin());
  _fields.emplace(std::make_pair(_posname, std::move(fx)));
}
//------------------------------------------------------------------------
template <std::floating_point T>
std::size_t Particles<T>::add_particle(
  std::span<const T> x, std::int32_t cell)
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
  std::copy_n(x.begin(), x.size(), _fields.at(_posname).data(pidx).begin());
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
template <std::floating_point T>
void Particles<T>::relocate_bbox_on_proc(
  const dolfinx::mesh::Mesh<T>& mesh, std::span<const std::size_t> pidxs)
{
  // Resize member if required, TODO: Handle ghosts
  std::shared_ptr<const dolfinx::common::IndexMap> map =
    mesh.topology()->index_map(mesh.topology()->dim());
  const std::size_t total_cells = map->size_local();
  if (_cell_to_particle.size() < total_cells)
    _cell_to_particle.resize(total_cells);

  // Create bbox tree for mesh
  std::vector<std::int32_t> cells(total_cells);
  std::iota(cells.begin(), cells.end(), 0);
  dolfinx::geometry::BoundingBoxTree<T> tree(
    mesh, mesh.topology()->dim(), cells);

  // Get positions of required pidxs
  std::span<const T> xp_all = field(_posname).data();
  const std::size_t gdim = field(_posname).value_shape()[0];
  std::vector<T> xp(pidxs.size() * gdim, 0.0);
  for (std::size_t i = 0; i < pidxs.size(); ++i)
  {
    std::copy_n(xp_all.begin() + pidxs[i] * gdim, gdim, xp.begin() + i * gdim);
  }

  dolfinx::graph::AdjacencyList<std::int32_t> cell_candidates =
    dolfinx::geometry::compute_collisions<T>(tree, xp);
  dolfinx::graph::AdjacencyList<std::int32_t> cells_collided =
    dolfinx::geometry::compute_colliding_cells<T>(mesh, cell_candidates, xp);

  std::vector<std::size_t> lost;
  for (std::size_t l = 0; l < cells_collided.num_nodes(); ++l)
  {
    if (cells_collided.links(l).empty())
    {
      lost.push_back(pidxs[l]);
    }
    else
    {
      const std::size_t pidx = pidxs[l];
      const std::int32_t new_cell = cells_collided.links(l)[0];
      if (_particle_to_cell[pidx] == new_cell)
        continue;

      // Update old and new cells' particles
      const auto [old_cell, local_pidx] = global_to_local(pidx);
      std::vector<std::size_t>& cps = _cell_to_particle[old_cell];
      cps.erase(cps.begin() + local_pidx);
      _cell_to_particle[new_cell].push_back(pidx);
      
      // Update particle's cell
      _particle_to_cell[pidx] = new_cell;
    }
  }

  for (const std::size_t pidx : lost)
  {
    const auto [old_cell, local_pidx] = global_to_local(pidx);
    delete_particle(old_cell, local_pidx);
  }
}
//------------------------------------------------------------------------
template <std::floating_point T>
void Particles<T>::relocate_bbox(
  const dolfinx::mesh::Mesh<T>& mesh, std::span<const std::size_t> pidxs)
{
  if (dolfinx::MPI::size(mesh.comm()) == 1)
  {
    relocate_bbox_on_proc(mesh, pidxs);
    return;
  }
  const std::int32_t rank = dolfinx::MPI::rank(mesh.comm());
  const std::size_t gdim = field(_posname).value_shape()[0];

  // Resize member if required, TODO: Handle ghosts
  std::shared_ptr<const dolfinx::common::IndexMap> map =
    mesh.topology()->index_map(mesh.topology()->dim());
  const std::size_t total_cells = map->size_local();
  if (_cell_to_particle.size() < total_cells)
    _cell_to_particle.resize(total_cells);

  // Find ownership of the geometry points
  const auto [src_owner, dest_owner, dest_points, dest_cells] =
    leopart::utils::determine_point_ownership<T>(*this, mesh, pidxs);
  std::span<const T> dest_points_span(dest_points);

  // Find lost particles (outside of the geometry)
  std::vector<std::size_t> lost;

  // Mark particles located outside the domain as lost.
  // Delete the local particles which are now off process to make room
  // for (potentially) incoming particles
  std::vector<std::size_t> pidxs_on_proc;
  for (std::size_t i = 0; i < src_owner.size(); ++i)
  {
    if (src_owner[i] != rank)
    {
      if (src_owner[i] < 0)
        lost.push_back(pidxs[i]);
      const auto [cell, local_pidx] = global_to_local(pidxs[i]);
      delete_particle(cell, local_pidx);
    }
    else
      pidxs_on_proc.push_back(pidxs[i]);
  }

  // Curate particles which are still local or those coming
  // from another process
  std::size_t on_proc_offset = 0;
  for (std::size_t i = 0; i < dest_owner.size(); ++i)
  {
    const std::int32_t new_cell = dest_cells[i];
    if (dest_owner[i] == rank)
    {
      // Particle is already on process
      const std::size_t pidx = pidxs_on_proc[on_proc_offset++];
      if (new_cell == _particle_to_cell[pidx])
        continue;

      // Particle changed cell: update old and new cells' particles
      const auto [old_cell, local_pidx] = global_to_local(pidx);
      std::vector<std::size_t>& cps = _cell_to_particle[old_cell];
      cps.erase(cps.begin() + local_pidx);
      _cell_to_particle[new_cell].push_back(pidx);

      // Update particle's cell
      _particle_to_cell[pidx] = new_cell;
    }
    else
    {
      // Particle came from another process
      add_particle(dest_points_span.subspan(i * gdim, gdim), new_cell);
    }
  }
}
//------------------------------------------------------------------------
template class Particles<double>;
//------------------------------------------------------------------------