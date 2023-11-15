// Copyright: (c) 2020-2023 Chris Richardson, Jakob Maljaars and Nathan Sime
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Field.h"
#include <vector>
#include <stdexcept>
#include <map>

namespace dolfinx
{
namespace mesh
{
  template <std::floating_point T>
  class Mesh;
}
}

namespace leopart
{

template <std::floating_point T>
class Particles
{
public:
  /// Initialise particles with position, the index of the containing cell
  /// and geometric dimension of the data.
  Particles(
    const std::vector<T>& x, const std::vector<std::int32_t>& cells,
    const std::size_t gdim);

  // Copy constructor
  Particles(const Particles& ptcls) = delete;

  /// Move constructor
  Particles(Particles&& ptcls) = default;

  /// Destructor
  ~Particles() = default;

  /// Add a field to the particles, with name and value shape
  void add_field(std::string name, const std::vector<std::size_t>& shape);

  /// List of particles in each cell
  const std::vector<std::vector<std::size_t>>& cell_to_particle() const
  {
    return _cell_to_particle;
  }

  /// List of unique cell assigned to enclose each particle
  const std::vector<std::int32_t>& particle_to_cell() const
  {
    return _particle_to_cell;
  }

  /// Add a particle to a cell
  /// @return New particle index
  std::size_t add_particle(std::span<const T> x, std::int32_t cell);

  /// Delete particle p in cell
  /// @note \p p is cell-local index
  void delete_particle(std::int32_t cell, std::size_t p_local);

  /// Access field by name (convenience)
  /// Used in Python wrapper
  Field<T>& field(std::string w)
  {
    return _fields.at(w);
  }

  // Const versions for internal use
  const Field<T>& field(std::string w) const
  {
    return _fields.at(w);
  }

  /// Generate process local indices of valid particles, i.e., those
  /// which have not been allocated as free for assignment with new
  /// data.
  inline std::vector<std::size_t> active_pidxs()
  {
    const std::size_t num_pidxs = _particle_to_cell.size();
    std::vector<std::size_t> valid_pidxs(num_pidxs);
    std::size_t n_valid_particles = 0;
    for (std::size_t pidx = 0; pidx < num_pidxs; ++pidx)
    {
      if (_particle_to_cell[pidx] != INVALID_CELL)
        valid_pidxs[n_valid_particles++] = pidx;
    }
    valid_pidxs.resize(n_valid_particles);
    return valid_pidxs;
  }

  /// Given a process local particle index, return the owning cell and cell
  /// local index.
  inline std::pair<const std::int32_t, const std::size_t> global_to_local(
    std::size_t p_global) const
  {
    const std::int32_t p_cell = _particle_to_cell[p_global];
    std::span<const std::size_t> cell_ps = _cell_to_particle[p_cell];
    const std::size_t p_local = std::distance(cell_ps.begin(),
      std::find(cell_ps.begin(), cell_ps.end(), p_global));
    return {p_cell, p_local};
  }

  /// As relocate_bbox. Non-parallel version.
  void relocate_bbox_on_proc(
    const dolfinx::mesh::Mesh<T>& mesh,
    std::span<const std::size_t> pidxs);

  /// Using a bounding box tree, compute the colliding cells of the provided
  /// global particle indices. These particles are then "relocated" to these
  /// cells.
  ///
  /// @note Does not check for deleted or invalid particles
  void relocate_bbox(
    const dolfinx::mesh::Mesh<T>& mesh,
    std::span<const std::size_t> pidxs);

private:
  // Process local indices of particles in each cell.
  std::vector<std::vector<std::size_t>> _cell_to_particle;

  // Local incides of cells to which particles belong
  std::vector<std::int32_t> _particle_to_cell;

  // List of process local particles which have been deleted and available
  // for reallocation
  std::vector<std::size_t> _free_list;

  // Data in fields over particles
  std::map<std::string, Field<T>> _fields;
  const std::string _posname = "x";
  const std::int32_t INVALID_CELL = -1;
};
} // namespace leopart