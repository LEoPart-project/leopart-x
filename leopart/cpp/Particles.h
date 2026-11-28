// Copyright (c) 2020-2023 Chris Richardson, Jakob Maljaars and Nathan Sime
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Field.h"
#include <algorithm>
#include <concepts>
#include <cstdint>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>


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

/// Particles in the cells of a mesh, each carrying the data of a set of
/// fields.
///
/// Particles are stored sorted by cell: the particles of cell c are the
/// contiguous range of indices given by cell_offsets(), and there are no
/// deleted particles in storage. Any operation that adds, deletes or
/// relocates particles keeps this so, and therefore changes particle
/// indices and reallocates the data of every field.
template <std::floating_point T>
class Particles
{
public:
  /// Initialise particles with position, the index of the containing cell
  /// and the number of components stored per position (2 or 3).
  /// Positions are stored with gdim components per particle. Routines
  /// which interface with DOLFINx geometry pad these to three components
  /// internally.
  ///
  /// @note The particles are stored sorted by cell, so they are in the
  /// order given only if \p cells is non-decreasing. Particles with a
  /// negative cell are dropped.
  Particles(
    const std::vector<T>& x, const std::vector<std::int32_t>& cells,
    const std::size_t gdim);

  // Copy constructor
  Particles(const Particles& ptcls) = delete;

  /// Move constructor
  Particles(Particles&& ptcls) = default;

  /// Destructor
  ~Particles() = default;

  /// Add a field to the particles, with name and value shape. Its
  /// values are zero.
  void add_field(std::string name, const std::vector<std::size_t>& shape);

  /// Number of particles on this process
  std::size_t num_particles() const { return _particle_to_cell.size(); }

  /// Number of cells the particles are indexed over. At least one more
  /// than the largest cell containing a particle.
  std::size_t num_cells() const { return _cell_offsets.size() - 1; }

  /// Offsets of each cell's particles. The particles in cell c are those
  /// with indices offsets[c], ..., offsets[c + 1] - 1. Size num_cells() + 1.
  std::span<const std::size_t> cell_offsets() const { return _cell_offsets; }

  /// Cell containing each particle. Non-decreasing.
  std::span<const std::int32_t> particle_to_cell() const
  {
    return _particle_to_cell;
  }

  /// Add particles. Every field of a new particle is zero.
  ///
  /// @param x Positions, shape `(cells.size(), x().value_size())`,
  /// row-major
  /// @param cells Cell containing each new particle
  /// @return Index of each new particle, in the order given
  /// @note Changes particle indices (see the class description).
  /// New particles follow the existing particles of their cell.
  std::vector<std::size_t> add_particles(
    std::span<const T> x, std::span<const std::int32_t> cells);

  /// Delete particles
  ///
  /// @param pidxs Indices of the particles to delete
  /// @note Changes particle indices (see the class description).
  void delete_particles(std::span<const std::size_t> pidxs);

  /// Access field by name (convenience)
  /// Used in Python wrapper
  Field<T>& field(const std::string w)
  {
    return _fields.at(w);
  }

  // Const versions for internal use
  const Field<T>& field(const std::string w) const
  {
    return _fields.at(w);
  }

  bool field_exists(const std::string w) const
  {
    return _fields.contains(w);
  }

  /// Particle positions field (non-const)
  Field<T>& x() { return _x; };

  /// Particle positions (non-const)
  const Field<T>& x() const { return _x; };

  /// Gather the positions of the provided particles, padded with zeros
  /// to three components per particle as required by DOLFINx's geometry
  /// routines.
  ///
  /// @param pidxs Process local particle indices
  /// @return Positions, shape `(pidxs.size(), 3)`, row-major
  std::vector<T> positions_padded(std::span<const std::size_t> pidxs) const
  {
    const std::size_t xdim = _x.value_size();
    const std::size_t ncopy = std::min<std::size_t>(xdim, 3);
    std::span<const T> xp_all = _x.data();
    std::vector<T> xp(pidxs.size() * 3, 0.0);
    for (std::size_t i = 0; i < pidxs.size(); ++i)
      std::copy_n(xp_all.begin() + pidxs[i] * xdim, ncopy, xp.begin() + i * 3);
    return xp;
  }

  /// As relocate_bbox. Non-parallel version.
  void relocate_bbox_on_proc(
    const dolfinx::mesh::Mesh<T>& mesh,
    std::span<const std::size_t> pidxs);

  /// Using a bounding box tree, compute the colliding cells of the
  /// provided process local particles, and assign them to those cells,
  /// sending them to another process if it owns the cell. Particles
  /// outside the mesh are deleted.
  ///
  /// Afterwards the particles are indexed over the cells of \p mesh.
  ///
  /// @note Changes particle indices (see the class description).
  void relocate_bbox(
    const dolfinx::mesh::Mesh<T>& mesh,
    std::span<const std::size_t> pidxs);

  /// Relocate every particle, as relocate_bbox.
  void relocate_bbox(const dolfinx::mesh::Mesh<T>& mesh);

  /// Adapted from dolfinx::geomety::utils::determine_point_ownership
  ///
  /// @brief Globally determine cell ownerships of provided points.
  ///
  /// Ownerships is determined by finding the processes with which
  /// the provided points collide. Cell collision is then employed
  /// on the colliding processes' cells to determine ownership.
  /// If no collision is found, ownership is given to the cell which
  /// minimises the l2 distance from the cell using the GJK algorithm.
  ///
  /// @param[in] mesh The mesh
  /// @param[in] points Points to check for collision (`shape=(num_points,
  /// 3)`). Storage is row-major.
  /// @return Tuple (src_owner, dest_owner, dest_points,
  /// dest_cells, dest_data). src_owner is a list of the ranks on which
  /// each input point is initialised. dest_owner is a list of ranks found to own
  /// the input ranks. dest_points is the original point data communicated to 
  /// the dest_owner rank. dest_cells contains the corresponding cell owning each
  /// point entry in dest_points. dest_data is any corresponding data from
  /// the particles' fields.
  ///
  /// @note dest_owner is sorted
  /// @note Returns -1 if no colliding process is found
  /// @note dest_points is flattened row-major, shape (dest_owner.size(), 3)
  /// @note Only looks through cells owned by the process
  std::tuple<std::vector<std::int32_t>, std::vector<std::int32_t>, std::vector<T>,
            std::vector<std::int32_t>, std::map<std::string, std::vector<T>>>
  determine_point_ownership(
    const dolfinx::mesh::Mesh<T>& mesh,
    std::span<const std::size_t> pidxs,
    T padding=1.0e-4);

  /// @brief Generate particles such that a minimum number per mesh cell
  /// is available
  ///
  /// @note The underlying particles are assumed to be configured for
  ///       the provided mesh. Consider relocating particles prior to call.
  ///
  /// @param[in] np_per_cell Minimum number of particles per cell
  void generate_minimum_particles_per_cell(
    const dolfinx::mesh::Mesh<T>& mesh, const std::size_t np_per_cell);

private:
  /// Marks a particle for deletion by the next sort_by_cell()
  static constexpr std::int32_t INVALID_CELL = -1;

  /// Append n particles to the end of storage, with every field zero
  /// and cell zero, leaving the storage unsorted until sort_by_cell().
  /// @return Index of the first new particle
  std::size_t append(std::size_t n);

  /// Reorder the particles so that each cell's particles are a
  /// contiguous range of indices, in increasing cell order, and drop
  /// particles whose cell is INVALID_CELL. Particles in the same cell
  /// keep their relative order. Restores the storage invariant after
  /// append() or after changing particles' cells.
  ///
  /// @param num_cells Number of cells to index over. Increased if a
  /// particle is in a cell beyond it.
  /// @note Changes particle indices, and reallocates the data of every
  /// field, so previously obtained indices, spans and views are invalid.
  void sort_by_cell(std::size_t num_cells);

  // Cell containing each particle. Non-decreasing, except between
  // append() or a change of cell and the following sort_by_cell().
  std::vector<std::int32_t> _particle_to_cell;

  // The particles of cell c are [_cell_offsets[c], _cell_offsets[c + 1])
  std::vector<std::size_t> _cell_offsets;

  // Data in fields over particles
  Field<T> _x;
  std::map<std::string, Field<T>> _fields;
};
} // namespace leopart
