#include "Field.h"
#include <vector>
#include <stdexcept>

#pragma once

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

  /// Add a field to the particles, with name and value shape
  void add_field(std::string name, const std::vector<std::size_t>& shape);

  /// List of particles in each cell
  const std::vector<std::vector<std::size_t>>& cell_particles() const
  {
    return _cell_particles;
  }

  /// Add a particle to a cell
  /// @return New particle index
  std::size_t add_particle(const std::span<T>& x, std::int32_t cell);

  /// Delete particle p in cell
  /// @note \p p is cell-local index
  void delete_particle(std::int32_t cell, std::size_t p);

  /// Field access (const)
  const Field<T>& field(int i) const { return _fields[i]; }

  /// Field access (non-const)
  Field<T>& field(int i) { return _fields[i]; }

  /// Access field by name (convenience)
  /// Used in Python wrapper
  Field<T>& field(std::string w)
  {
    for (Field<T>& f : _fields)
    {
      if (f.name == w)
        return f;
    }
    throw std::out_of_range("Field not found");
  }

  // Const versions for internal use
  const Field<T>& field(std::string w) const
  {
    for (const Field<T>& f : _fields)
    {
      if (f.name == w)
        return f;
    }
    throw std::out_of_range("Field not found");
  }

// private:
  // Indices of particles in each cell.
  std::vector<std::vector<std::size_t>> _cell_particles;

  // List of particles which have been deleted, and available for reallocation
  std::vector<std::size_t> _free_list;

  // Data in fields over particles
  std::vector<Field<T>> _fields;
};
} // namespace leopart