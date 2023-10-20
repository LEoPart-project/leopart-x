#include "Field.h"
#include <vector>
#include <stdexcept>
#include <map>

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
  std::size_t add_particle(const std::span<T>& x, std::int32_t cell);

  /// Delete particle p in cell
  /// @note \p p is cell-local index
  void delete_particle(std::int32_t cell, std::size_t p);

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

private:
  // Indices of particles in each cell.
  std::vector<std::vector<std::size_t>> _cell_to_particle;

  // Incides of cells to which particles belong
  std::vector<std::int32_t> _particle_to_cell;

  // List of particles which have been deleted, and available for reallocation
  std::vector<std::size_t> _free_list;

  // Data in fields over particles
  std::map<std::string, Field<T>> _fields;
  const std::string _posname = "x";
};
} // namespace leopart