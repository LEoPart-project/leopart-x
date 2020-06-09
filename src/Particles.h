#include "Field.h"
#include <Eigen/Dense>
#include <vector>

#pragma once

namespace leopart
{

class Particles
{
public:
  /// Initialise particles with position and the index of the cell which
  /// contains them.
  Particles(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>& x,
            const std::vector<int>& cells);

  /// Add a field to the particles, with name and value shape
  void add_field(std::string name, const std::vector<int>& shape);

  /// List of particles in each cell
  const std::vector<std::vector<int>>& cell_particles() const
  {
    return _cell_particles;
  }

  /// Field access (const)
  const Field& field(int i) const { return _fields[i]; }

  /// Field access (non-const)
  Field& field(int i) { return _fields[i]; }

  /// Access field by name (convenience)
  /// Used in Python wrapper
  Field& field(std::string w)
  {
    for (Field& f : _fields)
    {
      if (f.name == w)
        return f;
    }
    throw std::runtime_error("Field not found");
  }

private:
  // Indices of particles in each cell.
  std::vector<std::vector<int>> _cell_particles;

  // Data in fields over particles
  std::vector<Field> _fields;
};
} // namespace leopart