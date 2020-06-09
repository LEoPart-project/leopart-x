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

  /// Get the data for a given field (index idx) on particle p (non-const)
  Eigen::Map<Eigen::VectorXd> data(int p, int idx)
  {
    return _fields[idx].data(p);
  }

  /// Get the data for a given field (index idx) on particle p (const)
  Eigen::Map<const Eigen::VectorXd> data(int p, int idx) const
  {
    return _fields[idx].data(p);
  }

  const std::vector<std::vector<int>>& cell_particles() const
  {
    return _cell_particles;
  }

  const Field& field(int i) const { return _fields[i]; }

private:
  // Indices of particles in each cell.
  std::vector<std::vector<int>> _cell_particles;

  // Data in fields over particles
  std::vector<Field> _fields;
};
} // namespace leopart