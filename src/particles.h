#include <Eigen/Dense>
#include <vector>

#pragma once

namespace leopart
{

class particles
{
public:
  /// Initialise particles with position and the index of the cell which
  /// contains them.
  particles(const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x,
            const std::vector<int>& cells);

  /// Get the data for a given field (index idx) on particle p (non-const)
  Eigen::Map<Eigen::VectorXd> data(int p, int idx)
  {
    int size = _field_shape[idx][0];
    if (_field_shape[idx].size() > 1)
      size *= _field_shape[idx][1];
    return Eigen::Map<Eigen::VectorXd>(_field_data[idx].data() + size * p,
                                       size);
  }

  /// Get the data for a given field (index idx) on particle p (const)
  Eigen::Map<const Eigen::VectorXd> data(int p, int idx) const
  {
    int size = _field_shape[idx][0];
    if (_field_shape[idx].size() > 1)
      size *= _field_shape[idx][1];
    return Eigen::Map<const Eigen::VectorXd>(_field_data[idx].data() + size * p,
                                             size);
  }

  const std::vector<int>& cell_particles(int c) const
  {
    return _cell_particles[c];
  }

private:
  // Indices of particles in each cell.
  std::vector<std::vector<int>> _cell_particles;

  // Human readable names for each particle field, as described below.
  std::vector<std::string> _field_name;

  // Value shape (e.g. scalar=[1], vector=[3], tensor=[3, 3]) for each field
  std::vector<std::vector<int>> _field_shape;

  // List of arrays holding particle field data, one vector for each field.
  // Each vector should be the same length as the number of particles on this
  // process.
  std::vector<std::vector<double>> _field_data;
};
} // namespace leopart