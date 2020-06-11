// Copyright: (c) 2020 Chris Richardson and Jakob Maljaars
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <Eigen/Dense>
#include <vector>

#pragma once

namespace leopart
{

class Field
{
public:
  /// Constructor
  Field(std::string name_desc, const std::vector<int>& shape, int n);

  /// Get the data for a given particle p (const)
  Eigen::Map<const Eigen::VectorXd> data(int p) const;

  /// Get the data for a given particle p (non-const)
  Eigen::Map<Eigen::VectorXd> data(int p);

  /// Value shape
  const std::vector<int>& value_shape() const;

  /// Value size = product(value_shape). This is a convenience function, giving
  /// the cached value_size.
  int value_size() const;

  /// Total size of data - should be number of particles or more
  /// if some have been deleted (this will leave some unindexed, invalid
  /// entries).
  int size() const;

  /// Resize. Increase storage for data for new particles.
  void resize(int n);

  /// Text name
  std::string name;

private:
  // Shape
  std::vector<int> _value_shape;
  int _value_size;

  // Storage, using vector because it is easier to resize.
  std::vector<double> _data;
};
} // namespace leopart