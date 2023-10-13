// Copyright: (c) 2020 Chris Richardson and Jakob Maljaars
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <vector>
#include <string>
#include <span>

#pragma once

namespace leopart
{

class Field
{
public:
  /// Constructor
  Field(std::string name_desc, const std::vector<std::size_t>& shape, std::size_t n);

  /// Get the data for a given particle p (const)
  std::span<const double> data(std::size_t p) const;

  /// Get the data for a given particle p (non-const)
  std::span<double> data(std::size_t p);

  /// Get the associated field data
  std::span<double> data() { return std::span<double>(_data); };

  /// Value shape
  const std::vector<std::size_t>& value_shape() const;

  /// Value size = product(value_shape). This is a convenience function, giving
  /// the cached value_size.
  std::size_t value_size() const;

  /// Total size of data - should be number of particles or more
  /// if some have been deleted (this will leave some unindexed, invalid
  /// entries).
  std::size_t size() const;

  /// Resize. Increase storage for data for new particles.
  void resize(std::size_t n);

  /// Text name
  std::string name;

private:
  // Shape
  std::vector<std::size_t> _value_shape;
  std::size_t _value_size;

  // Storage, using vector because it is easier to resize.
  std::vector<double> _data;
};
} // namespace leopart