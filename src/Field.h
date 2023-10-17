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

template <class T>
concept field_dtype
  = std::is_floating_point_v<T> || std::is_integral_v<T>;

template <field_dtype T>
class Field
{
public:
  /// Constructor
  Field(std::string name_desc, const std::vector<std::size_t>& value_shape, std::size_t n) :
    name(name_desc), _value_shape(value_shape)
  {
    _value_size = 1;
    for (std::size_t q : value_shape)
      _value_size *= q;
    _data.resize(_value_size * n);
  }

  // Copy constructor
  Field(const Field& field) = delete;

  /// Move constructor
  Field(Field&& field) = default;

  /// Destructor
  ~Field() = default;

  /// Get the data for a given particle p (const)
  std::span<const T> data(std::size_t p) const
  {
    return std::span<const double>(_data).subspan(_value_size * p, _value_size);
  }

  /// Get the data for a given particle p (non-const)
  std::span<T> data(std::size_t p)
  {
    return std::span<double>(_data).subspan(_value_size * p, _value_size);
  }

  /// Get the associated field data
  std::span<T> data() { return std::span<T>(_data); };

  /// Value shape
  const std::vector<std::size_t>& value_shape() const { return _value_shape; };

  /// Value size = product(value_shape). This is a convenience function, giving
  /// the cached value_size.
  std::size_t value_size() const { return _value_size; };

  /// Total size of data - should be number of particles or more
  /// if some have been deleted (this will leave some unindexed, invalid
  /// entries).
  std::size_t size() const { return _data.size() / _value_size; };

  /// Resize. Increase storage for data for new particles.
  void resize(std::size_t n)  { _data.resize(n * _value_size); };

  /// Text name
  std::string name;

private:
  // Shape
  std::vector<std::size_t> _value_shape;
  std::size_t _value_size;

  // Storage, using vector because it is easier to resize.
  std::vector<T> _data;
};
} // namespace leopart