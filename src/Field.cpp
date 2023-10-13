// Copyright: (c) 2020 Chris Richardson and Jakob Maljaars
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Field.h"

using namespace leopart;

Field::Field(std::string name_desc, const std::vector<std::size_t>& value_shape, std::size_t n)
    : name(name_desc), _value_shape(value_shape)
{
  _value_size = 1;
  for (std::size_t q : value_shape)
    _value_size *= q;
  _data.resize(_value_size * n);
}

/// Get the data for a given particle p (const)
std::span<const double> Field::data(std::size_t p) const
{
  return std::span<const double>(_data).subspan(_value_size * p, _value_size);
}

/// Get the data for a given particle p (non-const)
std::span<double> Field::data(std::size_t p)
{
  return std::span<double>(_data).subspan(_value_size * p, _value_size);
}

/// Value shape
const std::vector<std::size_t>& Field::value_shape() const { return _value_shape; }

std::size_t Field::value_size() const { return _value_size; }

/// Total size of data - should be number of particles
std::size_t Field::size() const { return _data.size() / _value_size; };

void Field::resize(std::size_t n) { _data.resize(n * _value_size); }
