// Copyright: (c) 2020 Chris Richardson and Jakob Maljaars
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Field.h"

using namespace leopart;

Field::Field(std::string name_desc, const std::vector<int>& value_shape, int n)
    : name(name_desc), _value_shape(value_shape)
{
  _value_size = 1;
  for (int q : value_shape)
    _value_size *= q;
  _data.resize(_value_size * n);
}

/// Get the data for a given particle p (const)
Eigen::Map<const Eigen::VectorXd> Field::data(int p) const
{
  return Eigen::Map<const Eigen::VectorXd>(_data.data() + _value_size * p,
                                           _value_size);
}

/// Get the data for a given particle p (non-const)
Eigen::Map<Eigen::VectorXd> Field::data(int p)
{
  return Eigen::Map<Eigen::VectorXd>(_data.data() + _value_size * p,
                                     _value_size);
}

/// Value shape
const std::vector<int>& Field::value_shape() const { return _value_shape; }

int Field::value_size() const { return _value_size; }

/// Total size of data - should be number of particles
int Field::size() const { return _data.size() / _value_size; };

void Field::resize(int n) { _data.resize(n * _value_size); }
