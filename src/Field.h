#include <Eigen/Dense>
#include <vector>

#pragma once

namespace leopart
{

class Field
{
public:
  Field(std::string name_desc, const std::vector<int>& shape, int n)
      : name(name_desc), _shape(shape)
  {
    _value_size = 1;
    for (int q : shape)
      _value_size *= q;
    _data.resize(_value_size * n);
  }

  /// Get the data for a given particle p (const)
  Eigen::Map<const Eigen::VectorXd> data(int p) const
  {
    return Eigen::Map<const Eigen::VectorXd>(_data.data() + _value_size * p,
                                             _value_size);
  }

  /// Get the data for a given particle p (non-const)
  Eigen::Map<Eigen::VectorXd> data(int p)
  {
    return Eigen::Map<Eigen::VectorXd>(_data.data() + _value_size * p,
                                       _value_size);
  }

  /// Value shape
  const std::vector<int>& shape() const { return _shape; }

  /// Total size of data - should be number of particles
  int size() const { return _data.size() / _value_size; };

  void resize(int n) { _data.resize(n * _value_size); }

  /// Text name
  std::string name;

private:
  // Shape
  std::vector<int> _shape;
  int _value_size;

  // Storage, using vector because it is easier to resize.
  std::vector<double> _data;
};
} // namespace leopart