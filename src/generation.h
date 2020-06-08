
#include <Eigen/Dense>

#pragma once

namespace leopart
{
namespace generation
{
/// Create a set of n points at random positions within the reference triangle
Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>
random_reference_tetrahedron(int n);

/// Create a set of n points at random positions within the reference
/// tetrahedron
Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>
random_reference_triangle(int n);
} // namespace generation
}; // namespace leopart