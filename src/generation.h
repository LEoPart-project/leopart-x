// Copyright: (c) 2020 Chris Richardson and Jakob Maljaars
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <vector>

namespace dolfinx
{
  namespace mesh
  {
    template<std::floating_point T> class Mesh;
    enum class CellType;
  } // namespace mesh
} // namespace dolfinx

namespace leopart
{
namespace generation
{

/// Create a set of points, distributed pro rata to the cell volumes.
/// The total number of particles will approximately equal \p density particles.
/// @return tuple with point coordinates, cell indices
// std::tuple<
//     Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
//     std::vector<int>>
template<std::floating_point T>
std::tuple<std::vector<double>, std::vector<int32_t>>
mesh_fill(const dolfinx::mesh::Mesh<T>& mesh, double density);

/// Create a set of n points at random positions within the cell.
/// @return
std::vector<double>
random_reference(dolfinx::mesh::CellType celltype, std::size_t n);

/// Create a set of n points at random positions within the reference
/// tetrahedron
/// @return Array of R^3-coordinates
std::vector<double>
random_reference_tetrahedron(std::size_t n);

/// Create a set of n points at random positions within the reference
/// triangle
/// @return Array of R^2-coordinates
std::vector<double>
random_reference_triangle(std::size_t n);
} // namespace generation
}; // namespace leopart