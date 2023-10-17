// Copyright: (c) 2020 Chris Richardson and Jakob Maljaars
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cinttypes>
#include <dolfinx.h>
#include <span>
#include <stdexcept>
#include <tuple>
#include <random>
#include <vector>

namespace dolfinx
{
namespace mesh
{
template <std::floating_point T>
class Mesh;
enum class CellType;
} // namespace mesh
namespace fem
{
template <std::floating_point T>
class CoordinateElement;
}
} // namespace dolfinx

namespace leopart
{
namespace generation
{

/// Create a set of points, distributed pro rata to the cell volumes.
/// The total number of particles will approximately equal \p density particles.
///
/// @todo `dolfinx::fem::CoordinateElement::tabulate` performance is too slow
/// to generate a uniquely random set of points in each cell
///
/// @return tuple with point coordinates, cell indices
template<std::floating_point T>
std::tuple<std::vector<T>, std::vector<std::int32_t>>
mesh_fill(const dolfinx::mesh::Mesh<T>& mesh, const std::size_t np_per_cell)
{
  // Geometric dimension
  const std::size_t gdim = mesh.geometry().dim();
  const int tdim = mesh.topology()->dim();

  // Get coordinate map
  if (mesh.geometry().cmaps().size() > 1)
    throw std::runtime_error("Mixed topology not supported");
  const dolfinx::fem::CoordinateElement<T>& cmap = mesh.geometry().cmaps()[0];
  const std::vector<dolfinx::mesh::CellType> celltype = mesh.topology()->cell_types();

  // Prepare cell geometry
  auto x_dofmap = mesh.geometry().dofmap();
  const std::size_t num_dofs_g = cmap.dim();
  std::span<const T> x_g = mesh.geometry().x();

  // Array to hold coordinates to return
  const std::array<std::size_t, 2> Xshape = {np_per_cell, gdim};
  std::vector<T> coords(Xshape[0] * Xshape[1], 0);

  using mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using cmdspan4_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;

  // Loop over cells and tabulate dofs
  std::vector<T> x_b(Xshape[0] * Xshape[1]);
  mdspan2_t x(x_b.data(), Xshape[0], Xshape[1]);

  std::vector<T> coordinate_dofs_b(num_dofs_g * gdim);
  mdspan2_t coordinate_dofs(coordinate_dofs_b.data(), num_dofs_g, gdim);

  auto map = mesh.topology()->index_map(tdim);
  assert(map);
  const int num_cells = map->size_local() + map->num_ghosts();

  std::span<const std::uint32_t> cell_info;

  const std::array<std::size_t, 4> phi_shape
      = cmap.tabulate_shape(0, Xshape[0]);
  std::vector<T> phi_b(
      std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
  cmdspan4_t phi_full(phi_b.data(), phi_shape);
  auto phi = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE::
      submdspan(phi_full, 0, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

  // Data to be returned
  std::vector<std::int32_t> p_cells;
  std::vector<T> xp_all;

  // Generate coordinates on reference element
  const std::vector<T> X = random_reference<T>(celltype[0], np_per_cell);
  cmap.tabulate(0, X, Xshape, phi_b);

  for (int c = 0; c < num_cells; ++c)
  {
    // Extract cell geometry
    auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::
        MDSPAN_IMPL_PROPOSED_NAMESPACE::submdspan(
            x_dofmap, c, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
      for (std::size_t j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g[3 * x_dofs[i] + j];

    // Tabulate physical coordinates on cell
    cmap.push_forward(x, coordinate_dofs, phi);

    // Fill data to be returned
    xp_all.insert(xp_all.end(), x_b.begin(), x_b.end());
    const std::vector<int> npcells(np_per_cell, c);
    p_cells.insert(p_cells.end(), npcells.begin(), npcells.end());
  }

  return {xp_all, p_cells};
}

/// Create a set of n points at random positions within the cell.
/// @return
template<std::floating_point T>
std::vector<T> random_reference(dolfinx::mesh::CellType celltype,
                                std::size_t n)
{
  if (celltype == dolfinx::mesh::CellType::triangle)
    return random_reference_triangle<T>(n);
  if (celltype == dolfinx::mesh::CellType::tetrahedron)
    return random_reference_tetrahedron<T>(n);

  throw std::runtime_error("Unsupported cell type");
  return std::vector<T>{};
}

/// Create a set of n points at random positions within the reference
/// tetrahedron
/// @return Array of R^3-coordinates
template<std::floating_point T>
std::vector<double> random_reference_tetrahedron(std::size_t n)
{
  const std::size_t gdim = 3;
  std::vector<T> p(n * gdim, 1.0);
  std::vector<T> r(gdim);

  std::random_device rd;
  std::mt19937 rgen(rd());
  std::uniform_real_distribution<T> dist(-1.0, 1.0);

  for (int i = 0; i < n; ++i)
  {    
    // Eigen::RowVector3d r = Eigen::Vector3d::Random();
    std::generate(r.begin(), r.end(),
                  [&dist, &rgen]() { return dist(rgen); });
    T& x = r[0];
    T& y = r[1];
    T& z = r[2];

    // Fold cube into tetrahedron
    if ((x + z) > 0)
    {
      x = -x;
      z = -z;
    }

    if ((y + z) > 0)
    {
      z = -z - x - 1;
      y = -y;
    }
    else if ((x + y + z) > -1)
    {
      x = -x - z - 1;
      y = -y - z - 1;
    }

    const auto p_row = &p[i * gdim];
    for (int j = 0; j < gdim; ++j)
      p_row[j] += r[j];
  }

  for (T& val : p)
    val /= 2.0;
  return p;
}

/// Create a set of n points at random positions within the reference
/// triangle
/// @return Array of R^2-coordinates
template<std::floating_point T>
std::vector<T> random_reference_triangle(std::size_t n)
{
  const std::size_t gdim = 2;
  std::vector<T> p(n * gdim);

  std::random_device rd;
  std::mt19937 rgen(rd());
  std::uniform_real_distribution<T> dist(-1.0, 1.0);

  std::generate(p.begin(), p.end(), [&dist, &rgen]() { return dist(rgen); });

  for (std::size_t i = 0; i < n; ++i)
  {
    std::span x(p.begin() + i * gdim, gdim);
    if (x[0] + x[1] > 0.0)
    {
      x[0] = (1 - x[0]);
      x[1] = (1 - x[1]);
    }
    else
    {
      x[0] = (1 + x[0]);
      x[1] = (1 + x[1]);
    }
  }
  
  for (T& val : p)
    val /= 2.0;
  return p;
}

} // namespace generation
}; // namespace leopart