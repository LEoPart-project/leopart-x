// Copyright (c) 2020-2023 Chris Richardson, Jakob Maljaars and Nathan Sime
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <dolfinx.h>

#include "utils.h"


namespace leopart
{
namespace generation
{

// Forward declarations
template<std::floating_point T>
std::vector<T> random_reference(
  const dolfinx::mesh::CellType celltype,
  const std::size_t n, std::mt19937& rgen);

template<std::floating_point T>
std::vector<T> random_reference_tetrahedron(
  const std::size_t n, std::mt19937& rgen);

template<std::floating_point T>
std::vector<T> random_reference_triangle(
  const std::size_t n, std::mt19937& rgen);

template<std::floating_point T>
std::vector<T> random_reference_cube(
  const std::size_t n, const std::size_t gdim, std::mt19937& rgen);

/// Create a set of points, distributed pro rata to the cell volumes.
///
/// @todo `dolfinx::fem::CoordinateElement::tabulate` performance is slow
///
/// @note Particles are generated in process owned cells only
///
/// @tparam T Position data type
/// @param mesh Finite element mesh
/// @param[in] np_per_cell Number of particles per cell
/// @param[in] cells The cells in which to generate particles
/// @param[in] seed_int Seed for the random number generator. Each cell
/// uses its own generator seeded from `seed_int` and the cell's original
/// (input) index, so for a given seed a cell always receives the same
/// particles, regardless of the number of MPI ranks or which other cells
/// are filled. If not provided, a nondeterministic seed is used.
///
/// @return tuple with point coordinates, cell indices
template<std::floating_point T>
std::tuple<std::vector<T>, std::vector<std::int32_t>>
mesh_fill(
  const dolfinx::mesh::Mesh<T>& mesh,
  std::span<const std::size_t> np_per_cell,
  std::span<const std::int32_t> cells,
  const std::optional<std::size_t>& seed_int = std::nullopt)
{
  if (np_per_cell.size() != cells.size())
    throw std::runtime_error(
      "Length of cells (" + std::to_string(cells.size()) + ") "
      "and particles per cell (" + std::to_string(np_per_cell.size()) + ") "
      "must be same");
  
  namespace md = dolfinx::md;

  // Geometric and topological dimensions
  const std::size_t gdim = mesh.geometry().dim();
  const std::size_t tdim = mesh.topology()->dim();

  // Get coordinate map
  if (mesh.geometry().cmaps().size() > 1)
    throw std::runtime_error("Mixed topology not supported");
  const dolfinx::fem::CoordinateElement<T>& cmap
    = mesh.geometry().cmaps().front();
  const std::vector<dolfinx::mesh::CellType> celltype
    = mesh.topology()->cell_types();

  // Prepare cell geometry
  auto x_dofmap = mesh.geometry().dofmaps().front();
  const std::size_t num_dofs_g = cmap.dim();
  std::span<const T> x_g = mesh.geometry().x();

  using mdspan2_t = leopart::utils::mdspan_t<T, 2>;
  using cmdspan4_t = leopart::utils::mdspan_t<const T, 4>;

  // Data to be returned
  std::vector<std::int32_t> p_cells;
  std::vector<T> xp_all;

  // Each cell's generator is seeded from (seed, original cell index) so
  // that the particles generated in a cell are independent of the mesh
  // partitioning and of the order in which cells are filled. If the
  // original cell indices are unavailable, fall back to the global cell
  // index, which is unique but depends on the partitioning.
  const std::uint64_t seed = seed_int.value_or(std::random_device{}());
  std::vector<std::int64_t> cell_keys(cells.size());
  const std::vector<std::vector<std::int64_t>>& original_cell_index
    = mesh.topology()->original_cell_index;
  if (!original_cell_index.empty() and !original_cell_index.front().empty())
  {
    std::span<const std::int64_t> oci = original_cell_index.front();
    std::ranges::transform(cells, cell_keys.begin(),
                           [&oci](std::int32_t c) { return oci[c]; });
  }
  else
  {
    mesh.topology()->index_map(tdim)->local_to_global(cells, cell_keys);
  }

  const std::size_t idxs = cells.size();
  for (std::size_t i = 0; i < idxs; ++i)
  {
    const std::int32_t c = cells[i];
    const std::size_t np = np_per_cell[i];

    // Generate coordinates on reference element
    const std::array<std::size_t, 2> Xshape = {np, tdim};
    const std::uint64_t key = cell_keys[i];
    std::seed_seq seq{static_cast<std::uint32_t>(seed),
                      static_cast<std::uint32_t>(seed >> 32),
                      static_cast<std::uint32_t>(key),
                      static_cast<std::uint32_t>(key >> 32)};
    std::mt19937 rgen(seq);
    const std::vector<T> X = random_reference<T>(celltype[0], np, rgen);

    // Physical coordinates of the generated points
    std::vector<T> x_b(np * gdim);
    mdspan2_t x(x_b.data(), np, gdim);

    std::vector<T> coordinate_dofs_b(num_dofs_g * gdim);
    mdspan2_t coordinate_dofs(coordinate_dofs_b.data(), num_dofs_g, gdim);

    const std::array<std::size_t, 4> phi_shape
        = cmap.tabulate_shape(0, Xshape[0]);
    std::vector<T> phi_b(std::reduce(
      phi_shape.begin(), phi_shape.end(), std::size_t(1), std::multiplies{}));
    cmdspan4_t phi_full(phi_b.data(), phi_shape);
    auto phi = md::submdspan(phi_full, 0, md::full_extent, md::full_extent, 0);

    cmap.tabulate(0, X, Xshape, phi_b);

    // Extract cell geometry
    auto x_dofs = md::submdspan(x_dofmap, c, md::full_extent);
    for (std::size_t k = 0; k < x_dofs.size(); ++k)
      for (std::size_t j = 0; j < gdim; ++j)
        coordinate_dofs(k, j) = x_g[3 * x_dofs[k] + j];

    // Tabulate physical coordinates on cell
    cmap.push_forward(x, coordinate_dofs, phi);

    // Fill data to be returned
    xp_all.insert(xp_all.end(), x_b.begin(), x_b.end());
    const std::vector<int> npcells(np, c);
    p_cells.insert(p_cells.end(), npcells.begin(), npcells.end());
  }

  return {std::move(xp_all), std::move(p_cells)};
}

/// As `mesh_fill` for all cells in the mesh
template<std::floating_point T>
std::tuple<std::vector<T>, std::vector<std::int32_t>>
mesh_fill(
  const dolfinx::mesh::Mesh<T>& mesh,
  const std::size_t np_per_cell,
  const std::optional<std::size_t>& seed = std::nullopt)
{
  const int tdim = mesh.topology()->dim();
  auto map = mesh.topology()->index_map(tdim);
  assert(map);
  const int num_cells = map->size_local();
  std::vector<std::int32_t> cells(num_cells, 0);
  std::iota(cells.begin(), cells.end(), 0);
  const std::vector<std::size_t> np_per_cell_vec(num_cells, np_per_cell);
  return mesh_fill(mesh, np_per_cell_vec, cells, seed);
}

/// Create a set of n points at random positions within the cell.
///
/// @tparam Position data type
/// @param[in] celltype The DOLFINx cell type
/// @param[in] n Number of particles per cell
/// @param[in,out] rgen Random number generator
///
/// @return Array of \f(\mathbb{R}^d\f)-coordinates
template<std::floating_point T>
std::vector<T> random_reference(
  const dolfinx::mesh::CellType celltype,
  const std::size_t n, std::mt19937& rgen)
{
  if (celltype == dolfinx::mesh::CellType::triangle)
    return random_reference_triangle<T>(n, rgen);
  if (celltype == dolfinx::mesh::CellType::tetrahedron)
    return random_reference_tetrahedron<T>(n, rgen);
  if (celltype == dolfinx::mesh::CellType::quadrilateral)
    return random_reference_cube<T>(n, 2, rgen);
  if (celltype == dolfinx::mesh::CellType::hexahedron)
    return random_reference_cube<T>(n, 3, rgen);

  throw std::runtime_error("Unsupported cell type");
  return std::vector<T>{};
}

/// Create a set of n points at random positions within the reference
/// tetrahedron
///
/// @tparam T Position data type
/// @param[in] n Number of particles
/// @param[in,out] rgen Random number generator
///
/// @return Array of \f(\mathbb{R}^3\f)-coordinates
template<std::floating_point T>
std::vector<T> random_reference_tetrahedron(
  const std::size_t n, std::mt19937& rgen)
{
  const std::size_t gdim = 3;
  std::vector<T> p(n * gdim, 1.0);
  std::vector<T> r(gdim);

  std::uniform_real_distribution<T> dist(-1.0, 1.0);

  for (std::size_t i = 0; i < n; ++i)
  {
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
    for (std::size_t j = 0; j < gdim; ++j)
      p_row[j] += r[j];
  }

  for (T& val : p)
    val /= 2.0;
  return p;
}

/// Create a set of n points at random positions within the reference
/// triangle
///
/// @tparam T Position data type
/// @param[in] n Number of particles
/// @param[in,out] rgen Random number generator
///
/// @return Array of \f(\mathbb{R}^2\f)-coordinates
template<std::floating_point T>
std::vector<T> random_reference_triangle(
  const std::size_t n, std::mt19937& rgen)
{
  const std::size_t gdim = 2;
  std::vector<T> p(n * gdim);

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

/// Create a set of points in the given dimension, \f(d\f)
/// at random positions within the reference domain \f( (0, 1)^d \f)
///
/// @tparam T Position data type
/// @param[in] n Number of particles
/// @param[in] gdim Spatial dimension
/// @param[in,out] rgen Random number generator
///
/// @return Array of \f(\mathbb{R}^d\f)-coordinates
template<std::floating_point T>
std::vector<T> random_reference_cube(
  const std::size_t n, const std::size_t gdim, std::mt19937& rgen)
{
  std::vector<T> p(n * gdim);

  std::uniform_real_distribution<T> dist(0.0, 1.0);

  std::generate(p.begin(), p.end(), [&dist, &rgen]() { return dist(rgen); });
  return p;
}

/// As `random_reference` using a generator initialised with `seed`
template<std::floating_point T>
std::vector<T> random_reference(
  const dolfinx::mesh::CellType celltype,
  const std::size_t n, const std::size_t seed)
{
  std::mt19937 rgen(seed);
  return random_reference<T>(celltype, n, rgen);
}

/// As `random_reference_tetrahedron` using a generator initialised with
/// `seed`
template<std::floating_point T>
std::vector<T> random_reference_tetrahedron(
  const std::size_t n, const std::size_t seed)
{
  std::mt19937 rgen(seed);
  return random_reference_tetrahedron<T>(n, rgen);
}

/// As `random_reference_triangle` using a generator initialised with `seed`
template<std::floating_point T>
std::vector<T> random_reference_triangle(
  const std::size_t n, const std::size_t seed)
{
  std::mt19937 rgen(seed);
  return random_reference_triangle<T>(n, rgen);
}

/// As `random_reference_cube` using a generator initialised with `seed`
template<std::floating_point T>
std::vector<T> random_reference_cube(
  const std::size_t n, const std::size_t gdim, const std::size_t seed)
{
  std::mt19937 rgen(seed);
  return random_reference_cube<T>(n, gdim, rgen);
}

/// Create a set of points located at the DoF coordinates of a
/// function space in every cell.
///
/// @note Coordinates may overlap.
///
/// @tparam T Function space geometry data type
/// @param V Function space DoF coordinates to generate
///
/// @return Array of \f(\mathbb{R}^d\f)-coordinates
template<std::floating_point T>
std::tuple<std::vector<T>, std::vector<std::int32_t>>
generate_at_dof_coords(const dolfinx::fem::FunctionSpace<T>& V)
{
  const std::vector<T> dof_coords = V.tabulate_dof_coordinates(false);
  const std::size_t tdim = V.mesh()->topology()->dim();
  const std::size_t n_cells = V.mesh()->topology()->index_map(tdim)->size_local();
  const std::size_t gdim = 3;

  // Assumed single cell type
  const std::size_t ndof_per_cell = V.dofmap()->cell_dofs(0).size();
  std::vector<T> px(n_cells * ndof_per_cell * gdim, 0.0);
  std::vector<std::int32_t> p_to_cells(n_cells * ndof_per_cell, 0);

  for (std::size_t c = 0; c < n_cells; ++c)
  {
    std::size_t dof_offset = 0;
    for (const auto& dof : V.dofmap()->cell_dofs(c))
    {
      const std::span<const T> dof_x(dof_coords.begin() + dof * gdim, gdim);
      std::copy(dof_x.begin(), dof_x.end(),
       px.begin() + c * ndof_per_cell * gdim + dof_offset * gdim);
      ++dof_offset;
    }

    std::fill_n(p_to_cells.begin() + c * ndof_per_cell, ndof_per_cell, c);
  }

  return {std::move(px), std::move(p_to_cells)};
}

} // namespace generation
}; // namespace leopart