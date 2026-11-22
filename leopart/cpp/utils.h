// Copyright (c) 2020-2023 Chris Richardson, Jakob Maljaars and Nathan Sime
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx.h>
#include <iostream>
#include "Particles.h"


namespace leopart::utils
{
namespace md = dolfinx::md;

/// @brief Utility alias for frequently used mdpsan with dynamic extents
/// @tparam T Data type
/// @tparam d Rank
template <typename T, std::size_t d>
using mdspan_t = md::mdspan<T, md::dextents<std::size_t, d>>;

/// @brief Format and print prefixing the message with the source
/// MPI rank on MPI_COMM_WORLD
///
/// @param str Message to print
inline void print(const std::string str)
{
  const int rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  const std::string msg = "[" + std::to_string(rank) + "]: " + str;
  std::cout << msg << std::endl << std::flush;
}

/// Format and print
/// @param[in] span Rank 2 mdspan to format and print.
/// @param[in] prefix An identifier prefixing the string
void print_mdspan(
  const auto span, const std::string prefix = "")
{
  const int rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  std::string msg = "[" + std::to_string(rank) + "]: " + prefix + "\n";
  for (std::size_t i = 0; i < span.extent(0); ++i)
  {
    for (std::size_t j = 0; j < span.extent(1); ++j)
      msg += std::to_string(span(i, j)) + ", ";
    msg += "\n";
  }
  std::cout << msg << std::endl << std::flush;
}

/// Format and print
void print_iterable(
  const auto span, const std::string prefix = "")
{
  const int rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  std::string msg = "[" + std::to_string(rank) + "]: " + prefix + "\n";
  for (const auto& item: span)
    msg += std::to_string(item) + ", ";
  std::cout << msg << std::endl << std::flush;
}


/// Adapted from dolfinx_mpc::utils::evaluate_basis_functions
/// originally authored by Jorgen S. Dokken
///
/// Get basis values (not unrolled for block size) for a set of points and
/// corresponding cells.
/// @param[in] V The function space
/// @param[in] x The coordinates of the points. It has shape
/// (num_points, xdim), flattened row major, where xdim >= gdim.
/// @param[in] xdim The number of components stored for each point in x
/// @param[in] cells An array of cell indices. cells[i] is the index
/// of the cell that contains the point x(i). Negative cell indices
/// can be passed, and the corresponding point will be ignored.
/// @param[in] tol Tolerance of the Newton solver used in non-affine
/// pull-backs
/// @param[in] maxit Maximum Newton iterations used in non-affine pull-backs
/// @returns basis values (not unrolled for block size) for each point. shape
/// (num_points, number_of_dofs, value_size). Flattened row major
template <std::floating_point U>
std::pair<std::vector<U>, std::array<std::size_t, 3>>
evaluate_basis_functions(const dolfinx::fem::FunctionSpace<U>& V,
                         std::span<const U> x,
                         const std::size_t xdim,
                         std::span<const std::int32_t> cells,
                         const double tol = 1.0e-6,
                         const int maxit = 15)
{
  // Get mesh
  auto mesh = V.mesh();
  assert(mesh);
  const std::size_t gdim = mesh->geometry().dim();
  const std::size_t tdim = mesh->topology()->dim();

  if (xdim < gdim)
    throw std::runtime_error(
        "Number of point components must be at least the geometric dimension");
  if (x.size() % xdim != 0)
    throw std::runtime_error(
        "Size of point data is not a multiple of the point dimension");
  const std::size_t num_points = x.size() / xdim;
  if (num_points != cells.size())
  {
    throw std::runtime_error(
        "Number of points and number of cells must be equal.");
  }

  // Get geometry data
  if (mesh->geometry().cmaps().size() > 1)
  {
    throw std::runtime_error(
        "Multiple coordinate maps in evaluate basis functions");
  }
  const dolfinx::fem::CoordinateElement<U>& cmap
      = mesh->geometry().cmaps().front();
  auto x_dofmap = mesh->geometry().dofmaps().front();
  const std::size_t num_dofs_g = cmap.dim();
  std::span<const U> x_g = mesh->geometry().x();

  // Get element
  auto element = V.element();
  assert(element);
  const int bs_element = element->block_size();
  const std::size_t reference_value_size
      = element->reference_value_size();

  // If the space has sub elements, concatenate the evaluations on the
  // sub elements
  const int num_sub_elements = element->num_sub_elements();
  if (num_sub_elements > 1 and num_sub_elements != bs_element)
  {
    throw std::runtime_error(
        "Evaluation of basis functions is not supported for mixed "
        "elements. Extract subspaces.");
  }

  const std::array<std::size_t, 4> basis_shape
      = element->basix_element().tabulate_shape(0, num_points);
  assert(basis_shape[2]
         == std::size_t(element->space_dimension() / bs_element));
  const std::array<std::size_t, 3> reference_shape
      = {basis_shape[1], basis_shape[2], basis_shape[3]};
  std::vector<U> output_basis(std::reduce(
      reference_shape.begin(), reference_shape.end(), std::size_t(1),
      std::multiplies{}));

  // Return early if we have no points
  if (num_points == 0)
    return {output_basis, reference_shape};

  std::span<const std::uint32_t> cell_info;
  if (element->needs_dof_transformations())
  {
    mesh->topology_mutable()->create_entity_permutations();
    cell_info = std::span(mesh->topology()->get_cell_permutation_info());
  }

  using cmdspan4_t = mdspan_t<const U, 4>;
  using mdspan2_t = mdspan_t<U, 2>;
  using mdspan3_t = mdspan_t<U, 3>;

  // Create buffer for coordinate dofs and point in physical space
  std::vector<U> coord_dofs_b(num_dofs_g * gdim);
  mdspan2_t coord_dofs(coord_dofs_b.data(), num_dofs_g, gdim);
  std::vector<U> xp_b(1 * gdim);
  mdspan2_t xp(xp_b.data(), 1, gdim);

  // Evaluate geometry basis at point (0, 0, 0) on the reference cell.
  // Used in affine case.
  std::array<std::size_t, 4> phi0_shape = cmap.tabulate_shape(1, 1);
  std::vector<U> phi0_b(std::reduce(
      phi0_shape.begin(), phi0_shape.end(), std::size_t(1), std::multiplies{}));
  cmdspan4_t phi0(phi0_b.data(), phi0_shape);
  cmap.tabulate(1, std::vector<U>(tdim, 0), {1, tdim}, phi0_b);
  auto dphi0 = md::submdspan(phi0, std::pair(1, tdim + 1), 0,
                             md::full_extent, 0);

  // Data structure for evaluating geometry basis at specific points.
  // Used in non-affine case.
  std::array<std::size_t, 4> phi_shape = cmap.tabulate_shape(1, 1);
  std::vector<U> phi_b(std::reduce(
      phi_shape.begin(), phi_shape.end(), std::size_t(1), std::multiplies{}));
  cmdspan4_t phi(phi_b.data(), phi_shape);
  auto dphi = md::submdspan(phi, std::pair(1, tdim + 1), 0,
                            md::full_extent, 0);

  // Reference coordinates for each point
  std::vector<U> Xb(num_points * tdim);
  mdspan2_t X(Xb.data(), num_points, tdim);

  // Geometry data at each point
  std::vector<U> J_b(num_points * gdim * tdim);
  mdspan3_t J(J_b.data(), num_points, gdim, tdim);
  std::vector<U> K_b(num_points * tdim * gdim);
  mdspan3_t K(K_b.data(), num_points, tdim, gdim);
  std::vector<U> detJ(num_points);
  std::vector<U> det_scratch(2 * gdim * tdim);

  // Prepare geometry data in each cell
  for (std::size_t p = 0; p < cells.size(); ++p)
  {
    const std::int32_t cell_index = cells[p];

    // Skip negative cell indices
    if (cell_index < 0)
      continue;

    // Get cell geometry (coordinate dofs)
    auto x_dofs = md::submdspan(x_dofmap, cell_index, md::full_extent);
    for (std::size_t i = 0; i < num_dofs_g; ++i)
    {
      const std::size_t pos = 3 * x_dofs[i];
      for (std::size_t j = 0; j < gdim; ++j)
        coord_dofs(i, j) = x_g[pos + j];
    }

    for (std::size_t j = 0; j < gdim; ++j)
      xp(0, j) = x[xdim * p + j];

    auto _J = md::submdspan(J, p, md::full_extent, md::full_extent);
    auto _K = md::submdspan(K, p, md::full_extent, md::full_extent);

    std::array<U, 3> Xpb = {0, 0, 0};
    md::mdspan<U, md::extents<std::size_t, 1, md::dynamic_extent>>
        Xp(Xpb.data(), 1, tdim);

    // Compute reference coordinates X, and J, detJ and K
    if (cmap.is_affine())
    {
      dolfinx::fem::CoordinateElement<U>::compute_jacobian(
        dphi0, coord_dofs, _J);
      dolfinx::fem::CoordinateElement<U>::compute_jacobian_inverse(_J, _K);
      std::array<U, 3> x0 = {0, 0, 0};
      for (std::size_t i = 0; i < coord_dofs.extent(1); ++i)
        x0[i] += coord_dofs(0, i);
      dolfinx::fem::CoordinateElement<U>::pull_back_affine(Xp, _K, x0, xp);
      detJ[p]
          = dolfinx::fem::CoordinateElement<U>::compute_jacobian_determinant(
              _J, det_scratch);
    }
    else
    {
      // Pull-back physical point xp to reference coordinate Xp
      cmap.pull_back_nonaffine(Xp, xp, coord_dofs, tol, maxit);

      cmap.tabulate(1, std::span(Xpb.data(), tdim), {1, tdim}, phi_b);
      dolfinx::fem::CoordinateElement<U>::compute_jacobian(
        dphi, coord_dofs, _J);
      dolfinx::fem::CoordinateElement<U>::compute_jacobian_inverse(_J, _K);
      detJ[p]
          = dolfinx::fem::CoordinateElement<U>::compute_jacobian_determinant(
              _J, det_scratch);
    }

    for (std::size_t j = 0; j < X.extent(1); ++j)
      X(p, j) = Xpb[j];
  }

  // Compute basis on reference element
  std::vector<U> reference_basisb(std::reduce(
      basis_shape.begin(), basis_shape.end(), std::size_t(1),
      std::multiplies{}));
  element->tabulate(reference_basisb, Xb, {X.extent(0), X.extent(1)}, 0);

  // Data structure to hold basis for transformation
  const std::size_t num_basis_values = basis_shape[2] * basis_shape[3];
  std::vector<U> basis_valuesb(num_basis_values);
  mdspan2_t basis_values(basis_valuesb.data(), basis_shape[2], basis_shape[3]);

  using xu_t = mdspan_t<U, 2>;
  using xU_t = mdspan_t<const U, 2>;
  using xJ_t = mdspan_t<const U, 2>;
  using xK_t = mdspan_t<const U, 2>;
  auto push_forward_fn
      = element->basix_element().template map_fn<xu_t, xU_t, xJ_t, xK_t>();

  auto apply_dof_transformation
      = element->template dof_transformation_fn<U>(
          dolfinx::fem::doftransform::standard);

  for (std::size_t p = 0; p < cells.size(); ++p)
  {
    const std::int32_t cell_index = cells[p];
    // Skip negative cell indices
    if (cell_index < 0)
      continue;

    // Permute the reference values to account for the cell's orientation
    std::copy_n(std::next(reference_basisb.begin(), num_basis_values * p),
                num_basis_values, basis_valuesb.begin());
    apply_dof_transformation(basis_valuesb, cell_info, cell_index,
                             (int)reference_value_size);

    xu_t phys_basis(
      output_basis.data() + p * reference_shape[1] * reference_shape[2],
      reference_shape[1], reference_shape[2]);
    auto _J = md::submdspan(J, p, md::full_extent, md::full_extent);
    auto _K = md::submdspan(K, p, md::full_extent, md::full_extent);
    push_forward_fn(phys_basis, basis_values, _J, detJ[p], _K);
  }
  return {output_basis, reference_shape};
}
} // namespace leopart::utils
