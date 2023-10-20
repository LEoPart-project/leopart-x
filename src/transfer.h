// Copyright: (c) 2020 Chris Richardson and Jakob Maljaars
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Field.h"
#include "Particles.h"

#include <dolfinx.h>
#include <basix/math.h>
#include <iostream>

#pragma once

namespace leopart::transfer
{
// /// Return basis values for each particle in function space
// template <dolfinx::scalar T, std::floating_point U>
// std::vector<T>
// get_particle_contributions(
//     const Particles<T>& pax,
//     const dolfinx::fem::FunctionSpace<U>& function_space)
// {
//   std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh = function_space.mesh();
//   const int gdim = mesh->geometry().dim();
//   const int tdim = mesh->topology()->dim();

//   // Count up particles in each cell
//   int ncells = mesh->topology()->index_map(tdim)->size_local();

//   const std::vector<std::vector<std::size_t>>& cell_to_particle = pax.cell_to_particle();
//   std::size_t nparticles = 0;
//   for (const std::vector<std::size_t>& q : cell_to_particle)
//     nparticles += q.size();

//   // Get coordinate map
//   const dolfinx::fem::CoordinateElement<U>& cmap = mesh->geometry().cmaps()[0];

//   // Prepare cell geometry
//   auto x_dofmap = mesh->geometry().dofmap();
//   const std::size_t num_dofs_g = cmap.dim();
//   std::span<const T> x_g = mesh->geometry().x();

//   // // Get cell permutation data
//   // mesh->topology_mutable().create_entity_permutations();
//   // const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info
//   //     = mesh->topology().get_cell_permutation_info();

//   // // Get element
//   // assert(function_space.element());
//   // std::shared_ptr<const dolfinx::fem::FiniteElement> element
//   //     = function_space.element();
//   // assert(element);
//   // const int block_size = element->block_size();
//   // const int reference_value_size = element->reference_value_size() / block_size;
//   // const int value_size = element->value_size() / block_size;
//   // const int space_dimension = element->space_dimension() / block_size;

//   // // Prepare geometry data structures
//   // Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
//   //     coordinate_dofs(num_dofs_g, gdim);

//   // // Each row represents the contribution from the particle in its cell
//   // Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
//   //     basis_data(nparticles, space_dimension * value_size);

//   // int p = 0;
//   // for (int c = 0; c < ncells; ++c)
//   // {
//   //   int np = cell_to_particle[c].size();
//   //   if (np == 0)
//   //   {
//   //     std::cout << "WARNING: no particles in cell " << c << "\n";
//   //     continue;
//   //   }
//   //   // Get cell geometry (coordinate dofs)
//   //   auto x_dofs = x_dofmap.links(c);
//   //   for (int i = 0; i < num_dofs_g; ++i)
//   //     coordinate_dofs.row(i) = x_g.row(x_dofs[i]).head(gdim);

//   //   // Physical and reference coordinates
//   //   Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x(
//   //       np, tdim);
//   //   Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X(
//   //       np, tdim);
//   //   // Prepare basis function data structures
//   //   Eigen::Tensor<double, 3, Eigen::RowMajor> basis_reference_values(
//   //       np, space_dimension, reference_value_size);
//   //   Eigen::Tensor<double, 3, Eigen::RowMajor> basis_values(np, space_dimension,
//   //                                                          value_size);
//   //   Eigen::Tensor<double, 3, Eigen::RowMajor> J(np, gdim, tdim);
//   //   Eigen::Array<double, Eigen::Dynamic, 1> detJ(np);
//   //   Eigen::Tensor<double, 3, Eigen::RowMajor> K(np, tdim, gdim);

//   //   for (int i = 0; i < np; ++i)
//   //     x.row(i) = pax.field(0).data(cell_to_particle[c][i]).head(gdim);

//   //   cmap.compute_reference_geometry(X, J, detJ, K, x, coordinate_dofs);
//   //   // Compute basis on reference element
//   //   element->evaluate_reference_basis(basis_reference_values, X);
//   //   // Push basis forward to physical element
//   //   element->transform_reference_basis(basis_values, basis_reference_values, X,
//   //                                      J, detJ, K, cell_info[c]);

//   //   // FIXME: avoid copy by using Eigen::TensorMap
//   //   // Copy basis data
//   //   std::copy(basis_values.data(),
//   //             basis_values.data() + np * space_dimension * value_size,
//   //             basis_data.row(p).data());
//   //   p += np;
//   // }
//   // return basis_data;
// }

/// Get basis values (not unrolled for block size) for a set of points and
/// corresponding cells.
/// @param[in] V The function space
/// @param[in] x The coordinates of the points. It has shape
/// (num_points, 3), flattened row major
/// @param[in] cells An array of cell indices. cells[i] is the index
/// of the cell that contains the point x(i). Negative cell indices
/// can be passed, and the corresponding point will be ignored.
/// @param[in,out] u The values at the points. Values are not computed
/// for points with a negative cell index. This argument must be
/// passed with the correct size.
/// @returns basis values (not unrolled for block size) for each point. shape
/// (num_points, number_of_dofs, value_size). Flattened row major
template <std::floating_point U>
std::pair<std::vector<U>, std::array<std::size_t, 3>>
evaluate_basis_functions(const dolfinx::fem::FunctionSpace<U>& V,
                         std::span<const U> x,
                         std::span<const std::int32_t> cells)
{
  assert(x.size() % 3 == 0);
  const std::size_t num_points = x.size() / 3;
  if (num_points != cells.size())
  {
    throw std::runtime_error(
        "Number of points and number of cells must be equal.");
  }

  // Get mesh
  auto mesh = V.mesh();
  assert(mesh);
  const std::size_t gdim = mesh->geometry().dim();
  const std::size_t tdim = mesh->topology()->dim();
  auto map = mesh->topology()->index_map(tdim);

  // Get geometry data
  namespace stdex = std::experimental;
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const std::int32_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      x_dofmap = mesh->geometry().dofmap();

  auto cmaps = mesh->geometry().cmaps();
  if (cmaps.size() > 1)
  {
    throw std::runtime_error(
        "Multiple coordinate maps in evaluate basis functions");
  }
  const std::size_t num_dofs_g = cmaps[0].dim();
  std::span<const U> x_g = mesh->geometry().x();

  // Get element
  auto element = V.element();
  assert(element);
  const int bs_element = element->block_size();
  const std::size_t reference_value_size
      = element->reference_value_size() / bs_element;

  // If the space has sub elements, concatenate the evaluations on the
  // sub elements
  const int num_sub_elements = element->num_sub_elements();
  if (num_sub_elements > 1 and num_sub_elements != bs_element)
  {
    throw std::runtime_error(
        "Evaluation of basis functions is not supported for mixed "
        "elements. Extract subspaces.");
  }

  // Return early if we have no points
  std::array<std::size_t, 4> basis_shape
      = element->basix_element().tabulate_shape(0, num_points);

  assert(basis_shape[2]
         == std::size_t(element->space_dimension() / bs_element));
  assert(basis_shape[3] == std::size_t(element->value_size() / bs_element));
  std::array<std::size_t, 3> reference_shape
      = {basis_shape[1], basis_shape[2], basis_shape[3]};
  std::vector<U> output_basis(std::reduce(
      reference_shape.begin(), reference_shape.end(), 1, std::multiplies{}));

  if (num_points == 0)
    return {output_basis, reference_shape};

  std::span<const std::uint32_t> cell_info;
  if (element->needs_dof_transformations())
  {
    mesh->topology_mutable()->create_entity_permutations();
    cell_info = std::span(mesh->topology()->get_cell_permutation_info());
  }

  using cmdspan4_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;
  using mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using mdspan3_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>;

  // Create buffer for coordinate dofs and point in physical space
  std::vector<U> coord_dofs_b(num_dofs_g * gdim);
  mdspan2_t coord_dofs(coord_dofs_b.data(), num_dofs_g, gdim);
  std::vector<U> xp_b(1 * gdim);
  mdspan2_t xp(xp_b.data(), 1, gdim);

  // Evaluate geometry basis at point (0, 0, 0) on the reference cell.
  // Used in affine case.
  std::array<std::size_t, 4> phi0_shape = cmaps[0].tabulate_shape(1, 1);
  std::vector<U> phi0_b(
      std::reduce(phi0_shape.begin(), phi0_shape.end(), 1, std::multiplies{}));
  cmdspan4_t phi0(phi0_b.data(), phi0_shape);
  cmaps[0].tabulate(1, std::vector<U>(tdim, 0), {1, tdim}, phi0_b);
  auto dphi0 = stdex::submdspan(phi0, std::pair(1, tdim + 1), 0,
                                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

  // Data structure for evaluating geometry basis at specific points.
  // Used in non-affine case.
  std::array<std::size_t, 4> phi_shape = cmaps[0].tabulate_shape(1, 1);
  std::vector<U> phi_b(
      std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
  cmdspan4_t phi(phi_b.data(), phi_shape);
  auto dphi = stdex::submdspan(phi, std::pair(1, tdim + 1), 0,
                               MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

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
    const int cell_index = cells[p];

    // Skip negative cell indices
    if (cell_index < 0)
      continue;

    // Get cell geometry (coordinate dofs)
    auto x_dofs = stdex::submdspan(x_dofmap, cell_index,
                                   MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < num_dofs_g; ++i)
    {
      const int pos = 3 * x_dofs[i];
      for (std::size_t j = 0; j < gdim; ++j)
        coord_dofs(i, j) = x_g[pos + j];
    }

    for (std::size_t j = 0; j < gdim; ++j)
      xp(0, j) = x[3 * p + j];

    auto _J
        = stdex::submdspan(J, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
                           MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    auto _K
        = stdex::submdspan(K, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
                           MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

    std::array<U, 3> Xpb = {0, 0, 0};
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        U, MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
               std::size_t, 1, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent>>
        Xp(Xpb.data(), 1, tdim);

    // Compute reference coordinates X, and J, detJ and K
    if (cmaps[0].is_affine())
    {
      dolfinx::fem::CoordinateElement<U>::compute_jacobian(dphi0, coord_dofs,
                                                           _J);
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
      cmaps[0].pull_back_nonaffine(Xp, xp, coord_dofs);

      cmaps[0].tabulate(1, std::span(Xpb.data(), tdim), {1, tdim}, phi_b);
      dolfinx::fem::CoordinateElement<U>::compute_jacobian(dphi, coord_dofs,
                                                           _J);
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
      basis_shape.begin(), basis_shape.end(), 1, std::multiplies{}));
  element->tabulate(reference_basisb, Xb, {X.extent(0), X.extent(1)}, 0);

  // Data structure to hold basis for transformation
  const std::size_t num_basis_values = basis_shape[2] * basis_shape[3];
  std::vector<U> basis_valuesb(num_basis_values);
  mdspan2_t basis_values(basis_valuesb.data(), basis_shape[2], basis_shape[3]);

  using xu_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using xU_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using xJ_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using xK_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  auto push_forward_fn
      = element->basix_element().template map_fn<xu_t, xU_t, xJ_t, xK_t>();

  auto apply_dof_transformation
      = element->template get_dof_transformation_function<U>();

  mdspan3_t full_basis(output_basis.data(), reference_shape);
  for (std::size_t p = 0; p < cells.size(); ++p)
  {
    const int cell_index = cells[p];
    // Skip negative cell indices
    if (cell_index < 0)
      continue;

    // Permute the reference values to account for the cell's orientation
    std::copy_n(std::next(reference_basisb.begin(), num_basis_values * p),
                num_basis_values, basis_valuesb.begin());
    apply_dof_transformation(basis_valuesb, cell_info, cell_index,
                             (int)reference_value_size);

    auto _U = stdex::submdspan(full_basis, p,
                               MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
                               MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    auto _J
        = stdex::submdspan(J, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
                           MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    auto _K
        = stdex::submdspan(K, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
                           MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    push_forward_fn(_U, basis_values, _J, detJ[p], _K);
  }
  return {output_basis, reference_shape};
}

/// Transfer information from the FE \p field to the particles by
/// evaluating dofs at particle positions.
///
/// @tparam T The function scalar type.
template <dolfinx::scalar T>
void transfer_to_particles(
    const Particles<T>& pax, Field<T>& field,
    std::shared_ptr<const dolfinx::fem::Function<T>> f)
{
  const std::vector<std::int32_t>& p2c = pax.particle_to_cell();
  
  const std::span<const T> x = pax.field("x").data();
  const std::array<std::size_t, 2> xshape = {
    pax.field("x").size(),  pax.field("x").value_size()};

  const std::span<T> u = field.data();
  const std::array<std::size_t, 2> ushape = {
    field.size(), field.value_size()};

  f->eval(x, xshape, p2c, u, ushape);
}


template <typename T, std::size_t d>
using mdspan_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, d>>;
template <typename T, std::size_t d>
using mdspan_ct = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, d>>;


template <dolfinx::scalar T>
void transpose(mdspan_ct<T, 2> A, mdspan_t<T, 2> A_T)
{
  for (std::size_t i = 0; i < A_T.extent(0); ++i)
    for (std::size_t j = 0; j < A_T.extent(1); ++j)
      A_T(i, j) = A(j, i);
}


template <dolfinx::scalar T>
void matmult(mdspan_ct<T, 2> A, mdspan_ct<T, 2> B, mdspan_t<T, 2> C)
{
  for (std::size_t i = 0; i < A.extent(0); ++i)
    for (std::size_t j = 0; j < B.extent(1); ++j)
    {
      T sum{0};
      for (std::size_t k = 0; k < A.extent(1); ++k)
        sum += A(i, k) * B(k, j);
      C(i, j) = sum;
    }  
}


/// Use basis values to transfer function from field given by value_index to
/// dolfinx Function
template <dolfinx::scalar T, std::floating_point U>
void transfer_to_function(
    std::shared_ptr<dolfinx::fem::Function<T>> f,
    const Particles<T>& pax,
    const Field<T>& field)
{
  std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh = f->function_space()->mesh();
  const int tdim = mesh->topology()->dim();
  std::int32_t ncells = mesh->topology()->index_map(tdim)->size_local();

  // Get element
  assert(f->function_space()->element());
  std::shared_ptr<const dolfinx::fem::FiniteElement<T>> element
      = f->function_space()->element();
  assert(element);
  const int block_size = element->block_size();
  const int value_size = element->value_size() / block_size;
  const int space_dimension = element->space_dimension() / block_size;
  assert(basis_values.cols() == value_size * space_dimension);

  std::shared_ptr<const dolfinx::fem::DofMap> dm
      = f->function_space()->dofmap();

  // Vector of expansion_coefficients to be set
  std::span<T> expansion_coefficients = f->x()->mutable_array();
  const std::vector<std::vector<std::size_t>>& cell_to_particle
    = pax.cell_to_particle();

  // Basis evaluations, shape (np, space_dimension, value_size)
  const auto [basis_evals, basis_shape] = evaluate_basis_functions<T>(
    *f->function_space(), pax.field("x").data(), pax.particle_to_cell());
  const mdspan_ct<T, 3> basis_evals_md(basis_evals.data(), basis_shape);

  // std::string msg("basis_evals: ");
  // for (auto& e : basis_evals)
  //   msg += std::to_string(e) + ", ";
  // std::cout << msg << std::endl;
  // msg = "basis_shape";
  // for (auto& e : basis_shape)
  //   msg += std::to_string(e) + ", ";
  // std::cout << msg << std::endl;

  auto print_mdspan = [](const auto span, const std::string name)
  {
    std::string msg = name + ":\n";
    for (int i = 0; i < span.extent(0); ++i)
    {
      for (int j = 0; j < span.extent(1); ++j)
        msg += std::to_string(span(i, j)) + ", ";
      msg += "\n";
    }
    std::cout << msg << std::endl;
  };

  int row_offset = 0;
  for (int c = 0; c < ncells; ++c)
  {
    const std::vector<std::size_t> cell_particles = cell_to_particle[c];
    int cell_np = cell_particles.size();
    
    std::vector<T> Q_T_data(cell_np * space_dimension);
    mdspan_t<T, 2> Q_T(Q_T_data.data(), space_dimension, cell_np);

    std::vector<T> Q_data(cell_np * space_dimension);
    mdspan_t<T, 2> Q(Q_data.data(), cell_np, space_dimension);
    for (std::size_t cell_p = 0; cell_p < cell_np; ++cell_p)
    {
      const std::size_t p_idx =  cell_particles[cell_p];
      for (std::size_t i = 0; i < space_dimension; ++i)
        Q(cell_p, i) = basis_evals_md(p_idx, i, 0); // Assume shape 1 for now
    }
    transpose<T>(Q, Q_T);

    std::vector<T> L_data(cell_np * block_size);
    mdspan_t<T, 2> L(L_data.data(), cell_np, block_size);
    for (std::size_t cell_p = 0; cell_p < cell_np; ++cell_p)
    {
      const std::size_t p_idx =  cell_particles[cell_p];
      for (std::size_t b = 0; b < block_size; ++ b)
        L(cell_p, b) = field.data()[p_idx + b];
    }
    // std::vector<T> L_data(cell_np * block_size)
    // mdspan_ct<T, 2> Q(basis_evals.data(), space_dimension, np);
    // mdspan_ct<T, 2> L(field.data().data(), np, block_size);

    // const std::vector<T> soln = basix::math::solve(Q, Q);
    // mdspan_ct<T, 2> soln_md(soln.data(), Q.extents());

    print_mdspan(Q, "Q mat");
    print_mdspan(Q_T, "Q_T mat");
    print_mdspan(L, "L mat");


    std::vector<T> QT_Q_data(Q_T.extent(0) * Q.extent(1));
    mdspan_t<T, 2> QT_Q(QT_Q_data.data(), Q_T.extent(0), Q.extent(1));
    matmult<T>(Q_T, Q, QT_Q);
    print_mdspan(QT_Q, "QT_Q mat");


    std::vector<T> QT_L_data(Q_T.extent(0) * L.extent(1));
    mdspan_t<T, 2> QT_L(QT_L_data.data(), Q_T.extent(0), L.extent(1));
    matmult<T>(Q_T, L, QT_L);
    print_mdspan(QT_L, "QT_L mat");

    const std::vector<T> soln = basix::math::solve<T>(QT_Q, QT_L);
    mdspan_ct<T, 2> soln_md(soln.data(), soln.size(), 1);

    print_mdspan(soln_md, "soln");


    // const auto [q, l] = eval_particle_cell_contributions(
    //   cell_particles, field, basis_evals, row_offset, 
    //   space_dimension, block_size);

  //   // Solve projection where
  //   // - q has shape [ndofs/block_size, np]
  //   // - l has shape [np, block_size]
  //   Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> u_tmp
  //       = (q * q.transpose()).ldlt().solve(q * l);
  //   Eigen::Map<Eigen::VectorXd> u_i(u_tmp.data(), space_dimension * block_size);

  //   auto dofs = dm->cell_dofs(c);

  //   assert(dofs.size() == space_dimension);

  //   for (int i = 0; i < dofs.size(); ++i)
  //   {
  //     expansion_coefficients[dofs[i]] = u_i[i];
  //   }
  //   row_offset += cell_particles.size();
  }
}


/// Evaluate the basis functions at particle positions in a prescribed cell.
/// Writes results to an Eigen::Matrix \f$q\f$ and Eigen::Vector \f$f\f$ to
/// compose the l2 problem \f$q q^T = q f\f$.
template <std::floating_point T>
std::pair<mdspan_t<T, 2>, mdspan_t<T, 2>>
eval_particle_cell_contributions(
    const std::vector<std::int32_t>& cell_particles, const Field<T>& field,
    const std::span<T> basis_values,
    int row_offset, int space_dimension, int block_size)
{
  int np = cell_particles.size();
  mdspan_t<T, 2> Q(basis_values.data(), space_dimension, np);
  mdspan_t<T, 2> L(field.data(), np, block_size);

  // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> q(space_dimension, np),
  //     l(np, block_size);
  // for (int p = 0; p < np; ++p)
  // {
  //   int pidx = cell_particles[p];
  //   std::span<T> cell_basis_vals = basis_values.subspan(
  //     row_offset, space_dimension);
    // Eigen::Map<const Eigen::VectorXd> basis(
    //     basis_values.row(row_offset++).data(), space_dimension);

    // q.col(p) = basis;
    // l.row(p) = field.data(pidx);
  // }
  return std::make_pair(Q, L);
}
} // namespace leopart::transfer
