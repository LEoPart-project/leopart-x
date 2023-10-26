// Copyright: (c) 2020-2023 Chris Richardson, Jakob Maljaars and Nathan Sime
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Field.h"
#include "Particles.h"
#include "external/quadprog_mdspan/QuadProg++.hh"
#include "math.h"
#include "utils.h"
// #include "external/quadprog_mdspan/Array.hh"

#include <basix/math.h>
#include <dolfinx.h>
#include <iostream>

namespace leopart::transfer
{
using leopart::utils::mdspan_t;

/// Transfer information from the FE \p field to the particles by
/// interpolating the finite element function at particles' positions.
/// Given \f(n_p\f) particles with positions \f(x_p\f), we compute
///  particle data \f(u_p\f)
///
/// \f[
///    u_p = u(x_p), \quad p = 1,\ldots,n_p
/// \f]
///
/// @tparam T The function scalar type.
/// @param pax Particles collection
/// @param field Field data into which to store the interpolation
/// @param f The finite element function to be interpolated
template <dolfinx::scalar T>
void transfer_to_particles(const Particles<T>& pax, Field<T>& field,
                           std::shared_ptr<const dolfinx::fem::Function<T>> f)
{
  const std::vector<std::int32_t>& p2c = pax.particle_to_cell();

  const std::span<const T> x = pax.field("x").data();
  const std::array<std::size_t, 2> xshape
      = {pax.field("x").size(), pax.field("x").value_size()};

  const std::span<T> u = field.data();
  const std::array<std::size_t, 2> ushape = {field.size(), field.value_size()};

  f->eval(x, xshape, p2c, u, ushape);
}

/// Transfer the provided particle field data to the finite element
/// function using local l_2 projection.
/// We solve the problem: find \f(u_h \in V\f) such that
///
/// \f[
///    u_h(x_p) v(x_p) = u_p v(x_p) \quad \forall v \in V, \; p = 1,\ldots,n_p.
/// \f]
///
/// Here \f(u_p\f) is the \f(p\f)th particle's data, \f(u_p\f) is the \f(p\f)th
/// particle's position, \f(n_p\f) is the total number of particles
/// and \f(V\f) is the function space to which the provided finite element
/// function belongs.
///
/// @tparam T The function scalar type
/// @tparam U The function geometry type
/// @param f The finite element function
/// @param pax The particles collection
/// @param field The field data to be transferred
template <dolfinx::scalar T, std::floating_point U>
void transfer_to_function(std::shared_ptr<dolfinx::fem::Function<T>> f,
                          const Particles<T>& pax, const Field<T>& field)
{
  std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh
      = f->function_space()->mesh();
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

  std::shared_ptr<const dolfinx::fem::DofMap> dm
      = f->function_space()->dofmap();

  // Vector of expansion_coefficients to be set
  std::span<T> expansion_coefficients = f->x()->mutable_array();
  const std::vector<std::vector<std::size_t>>& cell_to_particle
      = pax.cell_to_particle();

  // Basis evaluations, shape (np, space_dimension, value_size)
  const auto [basis_evals, basis_shape]
      = leopart::utils::evaluate_basis_functions<T>(
          *f->function_space(), pax.field("x").data(), pax.particle_to_cell());
  const mdspan_t<const T, 3> basis_evals_md(basis_evals.data(), basis_shape);

  // Assemble and solve Q^T Q u = Q^T L in each cell, where
  // Q = \phi(x_p), Q^T = \psi(x_p), L = u_p,
  // \phi is the trial function, \psi is the test function, x_p
  // are particles' position and u_p are particles' datum/data.
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
      const std::size_t p_idx = cell_particles[cell_p];
      for (std::size_t i = 0; i < space_dimension; ++i)
        Q(cell_p, i) = basis_evals_md(p_idx, i, 0); // Assume shape 1 for now
    }
    leopart::math::transpose<T>(Q, Q_T);

    std::vector<T> L_data(cell_np * block_size);
    mdspan_t<T, 2> L(L_data.data(), cell_np, block_size);
    for (std::size_t cell_p = 0; cell_p < cell_np; ++cell_p)
    {
      const std::size_t p_idx = cell_particles[cell_p];
      for (std::size_t b = 0; b < block_size; ++b)
        L(cell_p, b) = field.data()[p_idx + b];
    }

    std::vector<T> QT_Q_data(Q_T.extent(0) * Q.extent(1));
    mdspan_t<T, 2> QT_Q(QT_Q_data.data(), Q_T.extent(0), Q.extent(1));
    leopart::math::matmult<T>(Q_T, Q, QT_Q);

    std::vector<T> QT_L_data(Q_T.extent(0) * L.extent(1));
    mdspan_t<T, 2> QT_L(QT_L_data.data(), Q_T.extent(0), L.extent(1));
    leopart::math::matmult<T>(Q_T, L, QT_L);

    const std::vector<T> soln = basix::math::solve<T>(QT_Q, QT_L);
    mdspan_t<const T, 2> soln_md(soln.data(), soln.size(), 1);

    const auto& dofs = dm->cell_dofs(c);
    for (int i = 0; i < dofs.size(); ++i)
      expansion_coefficients[dofs[i]] = soln[i];
  }
}

/// Transfer the provided particle field data to the finite element
/// function using constrained local l_2 projection.
///
/// @note Currently only data and functions with value shape = 1
/// are supported.
///
/// @tparam T The function scalar type
/// @tparam U The function geometry type
/// @param f The finite element function
/// @param pax The particles collection
/// @param field The field data to be transferred
/// @param l Constraint lower bound
/// @param u Constraint upper bound
template <dolfinx::scalar T, std::floating_point U>
void transfer_to_function_constrained(
    std::shared_ptr<dolfinx::fem::Function<T>> f, const Particles<T>& pax,
    const Field<T>& field, const T l, const T u)
{
  std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh
      = f->function_space()->mesh();
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

  std::shared_ptr<const dolfinx::fem::DofMap> dm
      = f->function_space()->dofmap();

  // Vector of expansion_coefficients to be set
  std::span<T> expansion_coefficients = f->x()->mutable_array();
  const std::vector<std::vector<std::size_t>>& cell_to_particle
      = pax.cell_to_particle();

  // Basis evaluations, shape (np, space_dimension, value_size)
  const auto [basis_evals, basis_shape]
      = leopart::utils::evaluate_basis_functions<T>(
          *f->function_space(), pax.field("x").data(), pax.particle_to_cell());
  const mdspan_t<const T, 3> basis_evals_md(basis_evals.data(), basis_shape);

  // QuadProg specifics for constraints
  // CE^T x + ce0 =  0
  // CI^T x + ci0 >= 0
  std::vector<T> CE_data(space_dimension, 0.0),
      CI_data(space_dimension * space_dimension * value_size * 2, 0.0);
  quadprogpp::mdMatrix<T> CE(CE_data.data(), space_dimension, 0);
  quadprogpp::mdMatrix<T> CI(CI_data.data(), space_dimension,
                             space_dimension * value_size * 2);
  std::vector<T> ce0(0, 0.0), ci0(space_dimension * value_size * 2, 0.0);

  for (std::size_t i = 0; i < space_dimension; i++)
  {
    CI(i, i) = 1.;
    CI(i, i + space_dimension) = -1;
    ci0[i] = -l;
    ci0[i + space_dimension] = u;
  }

  // Assemble and solve Q^T Q u = Q^T L in each cell, where
  // Q = \phi(x_p), Q^T = \psi(x_p), L = u_p,
  // \phi is the trial function, \psi is the test function, x_p
  // are particles' position and u_p are particles' datum/data.
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
      const std::size_t p_idx = cell_particles[cell_p];
      for (std::size_t i = 0; i < space_dimension; ++i)
        Q(cell_p, i) = basis_evals_md(p_idx, i, 0); // Assume shape 1 for now
    }
    leopart::math::transpose<T>(Q, Q_T);

    std::vector<T> L_data(cell_np * block_size);
    mdspan_t<T, 2> L(L_data.data(), cell_np, block_size);
    for (std::size_t cell_p = 0; cell_p < cell_np; ++cell_p)
    {
      const std::size_t p_idx = cell_particles[cell_p];
      for (std::size_t b = 0; b < block_size; ++b)
        L(cell_p, b) = field.data()[p_idx + b];
    }

    std::vector<T> QT_Q_data(Q_T.extent(0) * Q.extent(1));
    mdspan_t<T, 2> QT_Q(QT_Q_data.data(), Q_T.extent(0), Q.extent(1));
    leopart::math::matmult<T>(Q_T, Q, QT_Q);

    std::vector<T> QT_L_data(Q_T.extent(0) * L.extent(1));
    mdspan_t<T, 2> QT_L(QT_L_data.data(), Q_T.extent(0), L.extent(1));
    leopart::math::matmult<T>(Q_T, L, QT_L);

    quadprogpp::mdMatrix<T> G(QT_Q_data.data(), QT_Q.extent(0), QT_Q.extent(1));
    for (auto& v : QT_L_data)
      v *= -1.0;
    std::vector<T>& g0 = QT_L_data; //.data(), QT_L.extent(0));
    std::vector<T> x(QT_Q.extent(1), 0.0);
    quadprogpp::solve_quadprog(G, g0, CE, ce0, CI, ci0, x);

    const auto& dofs = dm->cell_dofs(c);
    for (int i = 0; i < dofs.size(); ++i)
      expansion_coefficients[dofs[i]] = x[i];
  }
}

} // namespace leopart::transfer
