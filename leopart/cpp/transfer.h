// Copyright (c) 2020-2023 Chris Richardson, Jakob Maljaars and Nathan Sime
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Field.h"
#include "Particles.h"
#include "external/quadprog_mdspan/QuadProg++.hh"
#include "math.h"
#include "utils.h"

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
/// @param particles Particles collection
/// @param field Field data into which to store the interpolation
/// @param f The finite element function to be interpolated
/// @param tol Tolerance of the Newton solver used in non-affine pull-backs
/// @param maxit Maximum Newton iterations used in non-affine pull-backs
template <dolfinx::scalar T>
void transfer_to_particles(const Particles<T>& particles, Field<T>& field,
                           std::shared_ptr<const dolfinx::fem::Function<T>> f,
                           const double tol = 1.0e-6, const int maxit = 15)
{
  const std::vector<std::int32_t>& p2c = particles.particle_to_cell();

  const std::span<const T> x = particles.x().data();
  const std::array<std::size_t, 2> xshape
      = {particles.x().size(), particles.x().value_size()};

  const std::span<T> u = field.data();
  const std::array<std::size_t, 2> ushape = {field.size(), field.value_size()};

  f->eval(x, xshape, p2c, u, ushape, tol, maxit);
}

/// Transfer the provided particle field data to the finite element
/// function using local l_2 projection.
/// We solve the problem: find \f(u_h \in V\f) such that
///
/// \f[
///    u_h(x_p) v(x_p) = u_p v(x_p) \quad \forall v \in V, \; p = 1,\ldots,n_p.
/// \f]
///
/// Here \f(u_p\f) is the \f(p\f)th particle's data, \f(x_p\f) is the \f(p\f)th
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
std::vector<std::int32_t> find_deficient_cells(
  std::shared_ptr<dolfinx::fem::Function<T>> f,
  const Particles<T>& pax)
{
  std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh
    = f->function_space()->mesh();
  std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> V
    = f->function_space();
  const int tdim = mesh->topology()->dim();
  std::int32_t ncells = mesh->topology()->index_map(tdim)->size_local();

  const std::vector<std::vector<std::size_t>>& c2p
    = pax.cell_to_particle();

  std::vector<std::int32_t> deficient_cells;
  for (std::int32_t c = 0; c < ncells; ++c)
  {
    const std::size_t ncdof = V->dofmap()->cell_dofs(c).size();
    const std::size_t ncp
      = (static_cast<std::size_t>(c) < c2p.size()) ? c2p[c].size() : 0;
    if (ncp < ncdof)
      deficient_cells.push_back(c);
  }
  return deficient_cells;
}

/// Throw unless cell-local l_2 projection of the particle field onto the
/// space is well defined. The projection solves one small problem per
/// cell, so it needs:
///  - a scalar-valued basis, possibly blocked (e.g. a vector DG space);
///  - DoFs that belong to a single cell. DoFs shared between cells, as
///    in a continuous space, would take the value from whichever cell
///    was solved last;
///  - a field value size equal to the space's block size.
///
/// @param V The function space to project onto
/// @param field The particle field data to be projected
template <dolfinx::scalar T, std::floating_point U>
void check_l2_projection(const dolfinx::fem::FunctionSpace<U>& V,
                         const Field<T>& field)
{
  std::shared_ptr<const dolfinx::fem::FiniteElement<U>> element
      = V.element();
  assert(element);
  const int block_size = element->block_size();

  if (element->num_sub_elements() > 1
      and element->num_sub_elements() != block_size)
  {
    throw std::runtime_error(
        "Projection onto mixed spaces is not supported. Project onto each "
        "subspace instead.");
  }

  if (element->reference_value_size() != 1)
  {
    throw std::runtime_error(
        "Projection requires a scalar-valued basis, optionally blocked "
        "(e.g. a vector DG space). Spaces with a vector-valued basis, such "
        "as Raviart-Thomas or Nedelec, are not supported.");
  }

  // DoFs on a vertex, edge or facet of the cell are shared with
  // neighbouring cells
  const std::vector<dolfinx::mesh::CellType> cell_types
      = V.mesh()->topology()->cell_types();
  if (cell_types.size() != 1)
    throw std::runtime_error("Mixed topology not supported");
  const dolfinx::mesh::CellType cell_type = cell_types.front();
  const int tdim = dolfinx::mesh::cell_dim(cell_type);
  const dolfinx::fem::ElementDofLayout& layout
      = V.dofmap()->element_dof_layout();
  for (int d = 0; d < tdim; ++d)
  {
    const int num_entities = dolfinx::mesh::cell_num_entities(cell_type, d);
    for (int e = 0; e < num_entities; ++e)
    {
      if (!layout.entity_dofs(d, e).empty())
      {
        throw std::runtime_error(
            "Projection requires a discontinuous space (e.g. DG): it "
            "solves cell by cell, so DoFs shared between cells would take "
            "the value from whichever cell was solved last.");
      }
    }
  }

  if (field.value_size() != static_cast<std::size_t>(block_size))
  {
    throw std::runtime_error(
        "Particle field \"" + field.name + "\" has value size "
        + std::to_string(field.value_size())
        + ", which does not match the function space's value size "
        + std::to_string(block_size) + ".");
  }
}

/// Transfer the provided particle field data to the finite element
/// function using local l_2 projection. The solve_callback function
/// returns \f(x\f) which solves (as required by the user)
///
/// \f[
///    Q^T Q x = Q^T L.
/// \f]
///
/// where \f(Q = u_h(x_p)\f), \f(Q^T = v(x_p)\f) \f(L = u_p\f),
/// \f(u_p\f) is the \f(p\f)th particle's data, \f(x_p\f) is the \f(p\f)th
/// particle's position.
///
///
/// @tparam T The function scalar type
/// @tparam U The function geometry type
/// @param f The finite element function
/// @param pax The particles collection
/// @param field The field data to be transferred
template <dolfinx::scalar T, std::floating_point U>
void transfer_to_function_l2_callback(
  std::shared_ptr<dolfinx::fem::Function<T>> f,
  const Particles<T>& pax, const Field<T>& field,
  std::function<const std::vector<T>(mdspan_t<T, 2>, mdspan_t<T, 2>)> solve_callback)
{
  check_l2_projection<T, U>(*f->function_space(), field);

  std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh
      = f->function_space()->mesh();
  const int tdim = mesh->topology()->dim();
  std::int32_t ncells = mesh->topology()->index_map(tdim)->size_local();

  // Get element
  assert(f->function_space()->element());
  std::shared_ptr<const dolfinx::fem::FiniteElement<U>> element
      = f->function_space()->element();
  assert(element);

  // @todo these definitions are legacy DOLFIN and should be refactored for
  // appropriate unrolling of DoFs.
  const int block_size = element->block_size();
  const std::size_t space_dimension = element->space_dimension() / block_size;

  std::shared_ptr<const dolfinx::fem::DofMap> dm
      = f->function_space()->dofmap();

  // Vector of expansion_coefficients to be set
  std::span<T> expansion_coefficients = f->x()->array();
  const std::vector<std::vector<std::size_t>>& cell_to_particle
      = pax.cell_to_particle();

  // Basis evaluations, shape (np, space_dimension, value_size)
  const auto [basis_evals, basis_shape]
      = leopart::utils::evaluate_basis_functions<U>(
          *f->function_space(), pax.x().data(), pax.x().value_size(),
          pax.particle_to_cell());
  const mdspan_t<const T, 3> basis_evals_md(basis_evals.data(), basis_shape);

  // Assemble and solve Q^T Q u = Q^T L in each cell, where
  // Q = \phi(x_p), Q^T = \psi(x_p), L = u_p,
  // \phi is the trial function, \psi is the test function, x_p
  // are particles' position and u_p are particles' datum/data.
  // Dimensions:
  //  Q: (n_p x space_dimension)
  //  QT: (space_dimension x n_p)
  //  QT . Q: (space_dimension x space_dimension)
  //  L: (n_p x block_size)
  //  QT . L: (space_dimension x block_size)
  const std::vector<std::size_t> no_particles;
  for (std::int32_t c = 0; c < ncells; ++c)
  {
    const std::vector<std::size_t>& cell_particles
      = (static_cast<std::size_t>(c) < cell_to_particle.size())
        ? cell_to_particle[c] : no_particles;
    const std::size_t cell_np = cell_particles.size();

    // Assemble Q
    std::vector<T> Q_T_data(cell_np * space_dimension);
    mdspan_t<T, 2> Q_T(Q_T_data.data(), space_dimension, cell_np);

    std::vector<T> Q_data(cell_np * space_dimension);
    mdspan_t<T, 2> Q(Q_data.data(), cell_np, space_dimension);
    for (std::size_t cell_p = 0; cell_p < cell_np; ++cell_p)
    {
      const std::size_t p_idx = cell_particles[cell_p];
      for (std::size_t i = 0; i < space_dimension; ++i)
        Q(cell_p, i) = basis_evals_md(p_idx, i, 0); // Scalar basis, checked above
    }
    leopart::math::transpose<T>(Q, Q_T);

    // Assemble L. Each column corresponds to a block's DoFs
    std::vector<T> L_data(cell_np * block_size);
    mdspan_t<T, 2> L(L_data.data(), cell_np, block_size);
    for (std::size_t cell_p = 0; cell_p < cell_np; ++cell_p)
    {
      const std::size_t p_idx = cell_particles[cell_p];
      for (int b = 0; b < block_size; ++b)
        L(cell_p, b) = field.data()[p_idx * block_size + b];
    }

    // Solve element local l2 minimisation. Solve for each column in L.
    std::vector<T> QT_Q_data(Q_T.extent(0) * Q.extent(1));
    mdspan_t<T, 2> QT_Q(QT_Q_data.data(), Q_T.extent(0), Q.extent(1));
    leopart::math::matmult<T>(Q_T, Q, QT_Q);

    std::vector<T> QT_L_data(Q_T.extent(0) * L.extent(1));
    mdspan_t<T, 2> QT_L(QT_L_data.data(), Q_T.extent(0), L.extent(1));
    leopart::math::matmult<T>(Q_T, L, QT_L);

    const std::vector<T> soln = solve_callback(QT_Q, QT_L);

    // Populate FE function DoFs
    const auto& dofs = dm->cell_dofs(c);
    for (std::size_t i = 0; i < dofs.size(); ++i)
      for (int k = 0; k < block_size; ++k)
        expansion_coefficients[dofs[i]*block_size + k] = soln[i*block_size + k];
  }
}

/// Transfer the provided particle field data to the finite element
/// function using local l_2 projection.
/// We solve the problem: find \f(u_h \in V\f) such that
///
/// \f[
///    u_h(x_p) v(x_p) = u_p v(x_p) \quad \forall v \in V, \; p = 1,\ldots,n_p.
/// \f]
///
/// Here \f(u_p\f) is the \f(p\f)th particle's data, \f(x_p\f) is the \f(p\f)th
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
  // Simply solve the particle mass matrix / rhs system
  std::function<const std::vector<T>(mdspan_t<T, 2>, mdspan_t<T, 2>)>
   solve_function = [](mdspan_t<T, 2> QT_Q, mdspan_t<T, 2> QT_L)
  {
    const std::vector<T> soln = basix::math::solve<T>(QT_Q, QT_L);
    return soln;
  };
  transfer_to_function_l2_callback<T, U>(f, pax, field, solve_function);
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
  assert(f->function_space()->element());
  std::shared_ptr<const dolfinx::fem::FiniteElement<U>> element
      = f->function_space()->element();
  assert(element);
  const int block_size = element->block_size();
  if (block_size != 1)
  {
    // The constraints are built for one block only
    throw std::runtime_error(
        "Constrained projection supports only spaces with block size 1 "
        "(scalar spaces), got block size " + std::to_string(block_size)
        + ".");
  }
  const int value_size = element->value_size() / block_size;
  const int space_dimension = element->space_dimension() / block_size;

  // QuadProg specifics for constraints
  // CE^T x + ce0 =  0
  // CI^T x + ci0 >= 0
  std::vector<T> CE_data(space_dimension, 0.0),
      CI_data(space_dimension * space_dimension * value_size * 2, 0.0);
  quadprogpp::mdMatrix<T> CE(CE_data.data(), space_dimension, 0);
  quadprogpp::mdMatrix<T> CI(CI_data.data(), space_dimension,
                             space_dimension * value_size * 2);
  std::vector<T> ce0(0, 0.0), ci0(space_dimension * value_size * 2, 0.0);

  for (int i = 0; i < space_dimension; i++)
  {
    CI(i, i) = 1.;
    CI(i, i + space_dimension) = -1;
    ci0[i] = -l;
    ci0[i + space_dimension] = u;
  }

  // Solve the  mass matrix / rhs optimisation problem with quadprog
  std::function<std::vector<T>(mdspan_t<T, 2>, mdspan_t<T, 2>)>
    solve_function = [&CE, &ce0, &CI, &ci0](
      mdspan_t<T, 2> QT_Q, mdspan_t<T, 2> QT_L)
  {
    std::span<T> g0(QT_L.data_handle(), QT_L.extent(0));
    for (auto& v : g0)
      v *= -1.0;
    std::vector<T> x(QT_Q.extent(1), 0.0);
    quadprogpp::solve_quadprog(QT_Q, g0, CE, ce0, CI, ci0, x);
    return x;
  };

  transfer_to_function_l2_callback<T, U>(f, pax, field, solve_function);
}
} // namespace leopart::transfer
