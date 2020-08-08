// Copyright: (c) 2020 Chris Richardson and Jakob Maljaars
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Field.h"
#include "Particles.h"
#include <dolfinx.h>
#include <iostream>

#pragma once

namespace leopart
{
namespace transfer
{
/// Return basis values for each particle in function space
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
get_particle_contributions(
    const Particles& pax,
    const dolfinx::function::FunctionSpace& function_space);

/// Use basis values to transfer function from field given by value_index to
/// dolfinx Function
template <typename T>
void transfer_to_function(
    std::shared_ptr<dolfinx::function::Function<T>> f, const Particles& pax,
    const Field& field,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        basis_values);

/// Transfer information from the FE \p field to the particles by
/// evaluating dofs at particle positions.
template <typename T>
void transfer_to_particles(
    Particles& pax, Field& field,
    std::shared_ptr<const dolfinx::function::Function<T>> f,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        basis_values);

/// Evaluate the basis functions at particle positions in a prescribed cell.
/// Writes results to an Eigen::Matrix \f$q\f$ and Eigen::Vector \f$f\f$ to
/// compose the l2 problem \f$q q^T = q f\f$.

std::pair<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>,
          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
eval_particle_cell_contributions(
    const std::vector<int>& cell_particles, const Field& field,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        basis_values,
    int row_offset, int space_dimension, int block_size);
//----------------------------------------------------------------------------
template <typename T>
void transfer_to_function(
    std::shared_ptr<dolfinx::function::Function<T>> f, const Particles& pax,
    const Field& field,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        basis_values)
{

  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = f->function_space()->mesh();
  const int tdim = mesh->topology().dim();
  int ncells = mesh->topology().index_map(tdim)->size_local();

  // Get element
  assert(f->function_space()->element());
  std::shared_ptr<const dolfinx::fem::FiniteElement> element
      = f->function_space()->element();
  assert(element);
  const int block_size = element->block_size();
  const int value_size = element->value_size() / block_size;
  const int space_dimension = element->space_dimension() / block_size;
  assert(basis_values.cols() == value_size * space_dimension);

  std::shared_ptr<const dolfinx::fem::DofMap> dm
      = f->function_space()->dofmap();

  // Vector of expansion_coefficients to be set
  Eigen::Matrix<T, Eigen::Dynamic, 1>& expansion_coefficients = f->x()->array();

  int row_offset = 0;
  for (int c = 0; c < ncells; ++c)
  {
    std::vector<int> cell_particles = pax.cell_particles()[c];
    std::pair<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>,
              Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
        ql = eval_particle_cell_contributions(cell_particles, field,
                                              basis_values, row_offset,
                                              space_dimension, block_size);

    // Solve projection where
    // - ql.first --> q has shape [ndofs/block_size, np]
    // - ql.second --> l has shape [np, block_size]
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> u_tmp
        = (ql.first * ql.first.transpose()).ldlt().solve(ql.first * ql.second);
    Eigen::Map<Eigen::VectorXd> u_i(u_tmp.data(), space_dimension * block_size);

    auto dofs = dm->cell_dofs(c);

    assert(dofs.size() == space_dimension);

    for (int i = 0; i < dofs.size(); ++i)
    {
      expansion_coefficients[dofs[i]] = u_i[i];
    }
    row_offset += cell_particles.size();
  }
}
//----------------------------------------------------------------------------
template <typename T>
void transfer_to_particles(
    Particles& pax, Field& field,
    std::shared_ptr<const dolfinx::function::Function<T>> f,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        basis_values)
{
  // Get element
  assert(f->function_space()->element());
  std::shared_ptr<const dolfinx::fem::FiniteElement> element
      = f->function_space()->element();
  assert(element);
  const int block_size = element->block_size();
  const int value_size = element->value_size() / block_size;
  const int space_dimension = element->space_dimension() / block_size;
  assert(basis_values.cols() == value_size * space_dimension);
  assert(field.value_size() == value_size);

  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = f->function_space()->mesh();
  const int tdim = mesh->topology().dim();
  int ncells = mesh->topology().index_map(tdim)->size_local();

  // Get particles in each cell
  const std::vector<std::vector<int>>& cell_particles = pax.cell_particles();

  // Count up particles in each cell
  std::shared_ptr<const dolfinx::fem::DofMap> dm
      = f->function_space()->dofmap();

  // Const array of expansion coefficients
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& f_array = f->x()->array();

  int idx = 0;
  for (int c = 0; c < ncells; ++c)
  {
    auto dofs = dm->cell_dofs(c);
    Eigen::VectorXd vals(dofs.size());
    for (int k = 0; k < dofs.size(); ++k)
    {
      vals[k] = f_array[dofs[k]];
    }
    // Cast as matrix of size [block_size, space_dimension/block_size]
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::ColMajor>>
        vals_mat(vals.data(), block_size, space_dimension);

    for (int pidx : cell_particles[c])
    {
      Eigen::Map<Eigen::VectorXd> ptr = field.data(pidx);
      ptr.setZero();
      Eigen::Map<const Eigen::VectorXd> q(basis_values.row(idx++).data(),
                                          space_dimension);

      ptr = vals_mat * q;
    }
  }
}
//----------------------------------------------------------------------------
std::pair<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>,
          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
eval_particle_cell_contributions(
    const std::vector<int>& cell_particles, const Field& field,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        basis_values,
    int row_offset, int space_dimension, int block_size)
{
  int np = cell_particles.size();
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> q(space_dimension, np),
      l(np, block_size);
  for (int p = 0; p < np; ++p)
  {
    int pidx = cell_particles[p];
    Eigen::Map<const Eigen::VectorXd> basis(
        basis_values.row(row_offset++).data(), space_dimension);
    q.col(p) = basis;
    l.row(p) = field.data(pidx);
  }
  return std::make_pair(q, l);
}
} // namespace transfer
} // namespace leopart
