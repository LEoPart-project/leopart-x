// Copyright: (c) 2020 Chris Richardson and Jakob Maljaars
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Field.h"
#include "Particles.h"
#include <dolfinx.h>

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

  // Get particles in each cell
  const std::vector<std::vector<int>>& cell_particles = pax.cell_particles();

  // Get element
  assert(f->function_space()->element());
  std::shared_ptr<const dolfinx::fem::FiniteElement> element
      = f->function_space()->element();
  assert(element);
  const int value_size = element->value_size();
  const int space_dimension = element->space_dimension();
  assert(basis_values.cols() == value_size * space_dimension);

  std::shared_ptr<const dolfinx::fem::DofMap> dm
      = f->function_space()->dofmap();

  // Vector of expansion_coefficients to be set
  Eigen::Matrix<T, Eigen::Dynamic, 1>& expansion_coefficients = f->x()->array();

  int idx = 0;
  for (int c = 0; c < ncells; ++c)
  {
    const int np = cell_particles[c].size();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> q(
        space_dimension, np * value_size);
    Eigen::VectorXd l(np * value_size);
    for (int p = 0; p < np; ++p)
    {
      int pidx = cell_particles[c][p];
      Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>
          basis(basis_values.row(idx++).data(), space_dimension, value_size);

      q.block(0, p * value_size, space_dimension, value_size) = basis;
      l.segment(p * value_size, value_size) = field.data(pidx);
    }

    Eigen::VectorXd u_i = (q * q.transpose()).ldlt().solve(q * l);
    auto dofs = dm->cell_dofs(c);

    assert(dofs.size() == space_dimension);

    for (int i = 0; i < dofs.size(); ++i)
    {
      expansion_coefficients[dofs[i]] = u_i[i];
    }
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
  const int value_size = element->value_size();
  const int space_dimension = element->space_dimension();
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
    for (int pidx : cell_particles[c])
    {
      Eigen::Map<Eigen::VectorXd> ptr = field.data(pidx);
      ptr.setZero();
      Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::ColMajor>>
          q(basis_values.row(idx++).data(), value_size, space_dimension);

      ptr = q * vals;
    }
  }
}
} // namespace transfer
} // namespace leopart