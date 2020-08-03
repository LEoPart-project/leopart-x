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

/// Evaluate the basis functions at particle positions in a prescribed cell.
/// Writes results to an Eigen::Matrix \f$q\f$ and Eigen::Vector \f$f\f$ to
/// compose the l2 problem \f$q q^T = q f\f$.
void eval_particle_cell_contributions(
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& q,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& l,
    int* row_idx, const Particles& pax, const Field& field, const int cidx,
    const int value_size, const int space_dimension, const int block_size,
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

  int row_idx = 0;
  for (int c = 0; c < ncells; ++c)
  {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> q;
    // Eigen::VectorXd l;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> l;
    eval_particle_cell_contributions(q, l, &row_idx, pax, field, c, value_size,
                                     space_dimension, block_size, basis_values);

    // Solve projection
    // Eigen::VectorXd u_i = (q * q.transpose()).ldlt().solve(q * l);
    // Eigen::VectorXd u_i = (q.transpose() * q).inverse() * (q.transpose() *
    // l);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> u_tmp
        = ((q.transpose() * q).inverse() * (q.transpose() * l));
    // Eigen::VectorXd u_i(Eigen::Map<Eigen::VectorXd>(u_tmp.data(),
    // u_tmp.cols()*u_tmp.rows()));
    Eigen::Map<Eigen::VectorXd> u_i(u_tmp.data(), space_dimension * block_size);
    if (c == 0)
    {
      std::cout << "This was q in cell 0 \n" << q << std::endl;
      std::cout << "This was l in cell 0 \n" << l << std::endl;
      std::cout << "Solution cell 0 is \n" << u_i << std::endl;
    }
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
    // Cast
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::ColMajor>>
        vals_mat(vals.data(), block_size, space_dimension);

    for (int pidx : cell_particles[c])
    {
      Eigen::Map<Eigen::VectorXd> ptr = field.data(pidx);
      ptr.setZero();
      // Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
      //                                Eigen::ColMajor>>
      //     q(basis_values.row(idx++).data(), value_size, space_dimension);
      Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::ColMajor>>
          q(basis_values.row(idx++).data(), space_dimension, value_size);
      // Eigen::Map<const Eigen::VectorXd>
      //     q(basis_values.row(idx++).data());
      // ptr = q * vals;
      ptr = vals_mat * q;
      if (c == 0 && pidx == 0)
      {
        std::cout << "IDX IS" << idx << std::endl;
        // std::cout << "Casted to \n"<< q << std::endl;
        std::cout << "Vals \n" << vals << std::endl;
        std::cout << "Ptr result \n" << ptr << std::endl;
      }
    }
  }
}
//----------------------------------------------------------------------------
void eval_particle_cell_contributions(
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& q,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& l,
    int* row_idx, const Particles& pax, const Field& field, const int cidx,
    const int value_size, const int space_dimension, int block_size,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        basis_values)
{
  // Get particles in cell cidx
  const std::vector<int>& cell_particles = pax.cell_particles(cidx);
  int np = cell_particles.size();
  // q.resize(space_dimension, np * value_size);
  // l.resize(np * value_size);
  q.resize(np, space_dimension);
  // l.resize(np * value_size);
  l.resize(np, block_size);
  if (cidx == 0)
  {
    std::cout << "Num particles " << np << std::endl;
  }
  for (int p = 0; p < np; ++p)
  {
    int pidx = cell_particles[p];
    Eigen::Map<const Eigen::VectorXd> basis(
        basis_values.row((*row_idx)++).data(), space_dimension);
    // Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
    //                                Eigen::RowMajor>>
    //     basis(basis_values.row((*row_idx)++).data(), 1, space_dimension);

    // q.block(p, 0, 1, space_dimension) = basis;
    q.row(p) = basis;
    // q.block(0, p * value_size, space_dimension, value_size) = basis;
    // l.segment(p * value_size, value_size) = field.data(pidx);
    // l.block(p, 0, 1, block_size) = Eigen::Map<field.data(pidx);
    l.row(p) = field.data(pidx);
    if (cidx == 0 && (p == 0 || p == 1))
    {
      std::cout << "Particle value " << p << "\n"
                << field.data(pidx) << std::endl;
    }
  }
}
} // namespace transfer
} // namespace leopart
