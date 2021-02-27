// Copyright: (c) 2020 Chris Richardson and Jakob Maljaars
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "transfer.h"
#include "Particles.h"
#include <dolfinx.h>
// Stick to eigen for the time being
#include <unsupported/Eigen/CXX11/Tensor>

using namespace leopart;

Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
transfer::get_particle_contributions(
    const Particles& pax, const dolfinx::fem::FunctionSpace& function_space)
{
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = function_space.mesh();
  const int gdim = mesh->geometry().dim();
  const int tdim = mesh->topology().dim();

  // Count up particles in each cell
  int ncells = mesh->topology().index_map(tdim)->size_local();

  const std::vector<std::vector<int>>& cell_particles = pax.cell_particles();
  int nparticles = 0;
  for (const std::vector<int>& q : cell_particles)
    nparticles += q.size();

  // Get coordinate map
  const dolfinx::fem::CoordinateElement& cmap = mesh->geometry().cmap();

  // Get geometry data
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();
  const int num_dofs_g = x_dofmap.num_links(0);
  const dolfinx::array2d<double>& x_g = mesh->geometry().x();

  // Get cell permutation data
  mesh->topology_mutable().create_entity_permutations();
  const std::vector<std::uint32_t>& cell_info
      = mesh->topology().get_cell_permutation_info();

  // Get element
  assert(function_space.element());
  std::shared_ptr<const dolfinx::fem::FiniteElement> element
      = function_space.element();
  assert(element);
  const int block_size = element->block_size();
  const int reference_value_size = element->reference_value_size() / block_size;
  const int value_size = element->value_size() / block_size;
  const int space_dimension = element->space_dimension() / block_size;

  // Prepare geometry data structures
  // Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  //     coordinate_dofs(num_dofs_g, gdim);
  dolfinx::array2d<double> coordinate_dofs(num_dofs_g, gdim);

  // Each row represents the contribution from the particle in its cell
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      basis_data(nparticles, space_dimension * value_size);

  // Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  //     basis_data(space_dimension * value_size, nparticles);

  int p = 0;
  for (int c = 0; c < ncells; ++c)
  {
    int np = cell_particles[c].size();
    if (np == 0)
    {
      std::cout << "WARNING: no particles in cell " << c << "\n";
      continue;
    }
    // Get cell geometry (coordinate dofs)
    auto x_dofs = x_dofmap.links(c);
    for (int i = 0; i < num_dofs_g; ++i)
      // TODO: make more efficient?
      for (int j = 0; j < gdim; ++j)
      {
        coordinate_dofs(i, j) = x_g(x_dofs[i], j);
      }
    // coordinate_dofs.row(i) = x_g.row(x_dofs[i]); // .head(gdim);

    // Physical and reference coordinates
    // Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x(
    //     np, tdim);
    // Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X(
    //     np, tdim);
    // Prepare basis function data structures
    // Eigen::Tensor<double, 3, Eigen::RowMajor> basis_reference_values(
    //     np, space_dimension, reference_value_size);
    // Eigen::Tensor<double, 3, Eigen::RowMajor> basis_values(np,
    // space_dimension,
    //                                                        value_size);
    // Eigen::Tensor<double, 3, Eigen::RowMajor> J(np, gdim, tdim);
    // Eigen::Array<double, Eigen::Dynamic, 1> detJ(np);
    // Eigen::Tensor<double, 3, Eigen::RowMajor> K(np, tdim, gdim);

    // Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x(
    //     np, tdim);
    // Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X(
    //     np, tdim);
    dolfinx::array2d<double> X(np, tdim);
    dolfinx::array2d<double> x(np, tdim);

    std::vector<double> basis_reference_values(np * space_dimension
                                               * reference_value_size);
    std::vector<double> basis_values(np * space_dimension
                                     * reference_value_size);
    std::vector<double> J(np * gdim * tdim);
    std::vector<double> detJ(np);
    std::vector<double> K(np * tdim * gdim);

    // double* ptr = &X(0, 0);
    // Eigen::Map<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
    // Eigen::RowMajor>> X_eigen(ptr); Fill particle coordinates
    for (int i = 0; i < np; ++i)
      // TODO: avoid this ugly copy
      for (int j = 0; j < tdim; ++j)
        x(i, j) = pax.field(0).data(cell_particles[c][i])[j];
    // x.row(i) = pax.field(0).data(cell_particles[c][i]).head(gdim);

    cmap.compute_reference_geometry(X, J, detJ, K, x, coordinate_dofs);
    // cmap.compute_reference_geometry(X_eigen, J, detJ, K, x, coordinate_dofs);
    // Compute basis on reference element
    element->evaluate_reference_basis(basis_reference_values, X);
    // Push basis forward to physical element
    // element->transform_reference_basis(basis_values, basis_reference_values,
    // X,
    //                                    J, detJ, K, cell_info[c]);
    element->transform_reference_basis(basis_values, basis_reference_values, X,
                                       J, detJ, K);

    // FIXME: avoid copy by using Eigen::TensorMap
    // Copy basis data
    for (const auto& val : basis_values)
    {
      std::cout << "Value " << val << std::endl;
    }
    std::cout << "=======" << std::endl;
    // std::copy(basis_values.data(),
    //           basis_values.data() + np * space_dimension * value_size,
    //           basis_data.col(p).data());

    for (size_t j = 0; j < space_dimension * value_size; ++j)
    {
      for (size_t i = 0; i < np; ++i)
      {
        basis_data(i + p, j) = basis_values[i + np * j];
      }
    }
    p += np;
    std::cout << "Pvalue " << p << std::endl;
  }
  std::cout << basis_data << std::endl;
  return basis_data;
}
//----------------------------------------------------------------------------
template <typename T>
void transfer::transfer_to_function(
    std::shared_ptr<dolfinx::fem::Function<T>> f, const Particles& pax,
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
  // Eigen::Matrix<T, Eigen::Dynamic, 1>&
  std::vector<T>& expansion_coefficients = f->x()->mutable_array();

  int row_offset = 0;
  for (int c = 0; c < ncells; ++c)
  {
    std::vector<int> cell_particles = pax.cell_particles()[c];
    auto [q, l] = eval_particle_cell_contributions(cell_particles, field,
                                                   basis_values, row_offset,
                                                   space_dimension, block_size);

    // Solve projection where
    // - q has shape [ndofs/block_size, np]
    // - l has shape [np, block_size]
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> u_tmp
        = (q * q.transpose()).ldlt().solve(q * l);
    Eigen::Map<Eigen::VectorXd> u_i(u_tmp.data(), space_dimension * block_size);
    std::cout << "U_i.size() " << u_i.size() << std::endl;
    auto dofs = dm->cell_dofs(c);

    assert(dofs.size() == space_dimension);

    // for (int i = 0; i < dofs.size(); ++i)
    // {
    //   expansion_coefficients[dofs[i]] = u_i[i];
    // }

    for (int k = 0; k < dofs.size(); ++k)
    {
      for (int l = 0; l < block_size; ++l)
      {
        expansion_coefficients[dofs[k] * block_size + l]
            = u_i[k * block_size + l];
      }
    }

    row_offset += cell_particles.size();
  }
}
//----------------------------------------------------------------------------
template <typename T>
void transfer::transfer_to_particles(
    Particles& pax, Field& field,
    std::shared_ptr<const dolfinx::fem::Function<T>> f,
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
  // const Eigen::Matrix<T, Eigen::Dynamic, 1>& f_array = f->x()->array();
  const std::vector<T>& f_array = f->x()->array();
  std::cout << "F size " << f_array.size() << std::endl;

  int idx = 0;
  for (int c = 0; c < ncells; ++c)
  {
    auto dofs = dm->cell_dofs(c);
    // std::cout << "dofs size "<<dofs.size()<<std::endl;
    // Eigen::VectorXd vals(dofs.size());
    // for (int k = 0; k < dofs.size(); ++k)
    // {
    //   vals[k] = f_array[dofs[k]];
    // }

    // UPDATED
    Eigen::VectorXd vals(dofs.size() * block_size);
    for (int k = 0; k < dofs.size(); ++k)
    {
      for (int l = 0; l < block_size; ++l)
      {
        std::cout << "Dof number " << dofs[k] * block_size + l << "\nValue "
                  << f_array[dofs[k] * block_size + l] << std::endl;
        vals[k * block_size + l] = f_array[dofs[k] * block_size + l];
      }
    }

    // Python, remove
    // cell_blocks = Q.dofmap.cell_dofs(i)
    // for (j,dof) in enumerate(cell_blocks):
    //     for k in range(block_size):
    //         local_cell_dofs[i, j*block_size+k] = dof*block_size + k

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

      std::cout << "Vals mat " << vals_mat << std::endl;
      std::cout << "[JM] q here " << q << std::endl;

      ptr = vals_mat * q;
      std::cout << "[JM] printing value at particle " << pidx
                << "position: " << pax.field(0).data(pidx) << " value: " << ptr
                << std::endl;
    }
  }
}
//----------------------------------------------------------------------------
std::pair<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>,
          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
transfer::eval_particle_cell_contributions(
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

// Explicit instantiation of template functions needed, see
// https://stackoverflow.com/questions/495021/why-can-templates-only-be-implemented-in-the-header-file
template void transfer::transfer_to_particles<>(
    Particles&, Field&,
    std::shared_ptr<const dolfinx::fem::Function<PetscScalar>>,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>&);

template void transfer::transfer_to_function<>(
    std::shared_ptr<dolfinx::fem::Function<PetscScalar>>, const Particles&,
    const Field&,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>&);
