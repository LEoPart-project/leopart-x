// Copyright: (c) 2020 Chris Richardson and Jakob Maljaars
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "transfer.h"
#include "Particles.h"
#include <dolfinx.h>

using namespace leopart;

Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
transfer::get_particle_contributions(
    const Particles& pax,
    const dolfinx::function::FunctionSpace& function_space)
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
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh->geometry().x();

  // Get cell permutation data
  mesh->topology_mutable().create_entity_permutations();
  const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info
      = mesh->topology().get_cell_permutation_info();

  // Get element
  assert(function_space.element());
  std::shared_ptr<const dolfinx::fem::FiniteElement> element
      = function_space.element();
  assert(element);
  const int reference_value_size = element->reference_value_size();
  const int value_size = element->value_size();
  const int space_dimension = element->space_dimension();

  // Prepare geometry data structures

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);

  // Each row represents the contribution from the particle in its cell
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      basis_data(nparticles, space_dimension * value_size);

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
      coordinate_dofs.row(i) = x_g.row(x_dofs[i]).head(gdim);

    // Physical and reference coordinates
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x(
        np, tdim);
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X(
        np, tdim);
    // Prepare basis function data structures
    Eigen::Tensor<double, 3, Eigen::RowMajor> basis_reference_values(
        np, space_dimension, reference_value_size);
    Eigen::Tensor<double, 3, Eigen::RowMajor> basis_values(np, space_dimension,
                                                           value_size);
    Eigen::Tensor<double, 3, Eigen::RowMajor> J(np, gdim, tdim);
    Eigen::Array<double, Eigen::Dynamic, 1> detJ(np);
    Eigen::Tensor<double, 3, Eigen::RowMajor> K(np, tdim, gdim);

    for (int i = 0; i < np; ++i)
      x.row(i) = pax.field(0).data(cell_particles[c][i]).head(gdim);

    cmap.compute_reference_geometry(X, J, detJ, K, x, coordinate_dofs);
    // Compute basis on reference element
    element->evaluate_reference_basis(basis_reference_values, X);
    // Push basis forward to physical element
    element->transform_reference_basis(basis_values, basis_reference_values, X,
                                       J, detJ, K, cell_info[c]);

    // FIXME: avoid copy by using Eigen::TensorMap
    // Copy basis data
    std::copy(basis_values.data(),
              basis_values.data() + np * space_dimension * value_size,
              basis_data.row(p).data());
    p += np;
  }

  return basis_data;
}
