#include "transfer.h"
#include "particles.h"
#include <dolfinx.h>

using namespace leopart;

void transfer::transfer_to_function(
    std::shared_ptr<dolfinx::function::Function> f, const particles& pax,
    int value_index,
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

  dolfinx::la::VecWrapper x(f->vector().vec());

  int idx = 0;
  for (int c = 0; c < ncells; ++c)
  {
    const int np = cell_particles[c].size();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> q(
        space_dimension, np * value_size);
    Eigen::VectorXd f(np * value_size);
    for (int p = 0; p < np; ++p)
    {
      int pidx = cell_particles[c][p];
      Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>
          basis(basis_values.row(idx++).data(), space_dimension, value_size);

      q.block(0, p * value_size, space_dimension, value_size) = basis;
      f.segment(p * value_size, value_size) = pax.data(pidx, value_index);
    }

    Eigen::VectorXd u_i = (q * q.transpose()).ldlt().solve(q * f);
    auto dofs = dm->cell_dofs(c);

    assert(dofs.size() == space_dimension);

    for (int i = 0; i < dofs.size(); ++i)
      x.x[dofs[i]] = u_i[i];
  }
}

void transfer::transfer_to_particles(
    particles& pax, std::shared_ptr<const dolfinx::function::Function> f,
    int value_index,
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
  int field_size = 1;
  for (int q : pax.field_shape(value_index))
    field_size *= q;
  assert(field_size == value_size);

  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = f->function_space()->mesh();
  const int tdim = mesh->topology().dim();
  int ncells = mesh->topology().index_map(tdim)->size_local();

  // Get particles in each cell
  const std::vector<std::vector<int>>& cell_particles = pax.cell_particles();

  // Count up particles in each cell
  std::shared_ptr<const dolfinx::fem::DofMap> dm
      = f->function_space()->dofmap();

  dolfinx::la::VecReadWrapper x(f->vector().vec());

  int idx = 0;
  for (int c = 0; c < ncells; ++c)
  {
    auto dofs = dm->cell_dofs(c);
    Eigen::VectorXd vals(dofs.size());
    for (int k = 0; k < dofs.size(); ++k)
      vals[k] = x.x[dofs[k]];
    for (int pidx : cell_particles[c])
    {
      Eigen::Map<Eigen::VectorXd> ptr = pax.data(pidx, value_index);
      ptr.setZero();
      Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::ColMajor>>
          q(basis_values.row(idx++).data(), value_size, space_dimension);

      ptr = q * vals;
    }
  }
}

Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
transfer::get_particle_contributions(
    const particles& pax,
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
      x.row(i) = pax.data(cell_particles[c][i], 0).head(gdim);

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
