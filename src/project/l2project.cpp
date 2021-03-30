#include "l2project.h"
#include "../external/quadprogpp/QuadProg++.hh"
#include "../transfer.h"

#include <Eigen/Dense>
#include <memory>

using namespace leopart;
using namespace leopart::project;

L2Project::L2Project(const Particles& pax,
                     std::shared_ptr<dolfinx::fem::Function<PetscScalar>> f,
                     std::string w)
    : _particles(std::make_shared<const Particles>(pax)), _f(f),
      _value_size(f->function_space()->element()->value_size()),
      _space_dimension(f->function_space()->element()->space_dimension()),
      _field(std::make_shared<const Field>(pax.field(w))){
          // Throw error if function space not DG (or CG)
      };

void L2Project::solve()
{
  // Evaluate basis functions at particle positions
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      basis_values = transfer::get_particle_contributions(
          *_particles, *_f->function_space());

  // transfer to function
  transfer::transfer_to_function(_f, *_particles, *_field, basis_values);
};

// TODO: template so as to make generic for arbitrary value size of the function
// space
void L2Project::solve(double l, double u)
{
  // Currently only scalar function spaces accepted
  if (_value_size > 1)
    throw std::runtime_error(
        "Bounded projection is implemented for scalar functions only");

  if (l > u)
  {
    throw std::runtime_error("Lower boundary cannot exceed upper boundary in "
                             "constrained projection");
  }

  // Get element
  assert(_f->function_space()->element());
  std::shared_ptr<const dolfinx::fem::FiniteElement> element
      = _f->function_space()->element();
  assert(element);
  const int block_size = element->block_size();
  const int value_size = _value_size / block_size;
  const int space_dimension = _space_dimension / block_size;

  // Initialize the matrices/vectors for the bound constraints (constant
  // throughout projection)
  Eigen::MatrixXd CE, CI;
  Eigen::VectorXd ce0, ci0;

  CE.resize(space_dimension, 0);
  ce0.resize(0);

  CI.resize(space_dimension, space_dimension * value_size * 2);
  CI.setZero();
  ci0.resize(space_dimension * value_size * 2);
  ci0.setZero();
  for (std::size_t i = 0; i < space_dimension; i++)
  {
    CI(i, i) = 1.;
    CI(i, i + space_dimension) = -1;
    ci0(i) = -l;
    ci0(i + space_dimension) = u;
  }

  // Evaluate basis functions at particle positions
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      basis_values = transfer::get_particle_contributions(
          *_particles, *_f->function_space());

  // Get number of cells local to process
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh
      = _f->function_space()->mesh();
  const int tdim = mesh->topology().dim();
  int ncells = mesh->topology().index_map(tdim)->size_local();

  std::shared_ptr<const dolfinx::fem::DofMap> dm
      = _f->function_space()->dofmap();

  // Vector of expansion_coefficients to be set
  std::vector<PetscScalar>& expansion_coefficients = _f->x()->mutable_array();

  int row_offset = 0;
  for (int c = 0; c < ncells; ++c)
  {
    std::vector<int> cell_particles = _particles->cell_particles()[c];
    auto [q, l] = transfer::eval_particle_cell_contributions(
        cell_particles, *_field, basis_values, row_offset, space_dimension,
        block_size);

    // Solve bounded lstsq projection
    Eigen::MatrixXd AtA = q * (q.transpose());
    Eigen::VectorXd Atf = -q * l;

    Eigen::VectorXd u_i;
    quadprogpp::solve_quadprog(AtA, Atf, CE, ce0, CI, ci0, u_i);
    auto dofs = dm->cell_dofs(c);

    assert(dofs.size() == space_dimension);

    for (int i = 0; i < dofs.size(); ++i)
    {
      expansion_coefficients[dofs[i]] = u_i[i];
    }
    row_offset += cell_particles.size();
  }
}