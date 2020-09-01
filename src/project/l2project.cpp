#include "l2project.h"
#include "../transfer.h"

#include <memory>

using namespace leopart;
using namespace leopart::project;

L2Project::L2Project(
    const Particles& pax,
    std::shared_ptr<dolfinx::function::Function<PetscScalar>> f, std::string w)
    : _particles(std::make_shared<const Particles>(pax)), _f(f),
      _value_size(f->function_space()->element()->value_size()),
      _space_dimension(f->function_space()->element()->space_dimension()),
      _field(std::make_shared<const Field>(pax.field(w))){
          // Throw error if function space not DG (or CG)
      };

void L2Project::solve()
{
  // Set function _f to zero?

  // Evaluate basis functions at particle positions
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      basis_values = transfer::get_particle_contributions(
          *_particles, *_f->function_space());

  // transfer to function
  transfer::transfer_to_function(_f, *_particles, *_field, basis_values);
};

// void L2Project::solve(double l, double u){

// }