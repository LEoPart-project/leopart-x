#include "l2project.h"
#include "../transfer.h"

using namespace leopart;
using namespace leopart::project;

l2project::l2project(
    const Particles& pax,
    std::shared_ptr<dolfinx::function::Function<PetscScalar>> f)
    : _P(&pax), _f(f),
      _value_size(f->function_space()->element()->value_size()),
      _space_dimension(f->function_space()->element()->space_dimension()){

          // Throw error if function space not DG (or CG)

      };

void l2project::solve()
{
  // Set function to zero?

  // Evaluate basis functions at particle positions
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      basis_values
      = transfer::get_particle_contributions(*_P, *_f->function_space());

  // transfer to particles
};