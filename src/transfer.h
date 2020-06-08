#include "particles.h"
#include <dolfinx.h>

#pragma once

namespace leopart
{
namespace transfer
{
/// Return basis values
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
get_particle_contributions(
    const particles& pax,
    const dolfinx::function::FunctionSpace& function_space);

void transfer_to_function(
    std::shared_ptr<dolfinx::function::Function> f, const particles& pax,
    int value_index,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        basis_values);

void transfer_to_particles(
    particles& pax, std::shared_ptr<const dolfinx::function::Function> f,
    int value_index,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        basis_values);
} // namespace transfer
} // namespace leopart