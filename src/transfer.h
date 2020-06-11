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
void transfer_to_function(
    std::shared_ptr<dolfinx::function::Function> f, const Particles& pax,
    const Field& field,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        basis_values);

void transfer_to_particles(
    Particles& pax, Field& field,
    std::shared_ptr<const dolfinx::function::Function> f,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        basis_values);
} // namespace transfer
} // namespace leopart