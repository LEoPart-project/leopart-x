// Copyright: (c) 2020 Chris Richardson and Jakob Maljaars
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Field.h"
#include "Particles.h"

// #include <dolfinx.h>
#include <iostream>

#pragma once

namespace leopart::transfer
{
// /// Return basis values for each particle in function space
// Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
// get_particle_contributions(
//     const Particles& pax,
//     const dolfinx::function::FunctionSpace& function_space);

// /// Use basis values to transfer function from field given by value_index to
// /// dolfinx Function
// template <typename T>
// void transfer_to_function(
//     std::shared_ptr<dolfinx::function::Function<T>> f, const Particles& pax,
//     const Field& field,
//     const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
//         basis_values);

// /// Transfer information from the FE \p field to the particles by
// /// evaluating dofs at particle positions.
// template <typename T>
// void transfer_to_particles(
//     Particles& pax, Field& field,
//     std::shared_ptr<const dolfinx::function::Function<T>> f,
//     const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
//         basis_values);

// /// Evaluate the basis functions at particle positions in a prescribed cell.
// /// Writes results to an Eigen::Matrix \f$q\f$ and Eigen::Vector \f$f\f$ to
// /// compose the l2 problem \f$q q^T = q f\f$.
// std::pair<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>,
//           Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
// eval_particle_cell_contributions(
//     const std::vector<int>& cell_particles, const Field& field,
//     const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
//         basis_values,
//     int row_offset, int space_dimension, int block_size);
} // namespace leopart::transfer
