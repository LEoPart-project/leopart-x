// Copyright: (c) 2020 Chris Richardson and Jakob Maljaars
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Particles.h"

// #include <dolfinx.h>
#include <memory>

namespace leopart
{
namespace project
{

// class L2Project
// {
// public:
//   L2Project(const Particles& pax,
//             std::shared_ptr<dolfinx::function::Function<PetscScalar>> f,
//             std::string w);

//   /// Solve l2 projection problem
//   void solve();

//   /// Solve l2 projection with box constraints with @param l the lower
//   /// bound and @param u the upper bound
//   void solve(double l, double u);

// private:
//   std::shared_ptr<const Particles> _particles;
//   std::shared_ptr<dolfinx::function::Function<PetscScalar>> _f;
//   std::shared_ptr<const Field> _field;

//   std::size_t _value_size, _space_dimension;
// };
} // namespace project
} // namespace leopart