// Copyright: (c) 2020 Chris Richardson and Jakob Maljaars
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "../Particles.h"
#include <dolfinx.h>

namespace leopart
{
namespace project
{

class l2project
{
public:
  l2project(const Particles& pax,
            std::shared_ptr<dolfinx::function::Function<PetscScalar>> f);

  void solve();

private:
  std::shared_ptr<const Particles> _P;
  std::shared_ptr<dolfinx::function::Function<PetscScalar>> _f;

  // Probably not needed as an attribute
  // std::shared_ptr<const dolfinx::fem::FiniteElement> _element;

  const std::size_t _value_size, _space_dimension;
  // = f->function_space()->element();
};
} // namespace project
} // namespace leopart