// Copyright: (c) 2020 Jakob Maljaars, Chris Richardson and Nate Sime
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Particles.h"

#include <iostream>
#include <memory>
#include <vector>

namespace dolfinx
{
namespace mesh
{
class Mesh;
}
} // namespace dolfinx

namespace leopart
{
namespace advect
{
enum class facet_type : std::uint8_t
{
  internal,
  closed,
  open,
  periodic,
  bounded
};

struct facet_info
{
  // Point midpoint;
  // Point normal;
  facet_type type;
};

// Advection related code should go into leopart::advect namespace
class Advect
{
public:
  Advect(std::shared_ptr<Particles>& particles,
         const std::shared_ptr<dolfinx::mesh::Mesh>& mesh);

private:
  void set_facet_info();

  std::shared_ptr<Particles> _particles;
  const std::shared_ptr<dolfinx::mesh::Mesh> _mesh;
  std::vector<facet_info> _facet_info;
};

} // namespace advect
} // namespace leopart