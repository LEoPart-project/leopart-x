// Functionality that was temporarily (?!) deleted from dolfinx

#pragma once

#include <Eigen/Core>
#include <dolfinx.h>

namespace leopart
{
namespace common
{

/// Compute (generalized) volume of mesh entities of given dimension
Eigen::ArrayXd volume_entities(const dolfinx::mesh::Mesh& mesh,
                               const Eigen::Ref<const Eigen::ArrayXi>& entities,
                               int dim);

} // namespace common
} // namespace leopart