#include "dolfinxutils.h"

namespace leopart
{
namespace common
{
//-----------------------------------------------------------------------------
template <typename T>
T volume_interval(const dolfinx::mesh::Mesh& mesh,
                  const Eigen::Ref<const Eigen::ArrayXi>& entities)
{
  const mesh::Geometry& geometry = mesh.geometry();
  const graph::AdjacencyList<std::int32_t>& x_dofs = geometry.dofmap();

  const dolfinx::array2d<double>& x_coords = geometry.x();

  T v(entities.rows());
  for (Eigen::Index i = 0; i < entities.rows(); ++i)
  {
    // Get the coordinates of the two vertices
    auto dofs = x_dofs.links(entities[i]);

    tcb::span<const double> x0 = x_coords.row(dofs[0]);
    tcb::span<const double> x1 = x_coords.row(dofs[1]);
    // const Eigen::Vector3d x0 = geometry.node(dofs[0]);
    // const Eigen::Vector3d x1 = geometry.node(dofs[1]);
    // TODO!
    // v[i] = (x1 - x0).norm();
    throw std::runtime_error("Interval volume not implemented");
    // v[i] = 0.;
  }

  return v;
}
//-----------------------------------------------------------------------------
template <typename T>
T volume_triangle(const dolfinx::mesh::Mesh& mesh,
                  const Eigen::Ref<const Eigen::ArrayXi>& entities)
{
  const mesh::Geometry& geometry = mesh.geometry();
  const int gdim = geometry.dim();
  assert(gdim == 2 or gdim == 3);
  const graph::AdjacencyList<std::int32_t>& x_dofs = geometry.dofmap();

  const dolfinx::array2d<double>& x_coords = geometry.x();

  T v(entities.rows());
  if (gdim == 2)
  {
    for (Eigen::Index i = 0; i < entities.rows(); ++i)
    {
      auto dofs = x_dofs.links(entities[i]);

      tcb::span<const double> x0 = x_coords.row(dofs[0]);
      tcb::span<const double> x1 = x_coords.row(dofs[1]);
      tcb::span<const double> x2 = x_coords.row(dofs[2]);

      //   const Eigen::Vector3d x0 = geometry.node(dofs[0]);
      //   const Eigen::Vector3d x1 = geometry.node(dofs[1]);
      //   const Eigen::Vector3d x2 = geometry.node(dofs[2]);

      // Compute area of triangle embedded in R^2
      double v2 = (x0[0] * x1[1] + x0[1] * x2[0] + x1[0] * x2[1])
                  - (x2[0] * x1[1] + x2[1] * x0[0] + x1[0] * x0[1]);

      // Formula for volume from http://mathworld.wolfram.com
      v[i] = 0.5 * std::abs(v2);
    }
  }
  else if (gdim == 3)
  {
    for (Eigen::Index i = 0; i < entities.rows(); ++i)
    {
      auto dofs = x_dofs.links(entities[i]);

      tcb::span<const double> x0 = x_coords.row(dofs[0]);
      tcb::span<const double> x1 = x_coords.row(dofs[1]);
      tcb::span<const double> x2 = x_coords.row(dofs[2]);
      //   const Eigen::Vector3d x0 = geometry.node(dofs[0]);
      //   const Eigen::Vector3d x1 = geometry.node(dofs[1]);
      //   const Eigen::Vector3d x2 = geometry.node(dofs[2]);

      // Compute area of triangle embedded in R^3
      const double v0 = (x0[1] * x1[2] + x0[2] * x2[1] + x1[1] * x2[2])
                        - (x2[1] * x1[2] + x2[2] * x0[1] + x1[1] * x0[2]);
      const double v1 = (x0[2] * x1[0] + x0[0] * x2[2] + x1[2] * x2[0])
                        - (x2[2] * x1[0] + x2[0] * x0[2] + x1[2] * x0[0]);
      const double v2 = (x0[0] * x1[1] + x0[1] * x2[0] + x1[0] * x2[1])
                        - (x2[0] * x1[1] + x2[1] * x0[0] + x1[0] * x0[1]);

      // Formula for volume from http://mathworld.wolfram.com
      v[i] = 0.5 * sqrt(v0 * v0 + v1 * v1 + v2 * v2);
    }
  }
  else
    throw std::runtime_error("Unexpected geometric dimension.");

  return v;
}
//-----------------------------------------------------------------------------
template <typename T>
T volume_tetrahedron(const dolfinx::mesh::Mesh& mesh,
                     const Eigen::Ref<const Eigen::ArrayXi>& entities)
{
  const mesh::Geometry& geometry = mesh.geometry();
  const graph::AdjacencyList<std::int32_t>& x_dofs = geometry.dofmap();

  const dolfinx::array2d<double>& x_coords = geometry.x();

  Eigen::ArrayXd v(entities.rows());
  for (Eigen::Index i = 0; i < entities.rows(); ++i)
  {
    auto dofs = x_dofs.links(entities[i]);
    tcb::span<const double> x0 = x_coords.row(dofs[0]);
    tcb::span<const double> x1 = x_coords.row(dofs[1]);
    tcb::span<const double> x2 = x_coords.row(dofs[2]);
    tcb::span<const double> x3 = x_coords.row(dofs[3]);

    // const Eigen::Vector3d x0 = geometry.node(dofs[0]);
    // const Eigen::Vector3d x1 = geometry.node(dofs[1]);
    // const Eigen::Vector3d x2 = geometry.node(dofs[2]);
    // const Eigen::Vector3d x3 = geometry.node(dofs[3]);

    // Formula for volume from http://mathworld.wolfram.com
    const double v_tmp
        = (x0[0]
               * (x1[1] * x2[2] + x3[1] * x1[2] + x2[1] * x3[2] - x2[1] * x1[2]
                  - x1[1] * x3[2] - x3[1] * x2[2])
           - x1[0]
                 * (x0[1] * x2[2] + x3[1] * x0[2] + x2[1] * x3[2]
                    - x2[1] * x0[2] - x0[1] * x3[2] - x3[1] * x2[2])
           + x2[0]
                 * (x0[1] * x1[2] + x3[1] * x0[2] + x1[1] * x3[2]
                    - x1[1] * x0[2] - x0[1] * x3[2] - x3[1] * x1[2])
           - x3[0]
                 * (x0[1] * x1[2] + x1[1] * x2[2] + x2[1] * x0[2]
                    - x1[1] * x0[2] - x2[1] * x1[2] - x0[1] * x2[2]));

    v[i] = std::abs(v_tmp) / 6.0;
  }

  return v;
}

/// Compute (generalized) volume of mesh entities of given dimension.
/// This templated versions allows for fixed size (statically allocated)
/// return arrays, which can be important for performance when computing
/// for a small number of entities.
template <typename T>
T volume_entities_tmpl(const dolfinx::mesh::Mesh& mesh,
                       const Eigen::Ref<const Eigen::ArrayXi>& entities,
                       int dim)
{
  const mesh::CellType type
      = cell_entity_type(mesh.topology().cell_type(), dim);
  switch (type)
  {
  case mesh::CellType::point:
  {
    T v(entities.rows());
    v.setOnes();
    return v;
  }
  case mesh::CellType::interval:
    return volume_interval<T>(mesh, entities);
  case mesh::CellType::triangle:
    assert(mesh.topology().dim() == dim);
    return volume_triangle<T>(mesh, entities);
  case mesh::CellType::tetrahedron:
    return volume_tetrahedron<T>(mesh, entities);
  case mesh::CellType::quadrilateral:
    throw std::runtime_error(
        "Volume compuation for quadrilateral not supported");
    // assert(mesh.topology().dim() == dim);
    // return volume_quadrilateral<T>(mesh, entities);
  case mesh::CellType::hexahedron:
    throw std::runtime_error(
        "Volume computation for hexahedral cell not supported.");
  default:
    throw std::runtime_error("Unknown cell type.");
    return T();
  }
}

//-----------------------------------------------------------------------------
Eigen::ArrayXd volume_entities(const dolfinx::mesh::Mesh& mesh,
                               const Eigen::Ref<const Eigen::ArrayXi>& entities,
                               int dim)
{
  return volume_entities_tmpl<Eigen::ArrayXd>(mesh, entities, dim);
}

} // namespace common
} // namespace leopart