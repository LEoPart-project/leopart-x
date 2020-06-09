
#include <Eigen/Dense>

#pragma once

namespace dolfinx
{
namespace mesh
{
class Mesh;
enum class CellType;
} // namespace mesh
} // namespace dolfinx

namespace leopart
{
namespace generation
{

std::tuple<
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
    std::vector<int>>
mesh_fill(const dolfinx::mesh::Mesh& mesh, double density);

/// Create a set of n points at random positions within the cell
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
random_reference(dolfinx::mesh::CellType celltype, int n);

/// Create a set of n points at random positions within the reference triangle
Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>
random_reference_tetrahedron(int n);

/// Create a set of n points at random positions within the reference
/// tetrahedron
Eigen::Array<double, Eigen::Dynamic, 2, Eigen::RowMajor>
random_reference_triangle(int n);
} // namespace generation
}; // namespace leopart