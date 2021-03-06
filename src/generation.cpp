// Copyright: (c) 2020 Chris Richardson and Jakob Maljaars
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "generation.h"
#include "common/dolfinxutils.h"
#include <Eigen/Dense>
#include <iostream>

namespace leopart
{
//------------------------------------------------------------------------
std::tuple<
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
    std::vector<int>>
generation::mesh_fill(const dolfinx::mesh::Mesh& mesh, double density)
{
  const int tdim = mesh.topology().dim();
  const int gdim = mesh.geometry().dim();
  const int num_cells = mesh.topology().index_map(tdim)->size_local();
  const dolfinx::mesh::CellType celltype = mesh.topology().cell_type();

  // Cell volumes
  Eigen::ArrayXi indices
      = Eigen::VectorXi::LinSpaced(num_cells, 0, num_cells - 1);
  Eigen::ArrayXd vol = leopart::common::volume_entities(mesh, indices, tdim);

  // Coordinate dofs
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh.geometry().dofmap();
  const int num_dofs_g = x_dofmap.num_links(0);
  const dolfinx::array2d<double>& x_g = mesh.geometry().x();
  dolfinx::array2d<double> cell_geometry(num_dofs_g, gdim);

  std::vector<int> cells;
  std::vector<double> xc;
  for (int i = 0; i < num_cells; ++i)
  {
    // Number of particles = int(density * volume)
    int np = (int)(density * vol[i]);
    // Warn if too large or small
    if (np < 5)
      std::cout << "Warning: np < 5 in cell " << i << "\n";
    if (np > 50)
      std::cout << "Warning: np > 50 in cell " << i << "\n";

    // Get random X values
    dolfinx::array2d<double> X = random_reference(celltype, np);
    dolfinx::array2d<double> x(X.shape[0], gdim);

    // Convert to physical x values
    auto x_dofs = x_dofmap.links(i);
    for (int j = 0; j < num_dofs_g; ++j)
      // TODO: make more efficient?
      for (int k = 0; k < gdim; ++k)
      {
        cell_geometry(j, k) = x_g(x_dofs[j], k);
      }

    mesh.geometry().cmap().push_forward(x, X, cell_geometry);
    // Append to list
    xc.insert(xc.end(), x.data(), x.data() + x.size());
    std::vector<int> npcells(np, i);
    cells.insert(cells.end(), npcells.begin(), npcells.end());
  }

  Eigen::Map<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      xc_eigen(xc.data(), xc.size() / gdim, gdim);

  // return list of x, cells
  return {xc_eigen, cells};
}
//------------------------------------------------------------------------
dolfinx::array2d<double>
generation::random_reference(dolfinx::mesh::CellType celltype, int n)
{
  if (celltype == dolfinx::mesh::CellType::triangle)
    return random_reference_triangle(n);
  if (celltype == dolfinx::mesh::CellType::tetrahedron)
    return random_reference_tetrahedron(n);

  throw std::runtime_error("Unsupported cell type");

  return dolfinx::array2d<double>(0, 0);
}
//------------------------------------------------------------------------
dolfinx::array2d<double> generation::random_reference_triangle(int n)
{
  dolfinx::array2d<double> p(n, 2);

  for (int i = 0; i < n; ++i)
  {
    Eigen::Vector2d x = Eigen::Vector2d::Random();
    if (x[0] + x[1] > 0.0)
    {
      x[0] = (1 - x[0]);
      x[1] = (1 - x[1]);
    }
    else
    {
      x[0] = (1 + x[0]);
      x[1] = (1 + x[1]);
    }
    // TODO: make more efficient
    // as in // p.row(i) = x;
    p(i, 0) = 0.5 * x[0];
    p(i, 1) = 0.5 * x[1];
  }
  // p /= 2.0;
  return p;
}
//------------------------------------------------------------------------
dolfinx::array2d<double> generation::random_reference_tetrahedron(int n)
{
  // Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> p(n, 3);
  // p.fill(1.0);

  dolfinx::array2d<double> p(n, 3, 1.0);

  for (int i = 0; i < n; ++i)
  {
    Eigen::RowVector3d r = Eigen::Vector3d::Random();
    double& x = r[0];
    double& y = r[1];
    double& z = r[2];

    // Fold cube into tetrahedron
    if ((x + z) > 0)
    {
      x = -x;
      z = -z;
    }

    if ((y + z) > 0)
    {
      z = -z - x - 1;
      y = -y;
    }
    else if ((x + y + z) > -1)
    {
      x = -x - z - 1;
      y = -y - z - 1;
    }

    // p.row(i) += r;
    // TODO: make more efficient
    p(i, 0) += 0.5 * r[0];
    p(i, 1) += 0.5 * r[1];
    p(i, 2) += 0.5 * r[2];
  }
  // p /= 2.0;

  return p;
}
} // namespace leopart