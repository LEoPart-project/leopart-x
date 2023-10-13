// Copyright: (c) 2020 Chris Richardson and Jakob Maljaars
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "generation.h"
#include <algorithm>
#include <dolfinx.h>
#include <iostream>
#include <random>
#include <stdint.h>

using namespace leopart;

//------------------------------------------------------------------------
template <std::floating_point T>
std::tuple<std::vector<double>, std::vector<std::int32_t>>
generation::mesh_fill(const dolfinx::mesh::Mesh<T>& mesh, double density)
{
  const int tdim = mesh.topology().dim();
  const int gdim = mesh.geometry().dim();
  const int num_cells = mesh.topology().index_map(tdim)->size_local();
  const dolfinx::mesh::CellType celltype = mesh.topology().cell_type();

  // // Cell volumes
  // Eigen::ArrayXi indices
  //     = Eigen::VectorXi::LinSpaced(num_cells, 0, num_cells - 1);
  // Eigen::ArrayXd vol = dolfinx::mesh::volume_entities(mesh, indices, tdim);

  // // Coordinate dofs
  // const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
  //     = mesh.geometry().dofmap();
  // const int num_dofs_g = x_dofmap.num_links(0);
  // const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
  //     = mesh.geometry().x();
  // Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  //     cell_geometry(num_dofs_g, gdim);

  // std::vector<int> cells;
  // std::vector<double> xc;
  // for (int i = 0; i < num_cells; ++i)
  // {
  //   // Number of particles = int(density * volume)
  //   int np = (int)(density * vol[i]);
  //   // Warn if too large or small
  //   if (np < 5)
  //     std::cout << "Warning: np < 5 in cell " << i << "\n";
  //   if (np > 50)
  //     std::cout << "Warning: np > 50 in cell " << i << "\n";

  //   // get random X values
  //   Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X
  //       = random_reference(celltype, np);
  //   Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x(
  //       X.rows(), gdim);
  //   // Convert to physical x values
  //   auto x_dofs = x_dofmap.links(i);
  //   for (int j = 0; j < num_dofs_g; ++j)
  //     cell_geometry.row(j) = x_g.row(x_dofs[j]).head(gdim);

  //   mesh.geometry().cmap().push_forward(x, X, cell_geometry);
  //   // append to list
  //   xc.insert(xc.end(), x.data(), x.data() + x.rows() * x.cols());
  //   std::vector<int> npcells(np, i);
  //   cells.insert(cells.end(), npcells.begin(), npcells.end());
  // }

  // Eigen::Map<
  //     Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
  //     xc_eigen(xc.data(), xc.size() / gdim, gdim);

  // // return list of x, cells
  // return {xc_eigen, cells};
  return {std::vector<double>{}, std::vector<std::int32_t>{}};
}
//------------------------------------------------------------------------
std::vector<double>
generation::random_reference(
  dolfinx::mesh::CellType celltype, std::size_t n)
{
  if (celltype == dolfinx::mesh::CellType::triangle)
    return random_reference_triangle(n);
  if (celltype == dolfinx::mesh::CellType::tetrahedron)
    return random_reference_tetrahedron(n);

  throw std::runtime_error("Unsupported cell type");

  return std::vector<double>{};
}
//------------------------------------------------------------------------
std::vector<double> generation::random_reference_triangle(std::size_t n)
{
  const int gdim = 2;
  std::vector<double> p(n * gdim);

  std::random_device rd;
  std::mt19937 rgen(rd());
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  std::generate(p.begin(), p.end(), [&dist, &rgen]() { return dist(rgen); });

  for (int i = 0; i < n; ++i)
  {
    std::span x(p.begin() + i * gdim, gdim);
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
  }
  
  for (double& val : p)
    val /= 2.0;
  return p;
}
//------------------------------------------------------------------------
std::vector<double>
generation::random_reference_tetrahedron(std::size_t n)
{
  const int gdim = 3;
  std::vector<double> p(n * gdim, 1.0);
  std::vector<double> r(gdim);

  std::random_device rd;
  std::mt19937 rgen(rd());
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  for (int i = 0; i < n; ++i)
  {    
    // Eigen::RowVector3d r = Eigen::Vector3d::Random();
    std::generate(r.begin(), r.end(),
                  [&dist, &rgen]() { return dist(rgen); });
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

    const auto p_row = &p[i * gdim];
    for (int j = 0; j < gdim; ++j)
      p_row[j] += r[j];
  }

  for (double& val : p)
    val /= 2.0;
  return p;
}
