// Copyright: (c) 2020-2023 Chris Richardson, Jakob Maljaars and Nathan Sime
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx.h>
#include "Particles.h"


namespace leopart::utils
{
/// @brief Utility alias for frequently used mdpsan with dynamic extents
/// @tparam T Data type
/// @tparam d Rank
template <typename T, std::size_t d>
using mdspan_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, d>>;

/// Format and print 
/// @param[in] span Rank 2 mdspan to format and print.
/// @param[in] prefix An identifier prefixing the string
void print_mdspan(
  const auto span, const std::string prefix = "")
{
  std::string msg = prefix + ":\n";
  for (int i = 0; i < span.extent(0); ++i)
  {
    for (int j = 0; j < span.extent(1); ++j)
      msg += std::to_string(span(i, j)) + ", ";
    msg += "\n";
  }
  std::cout << msg << std::endl << std::flush;
};

/// Format and print 
void print_iterable(
  const auto span, const std::string prefix = "")
{
  std::string msg = prefix + ":\n";
  for (const auto& item: span)
    msg += std::to_string(item) + ", ";
  std::cout << msg << std::endl << std::flush;
};


/// Adapted from dolfinx_mpc::utils::evaluate_basis_functions
/// originally authored by Jorgen S. Dokken
///
/// Get basis values (not unrolled for block size) for a set of points and
/// corresponding cells.
/// @param[in] V The function space
/// @param[in] x The coordinates of the points. It has shape
/// (num_points, 3), flattened row major
/// @param[in] cells An array of cell indices. cells[i] is the index
/// of the cell that contains the point x(i). Negative cell indices
/// can be passed, and the corresponding point will be ignored.
/// @param[in,out] u The values at the points. Values are not computed
/// for points with a negative cell index. This argument must be
/// passed with the correct size.
/// @returns basis values (not unrolled for block size) for each point. shape
/// (num_points, number_of_dofs, value_size). Flattened row major
template <std::floating_point U>
std::pair<std::vector<U>, std::array<std::size_t, 3>>
evaluate_basis_functions(const dolfinx::fem::FunctionSpace<U>& V,
                         std::span<const U> x,
                         std::span<const std::int32_t> cells)
{
  assert(x.size() % 3 == 0);
  const std::size_t num_points = x.size() / 3;
  if (num_points != cells.size())
  {
    throw std::runtime_error(
        "Number of points and number of cells must be equal.");
  }

  // Get mesh
  auto mesh = V.mesh();
  assert(mesh);
  const std::size_t gdim = mesh->geometry().dim();
  const std::size_t tdim = mesh->topology()->dim();
  auto map = mesh->topology()->index_map(tdim);

  // Get geometry data
  namespace stdex = std::experimental;
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const std::int32_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      x_dofmap = mesh->geometry().dofmap();

  auto cmaps = mesh->geometry().cmaps();
  if (cmaps.size() > 1)
  {
    throw std::runtime_error(
        "Multiple coordinate maps in evaluate basis functions");
  }
  const std::size_t num_dofs_g = cmaps[0].dim();
  std::span<const U> x_g = mesh->geometry().x();

  // Get element
  auto element = V.element();
  assert(element);
  const int bs_element = element->block_size();
  const std::size_t reference_value_size
      = element->reference_value_size() / bs_element;

  // If the space has sub elements, concatenate the evaluations on the
  // sub elements
  const int num_sub_elements = element->num_sub_elements();
  if (num_sub_elements > 1 and num_sub_elements != bs_element)
  {
    throw std::runtime_error(
        "Evaluation of basis functions is not supported for mixed "
        "elements. Extract subspaces.");
  }

  // Return early if we have no points
  std::array<std::size_t, 4> basis_shape
      = element->basix_element().tabulate_shape(0, num_points);

  assert(basis_shape[2]
         == std::size_t(element->space_dimension() / bs_element));
  assert(basis_shape[3] == std::size_t(element->value_size() / bs_element));
  std::array<std::size_t, 3> reference_shape
      = {basis_shape[1], basis_shape[2], basis_shape[3]};
  std::vector<U> output_basis(std::reduce(
      reference_shape.begin(), reference_shape.end(), 1, std::multiplies{}));

  if (num_points == 0)
    return {output_basis, reference_shape};

  std::span<const std::uint32_t> cell_info;
  if (element->needs_dof_transformations())
  {
    mesh->topology_mutable()->create_entity_permutations();
    cell_info = std::span(mesh->topology()->get_cell_permutation_info());
  }

  using cmdspan4_t = mdspan_t<const U, 4>;
  using mdspan2_t = mdspan_t<U, 2>;
  using mdspan3_t = mdspan_t<U, 3>;

  // Create buffer for coordinate dofs and point in physical space
  std::vector<U> coord_dofs_b(num_dofs_g * gdim);
  mdspan2_t coord_dofs(coord_dofs_b.data(), num_dofs_g, gdim);
  std::vector<U> xp_b(1 * gdim);
  mdspan2_t xp(xp_b.data(), 1, gdim);

  // Evaluate geometry basis at point (0, 0, 0) on the reference cell.
  // Used in affine case.
  std::array<std::size_t, 4> phi0_shape = cmaps[0].tabulate_shape(1, 1);
  std::vector<U> phi0_b(
      std::reduce(phi0_shape.begin(), phi0_shape.end(), 1, std::multiplies{}));
  cmdspan4_t phi0(phi0_b.data(), phi0_shape);
  cmaps[0].tabulate(1, std::vector<U>(tdim, 0), {1, tdim}, phi0_b);
  auto dphi0 = stdex::submdspan(phi0, std::pair(1, tdim + 1), 0,
                                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

  // Data structure for evaluating geometry basis at specific points.
  // Used in non-affine case.
  std::array<std::size_t, 4> phi_shape = cmaps[0].tabulate_shape(1, 1);
  std::vector<U> phi_b(
      std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
  cmdspan4_t phi(phi_b.data(), phi_shape);
  auto dphi = stdex::submdspan(phi, std::pair(1, tdim + 1), 0,
                               MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

  // Reference coordinates for each point
  std::vector<U> Xb(num_points * tdim);
  mdspan2_t X(Xb.data(), num_points, tdim);

  // Geometry data at each point
  std::vector<U> J_b(num_points * gdim * tdim);
  mdspan3_t J(J_b.data(), num_points, gdim, tdim);
  std::vector<U> K_b(num_points * tdim * gdim);
  mdspan3_t K(K_b.data(), num_points, tdim, gdim);
  std::vector<U> detJ(num_points);
  std::vector<U> det_scratch(2 * gdim * tdim);

  // Prepare geometry data in each cell
  for (std::size_t p = 0; p < cells.size(); ++p)
  {
    const int cell_index = cells[p];

    // Skip negative cell indices
    if (cell_index < 0)
      continue;

    // Get cell geometry (coordinate dofs)
    auto x_dofs = stdex::submdspan(x_dofmap, cell_index,
                                   MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < num_dofs_g; ++i)
    {
      const int pos = 3 * x_dofs[i];
      for (std::size_t j = 0; j < gdim; ++j)
        coord_dofs(i, j) = x_g[pos + j];
    }

    for (std::size_t j = 0; j < gdim; ++j)
      xp(0, j) = x[3 * p + j];

    auto _J
        = stdex::submdspan(J, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
                           MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    auto _K
        = stdex::submdspan(K, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
                           MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

    std::array<U, 3> Xpb = {0, 0, 0};
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        U, MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
               std::size_t, 1, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent>>
        Xp(Xpb.data(), 1, tdim);

    // Compute reference coordinates X, and J, detJ and K
    if (cmaps[0].is_affine())
    {
      dolfinx::fem::CoordinateElement<U>::compute_jacobian(dphi0, coord_dofs,
                                                           _J);
      dolfinx::fem::CoordinateElement<U>::compute_jacobian_inverse(_J, _K);
      std::array<U, 3> x0 = {0, 0, 0};
      for (std::size_t i = 0; i < coord_dofs.extent(1); ++i)
        x0[i] += coord_dofs(0, i);
      dolfinx::fem::CoordinateElement<U>::pull_back_affine(Xp, _K, x0, xp);
      detJ[p]
          = dolfinx::fem::CoordinateElement<U>::compute_jacobian_determinant(
              _J, det_scratch);
    }
    else
    {
      // Pull-back physical point xp to reference coordinate Xp
      cmaps[0].pull_back_nonaffine(Xp, xp, coord_dofs);

      cmaps[0].tabulate(1, std::span(Xpb.data(), tdim), {1, tdim}, phi_b);
      dolfinx::fem::CoordinateElement<U>::compute_jacobian(dphi, coord_dofs,
                                                           _J);
      dolfinx::fem::CoordinateElement<U>::compute_jacobian_inverse(_J, _K);
      detJ[p]
          = dolfinx::fem::CoordinateElement<U>::compute_jacobian_determinant(
              _J, det_scratch);
    }

    for (std::size_t j = 0; j < X.extent(1); ++j)
      X(p, j) = Xpb[j];
  }

  // Compute basis on reference element
  std::vector<U> reference_basisb(std::reduce(
      basis_shape.begin(), basis_shape.end(), 1, std::multiplies{}));
  element->tabulate(reference_basisb, Xb, {X.extent(0), X.extent(1)}, 0);

  // Data structure to hold basis for transformation
  const std::size_t num_basis_values = basis_shape[2] * basis_shape[3];
  std::vector<U> basis_valuesb(num_basis_values);
  mdspan2_t basis_values(basis_valuesb.data(), basis_shape[2], basis_shape[3]);

  using xu_t = mdspan_t<U, 2>;
  using xU_t = mdspan_t<const U, 2>;
  using xJ_t = mdspan_t<const U, 2>;
  using xK_t = mdspan_t<const U, 2>;
  auto push_forward_fn
      = element->basix_element().template map_fn<xu_t, xU_t, xJ_t, xK_t>();

  auto apply_dof_transformation
      = element->template get_dof_transformation_function<U>();

  mdspan3_t full_basis(output_basis.data(), reference_shape);
  for (std::size_t p = 0; p < cells.size(); ++p)
  {
    const int cell_index = cells[p];
    // Skip negative cell indices
    if (cell_index < 0)
      continue;

    // Permute the reference values to account for the cell's orientation
    std::copy_n(std::next(reference_basisb.begin(), num_basis_values * p),
                num_basis_values, basis_valuesb.begin());
    apply_dof_transformation(basis_valuesb, cell_info, cell_index,
                             (int)reference_value_size);

    auto _U = stdex::submdspan(full_basis, p,
                               MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
                               MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    auto _J
        = stdex::submdspan(J, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
                           MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    auto _K
        = stdex::submdspan(K, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
                           MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    push_forward_fn(_U, basis_values, _J, detJ[p], _K);
  }
  return {output_basis, reference_shape};
}


/// Adapted from dolfinx::geomety::utils::determine_point_ownership
///
/// @brief Given a set of points, determine which process is colliding,
/// using the GJK algorithm on cells to determine collisions.
///
/// @todo This docstring is unclear. Needs fixing.
///
/// @param[in] mesh The mesh
/// @param[in] points Points to check for collision (`shape=(num_points,
/// 3)`). Storage is row-major.
/// @return Tuple (src_owner, dest_owner, dest_points,
/// dest_cells), where src_owner is a list of ranks corresponding to the
/// input points. dest_owner is a list of ranks found to own
/// dest_points. dest_cells contains the corresponding cell for each
/// entry in dest_points.
///
/// @note dest_owner is sorted
/// @note Returns -1 if no colliding process is found
/// @note dest_points is flattened row-major, shape (dest_owner.size(), 3)
/// @note Only looks through cells owned by the process
template <std::floating_point T>
std::tuple<std::vector<std::int32_t>, std::vector<std::int32_t>, std::vector<T>,
           std::vector<std::int32_t>>
determine_point_ownership(
  const leopart::Particles<T>& ptcls, const mesh::Mesh<T>& mesh,
  std::span<const std::size_t> pidxs)
{
  using namespace dolfinx;
  using namespace dolfinx::geometry;
  
  MPI_Comm comm = mesh.comm();

  // Create a global bounding-box tree to find candidate processes with
  // cells that could collide with the points
  constexpr T padding = 1.0e-4;
  const int tdim = mesh.topology()->dim();
  auto cell_map = mesh.topology()->index_map(tdim);
  const std::int32_t num_cells = cell_map->size_local();
  // NOTE: Should we send the cells in as input?
  std::vector<std::int32_t> cells(num_cells, 0);
  std::iota(cells.begin(), cells.end(), 0);
  BoundingBoxTree bb(mesh, tdim, cells, padding);
  BoundingBoxTree midpoint_tree = create_midpoint_tree(mesh, tdim, cells);
  BoundingBoxTree global_bbtree = bb.create_global_tree(comm);

  // Get positions of required pidxs
  std::span<const T> xp_all = ptcls.field("x").data();
  const std::size_t gdim = ptcls.field("x").value_shape()[0];
  std::vector<T> points(pidxs.size() * gdim, 0.0);
  for (std::size_t i = 0; i < pidxs.size(); ++i)
  {
    std::copy_n(xp_all.begin() + pidxs[i] * gdim, gdim, points.begin() + i * gdim);
  }

  // Compute collisions:
  // For each point in `x` get the processes it should be sent to
  graph::AdjacencyList collisions = compute_collisions(global_bbtree, std::span<const T>(points));

  // Get unique list of outgoing ranks
  std::vector<std::int32_t> out_ranks = collisions.array();
  std::sort(out_ranks.begin(), out_ranks.end());
  out_ranks.erase(std::unique(out_ranks.begin(), out_ranks.end()),
                  out_ranks.end());

  // Compute incoming edges (source processes)
  std::vector in_ranks = dolfinx::MPI::compute_graph_edges_nbx(comm, out_ranks);
  std::sort(in_ranks.begin(), in_ranks.end());

  // Create neighborhood communicator in forward direction
  MPI_Comm forward_comm;
  MPI_Dist_graph_create_adjacent(
      comm, in_ranks.size(), in_ranks.data(), MPI_UNWEIGHTED, out_ranks.size(),
      out_ranks.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &forward_comm);

  // Compute map from global mpi rank to neighbor rank, "collisions" uses
  // global rank
  std::map<std::int32_t, std::int32_t> rank_to_neighbor;
  for (std::size_t i = 0; i < out_ranks.size(); i++)
    rank_to_neighbor[out_ranks[i]] = i;

  // -----------------------------------------------------------------------
  // Send geometry to global process colliding ranks
  // -----------------------------------------------------------------------

  struct forward_comm_result
  {
    const std::size_t value_size;
    const std::vector<T> received_data;
    std::vector<std::int32_t> recv_sizes;
    std::vector<std::int32_t> recv_offsets;
    std::vector<std::int32_t> send_sizes;
    std::vector<std::int32_t> send_offsets;
    const std::vector<std::int32_t> unpack_map;
    std::vector<std::int32_t> counter;
  };

  auto send_data_foward = [&forward_comm, &out_ranks, &rank_to_neighbor,
                           &in_ranks, &collisions]
                          (const std::vector<T>& data,
                           const std::size_t value_size)
                          -> forward_comm_result
  {
    // Count the number of data to send per neighbor process
    std::vector<std::int32_t> send_sizes(out_ranks.size());
    for (std::size_t i = 0; i < data.size() / value_size; ++i)
      for (auto p : collisions.links(i))
        send_sizes[rank_to_neighbor[p]] += value_size;

    // Compute receive sizes
    std::vector<std::int32_t> recv_sizes(in_ranks.size());
    send_sizes.reserve(1);
    recv_sizes.reserve(1);
    MPI_Request sizes_request;
    MPI_Ineighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1,
                           MPI_INT, forward_comm, &sizes_request);

    // Compute sending offsets
    std::vector<std::int32_t> send_offsets(send_sizes.size() + 1, 0);
    std::partial_sum(send_sizes.begin(), send_sizes.end(),
                    std::next(send_offsets.begin(), 1));

    // Pack data to send and store unpack map
    std::vector<T> send_data(send_offsets.back());
    std::vector<std::int32_t> counter(send_sizes.size(), 0);
    // unpack map: [index in adj list][pos in x]
    std::vector<std::int32_t> unpack_map(send_offsets.back() / value_size);
    for (std::size_t i = 0; i < data.size() / value_size; ++i)
    {
      for (auto p : collisions.links(i))
      {
        const int neighbor = rank_to_neighbor[p];
        const int pos = send_offsets[neighbor] + counter[neighbor];
        std::copy(std::next(data.begin(), i * value_size),
                  std::next(data.begin(), (i + 1) * value_size),
                  std::next(send_data.begin(), pos));
        unpack_map[pos / value_size] = i;
        counter[neighbor] += value_size;
      }
    }

    MPI_Wait(&sizes_request, MPI_STATUS_IGNORE);
    std::vector<std::int32_t> recv_offsets(in_ranks.size() + 1, 0);
    std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                    std::next(recv_offsets.begin(), 1));

    std::vector<T> received_data((std::size_t)recv_offsets.back());
    MPI_Neighbor_alltoallv(
        send_data.data(), send_sizes.data(), send_offsets.data(),
        dolfinx::MPI::mpi_type<T>(), received_data.data(), recv_sizes.data(),
        recv_offsets.data(), dolfinx::MPI::mpi_type<T>(), forward_comm);

    return {
      .value_size=value_size,
      .received_data=std::move(received_data),
      .recv_sizes=std::move(recv_sizes),
      .recv_offsets=std::move(recv_offsets),
      .send_sizes=std::move(send_sizes),
      .send_offsets=std::move(send_offsets),
      .unpack_map=std::move(unpack_map),
      .counter=std::move(counter)};
  };

  forward_comm_result res_x = send_data_foward(points, 3);

  

  // -----------------------------------------------------------------------
  // Compute closest entities on the owning ranks
  // -----------------------------------------------------------------------

  // Each process checks which local cell is closest and computes the squared
  // distance to the cell
  const int rank = dolfinx::MPI::rank(comm);
  const std::vector<std::int32_t> closest_cells = compute_closest_entity(
      bb, midpoint_tree, mesh,
      std::span<const T>(res_x.received_data.data(), res_x.received_data.size()));
  const std::vector<T> squared_distances = squared_distance(
      mesh, tdim, closest_cells,
      std::span<const T>(res_x.received_data.data(), res_x.received_data.size()));

  // -----------------------------------------------------------------------
  // Communicate reverse the squared differences
  // -----------------------------------------------------------------------

  // Create neighborhood communicator in the reverse direction: send
  // back col to requesting processes
  MPI_Comm reverse_comm;
  MPI_Dist_graph_create_adjacent(
      comm, out_ranks.size(), out_ranks.data(), MPI_UNWEIGHTED, in_ranks.size(),
      in_ranks.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &reverse_comm);

  // Reuse sizes and offsets from first communication set
  // but divide by three
  {
    auto rescale = [](auto& x)
    {
      std::transform(x.cbegin(), x.cend(), x.begin(),
                     [](auto e) { return (e / 3); });
    };
    rescale(res_x.recv_sizes);
    rescale(res_x.recv_offsets);
    rescale(res_x.send_sizes);
    rescale(res_x.send_offsets);

    // The communication is reversed, so swap recv to send offsets
    std::swap(res_x.recv_sizes, res_x.send_sizes);
    std::swap(res_x.recv_offsets, res_x.send_offsets);
  }

  // Get distances from closest entity of points that were on the other process
  std::vector<T> recv_distances(res_x.recv_offsets.back());
  MPI_Neighbor_alltoallv(
      squared_distances.data(), res_x.send_sizes.data(), res_x.send_offsets.data(),
      dolfinx::MPI::mpi_type<T>(), recv_distances.data(), res_x.recv_sizes.data(),
      res_x.recv_offsets.data(), dolfinx::MPI::mpi_type<T>(), reverse_comm);

  // -----------------------------------------------------------------------
  // For each point find the owning process which minimises the square
  // distance to a cell
  // -----------------------------------------------------------------------

  std::vector<std::int32_t> point_owners(points.size() / 3, -1);
  std::vector<T> closest_distance(points.size() / 3, -1);
  for (std::size_t i = 0; i < out_ranks.size(); i++)
  {
    for (std::int32_t j = res_x.recv_offsets[i]; j < res_x.recv_offsets[i + 1]; j++)
    {
      const std::int32_t pos = res_x.unpack_map[j];
      // If point has not been found yet distance is negative
      // If new received distance smaller than current distance choose owner
      if (auto d = closest_distance[pos]; d < 0 or d > recv_distances[j])
      {
        point_owners[pos] = out_ranks[i];
        closest_distance[pos] = recv_distances[j];
      }
    }
  }

  // -----------------------------------------------------------------------
  // Communicate forward the destination ranks for each point
  // -----------------------------------------------------------------------

  // Communication is reversed again to send dest ranks to all processes
  std::swap(res_x.send_sizes, res_x.recv_sizes);
  std::swap(res_x.send_offsets, res_x.recv_offsets);

  // Pack ownership data
  std::vector<std::int32_t> send_owners(res_x.send_offsets.back());
  std::fill(res_x.counter.begin(), res_x.counter.end(), 0);
  for (std::size_t i = 0; i < points.size() / 3; ++i)
  {
    for (auto p : collisions.links(i))
    {
      int neighbor = rank_to_neighbor[p];
      send_owners[res_x.send_offsets[neighbor] + res_x.counter[neighbor]++]
          = point_owners[i];
    }
  }

  // Send ownership info
  std::vector<std::int32_t> dest_ranks(res_x.recv_offsets.back());
  MPI_Neighbor_alltoallv(
      send_owners.data(), res_x.send_sizes.data(), res_x.send_offsets.data(),
      dolfinx::MPI::mpi_type<std::int32_t>(), dest_ranks.data(),
      res_x.recv_sizes.data(), res_x.recv_offsets.data(),
      dolfinx::MPI::mpi_type<std::int32_t>(), forward_comm);


  // -----------------------------------------------------------------------
  // Compute the closest cell on the destination rank
  // -----------------------------------------------------------------------

  // Unpack dest ranks if point owner is this rank
  std::vector<std::int32_t> owned_recv_ranks;
  owned_recv_ranks.reserve(res_x.recv_offsets.back());
  std::vector<T> owned_recv_points;
  std::vector<std::int32_t> owned_recv_cells;
  for (std::size_t i = 0; i < in_ranks.size(); i++)
  {
    for (std::int32_t j = res_x.recv_offsets[i]; j < res_x.recv_offsets[i + 1]; j++)
    {
      if (rank == dest_ranks[j])
      {
        owned_recv_ranks.push_back(in_ranks[i]);
        owned_recv_points.insert(
            owned_recv_points.end(), std::next(res_x.received_data.cbegin(), 3 * j),
            std::next(res_x.received_data.cbegin(), 3 * (j + 1)));
        owned_recv_cells.push_back(closest_cells[j]);
      }
    }
  }

  MPI_Comm_free(&forward_comm);
  MPI_Comm_free(&reverse_comm);

  return std::make_tuple(point_owners, owned_recv_ranks, owned_recv_points,
                         owned_recv_cells);
}
}