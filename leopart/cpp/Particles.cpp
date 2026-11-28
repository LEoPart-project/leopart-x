// Copyright (c) 2020-2023 Chris Richardson, Jakob Maljaars and Nathan Sime
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <algorithm>
#include <cassert>
#include <dolfinx.h>
#include <numeric>
#include <ranges>

#include "generation.h"
#include "Particles.h"

using namespace leopart;

template <std::floating_point T>
Particles<T>::Particles(const std::vector<T>& x,
                        const std::vector<std::int32_t>& cells,
                        const std::size_t gdim) :
                        _particle_to_cell(cells),
                        _x("x", {gdim}, cells.size())
{
  if (x.size() != cells.size() * gdim)
    throw std::invalid_argument(
      "Number of particle positions and cells differ");
  std::copy(x.cbegin(), x.cend(), _x.data().begin());

  // Negative cells, conventionally -1, mark particles to drop
  for (std::int32_t& c : _particle_to_cell)
    c = std::max(c, INVALID_CELL);
  sort_by_cell(0);
}
//------------------------------------------------------------------------
template <std::floating_point T>
std::size_t Particles<T>::append(std::size_t n)
{
  // Growing a std::vector value-initialises the new entries, so every
  // field of a new particle is zero
  const std::size_t first = num_particles();
  _x.resize(first + n);
  for (auto& [name, f] : _fields)
    f.resize(first + n);
  _particle_to_cell.resize(first + n, 0);
  return first;
}
//------------------------------------------------------------------------
template <std::floating_point T>
std::vector<std::size_t> Particles<T>::add_particles(
  std::span<const T> x, std::span<const std::int32_t> cells)
{
  const std::size_t xdim = _x.value_size();
  if (x.size() != cells.size() * xdim)
    throw std::invalid_argument(
      "Number of particle positions and cells differ");
  if (std::ranges::any_of(cells, [](std::int32_t c) { return c < 0; }))
    throw std::invalid_argument("Particle cells must be non-negative");

  const std::size_t first = append(cells.size());
  std::copy(x.begin(), x.end(), _x.data().begin() + first * xdim);
  std::copy(cells.begin(), cells.end(), _particle_to_cell.begin() + first);
  sort_by_cell(num_cells());

  // The sort is stable and the new particles were appended, so in each
  // cell they are the last particles, in the order given
  std::vector<std::size_t> next(num_cells(), 0);
  for (const std::int32_t c : cells)
    ++next[c];
  for (std::size_t c = 0; c < next.size(); ++c)
    next[c] = _cell_offsets[c + 1] - next[c];
  std::vector<std::size_t> pidxs(cells.size());
  for (std::size_t i = 0; i < cells.size(); ++i)
    pidxs[i] = next[cells[i]]++;
  return pidxs;
}
//------------------------------------------------------------------------
template <std::floating_point T>
void Particles<T>::delete_particles(std::span<const std::size_t> pidxs)
{
  if (std::ranges::any_of(pidxs, [n = num_particles()](std::size_t p)
                          { return p >= n; }))
    throw std::out_of_range("Particle index out of range");
  for (const std::size_t p : pidxs)
    _particle_to_cell[p] = INVALID_CELL;
  sort_by_cell(num_cells());
}
//------------------------------------------------------------------------
template <std::floating_point T>
void Particles<T>::add_field(
  std::string name, const std::vector<std::size_t>& shape)
{
  if (_fields.find(name) != _fields.end())
    throw std::runtime_error("Field name \"" + name + "\" already in use");

  // Give the field the same number of entries as _posname
  // (which must exist)
  Field<T> f(name, shape, _x.size());
  _fields.emplace(std::make_pair(name, std::move(f)));
}
//------------------------------------------------------------------------
template <std::floating_point T>
void Particles<T>::sort_by_cell(std::size_t num_cells)
{
  dolfinx::common::Timer timer("leopart::Particles::sort_by_cell");

  const std::size_t num_slots = _particle_to_cell.size();
  for (const std::int32_t c : _particle_to_cell)
    num_cells = std::max<std::size_t>(num_cells, c + 1);

  // Count the particles in each cell, then offsets[c] is the new index
  // of the first particle in cell c
  std::vector<std::size_t> offsets(num_cells + 1, 0);
  for (const std::int32_t c : _particle_to_cell)
  {
    if (c != INVALID_CELL)
      ++offsets[c + 1];
  }
  std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
  const std::size_t num_particles = offsets.back();

  // src[i] is the current index of the particle that moves to index i.
  // Visiting particles in index order makes the sort stable.
  std::vector<std::size_t> src(num_particles);
  {
    std::vector<std::size_t> next(offsets.begin(), offsets.end() - 1);
    for (std::size_t p = 0; p < num_slots; ++p)
    {
      const std::int32_t c = _particle_to_cell[p];
      if (c != INVALID_CELL)
        src[next[c]++] = p;
    }
  }

  // Move the particle data, unless it is already in place
  const bool identity = num_particles == num_slots
    and std::ranges::equal(src, std::views::iota(std::size_t(0), num_slots));
  if (!identity)
  {
    _x.gather(src);
    for (auto& [name, f] : _fields)
      f.gather(src);
  }

  _particle_to_cell.resize(num_particles);
  for (std::size_t c = 0; c < num_cells; ++c)
  {
    std::fill(_particle_to_cell.begin() + offsets[c],
              _particle_to_cell.begin() + offsets[c + 1],
              static_cast<std::int32_t>(c));
  }
  _cell_offsets = std::move(offsets);
}
//------------------------------------------------------------------------
template <std::floating_point T>
void Particles<T>::relocate_bbox_on_proc(
  const dolfinx::mesh::Mesh<T>& mesh, std::span<const std::size_t> pidxs)
{
  dolfinx::common::Timer timer("leopart::Particles::relocate_bbox_on_proc");

  // TODO: Handle ghosts
  const int tdim = mesh.topology()->dim();
  std::shared_ptr<const dolfinx::common::IndexMap> map =
    mesh.topology()->index_map(tdim);
  const std::size_t total_cells = map->size_local();

  // Create bbox tree for mesh
  std::vector<std::int32_t> cells(total_cells);
  std::iota(cells.begin(), cells.end(), 0);
  dolfinx::geometry::BoundingBoxTree<T> tree(
    mesh, tdim, 0.0, std::span<const std::int32_t>(cells));

  // Get positions of required pidxs, padded to 3 components
  const std::vector<T> xp = positions_padded(pidxs);

  dolfinx::common::Timer timer1("leopart::Particles::relocate_bbox_on_proc collisions");
  dolfinx::graph::AdjacencyList<std::int32_t> cell_candidates =
    dolfinx::geometry::compute_collisions<T>(tree, std::span<const T>(xp));
  dolfinx::graph::AdjacencyList<std::int32_t> cells_collided =
    dolfinx::geometry::compute_colliding_cells<T>(
      mesh, cell_candidates, std::span<const T>(xp));
  timer1.stop();
  timer1.flush();

  // Assign each particle its new cell. Particles outside the mesh are
  // dropped by the sort.
  dolfinx::common::Timer timer2("leopart::Particles::relocate_bbox_on_proc post process");
  for (std::int32_t l = 0; l < cells_collided.num_nodes(); ++l)
  {
    std::span<const std::int32_t> collided = cells_collided.links(l);
    _particle_to_cell[pidxs[l]] = collided.empty() ? INVALID_CELL : collided[0];
  }
  timer2.stop();
  timer2.flush();

  sort_by_cell(total_cells);
}
//------------------------------------------------------------------------
template <std::floating_point T>
void Particles<T>::relocate_bbox(const dolfinx::mesh::Mesh<T>& mesh)
{
  std::vector<std::size_t> pidxs(num_particles());
  std::iota(pidxs.begin(), pidxs.end(), 0);
  relocate_bbox(mesh, pidxs);
}
//------------------------------------------------------------------------
template <std::floating_point T>
void Particles<T>::relocate_bbox(
  const dolfinx::mesh::Mesh<T>& mesh, std::span<const std::size_t> pidxs)
{
  dolfinx::common::Timer timer("leopart::Particles::relocate_bbox");

  if (dolfinx::MPI::size(mesh.comm()) == 1)
  {
    relocate_bbox_on_proc(mesh, pidxs);
    return;
  }
  const std::int32_t rank = dolfinx::MPI::rank(mesh.comm());
  const std::size_t xdim = _x.value_size();

  // TODO: Handle ghosts
  std::shared_ptr<const dolfinx::common::IndexMap> map =
    mesh.topology()->index_map(mesh.topology()->dim());
  const std::size_t total_cells = map->size_local();

  // Find ownership of the geometry points
  const auto [src_owner, dest_owner, dest_points, dest_cells, dest_data] =
    determine_point_ownership(mesh, pidxs);
  std::span<const T> dest_points_span(dest_points);

  // Particles leaving this process, to another process or out of the
  // mesh, are dropped by the sort
  std::vector<std::size_t> pidxs_on_proc;
  for (std::size_t i = 0; i < src_owner.size(); ++i)
  {
    if (src_owner[i] == rank)
      pidxs_on_proc.push_back(pidxs[i]);
    else
      _particle_to_cell[pidxs[i]] = INVALID_CELL;
  }

  // Assign particles staying on this process their new cell, and count
  // those arriving from other processes
  std::size_t num_incoming = 0;
  {
    std::size_t on_proc_offset = 0;
    for (std::size_t i = 0; i < dest_owner.size(); ++i)
    {
      if (dest_owner[i] == rank)
        _particle_to_cell[pidxs_on_proc[on_proc_offset++]] = dest_cells[i];
      else
        ++num_incoming;
    }
  }

  // Append the arriving particles. Communicated points are padded to 3
  // components.
  std::size_t p_new = append(num_incoming);
  std::span<T> x_data = _x.data();
  for (std::size_t i = 0; i < dest_owner.size(); ++i)
  {
    if (dest_owner[i] == rank)
      continue;
    std::copy_n(dest_points_span.begin() + i * 3, xdim,
                x_data.begin() + p_new * xdim);
    _particle_to_cell[p_new] = dest_cells[i];
    for (const auto& [field_name, field_data] : dest_data)
    {
      Field<T>& f = field(field_name);
      const std::size_t field_vs = f.value_size();
      std::copy_n(field_data.begin() + i * field_vs, field_vs,
                  f.data(p_new).begin());
    }
    ++p_new;
  }

  sort_by_cell(total_cells);
}
//------------------------------------------------------------------------
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
           std::vector<std::int32_t>, std::map<std::string, std::vector<T>>>
Particles<T>::determine_point_ownership(
  const dolfinx::mesh::Mesh<T>& mesh,
  std::span<const std::size_t> pidxs,
  T padding)
{
  using namespace dolfinx;
  using namespace dolfinx::geometry;

  dolfinx::common::Timer timer("leopart::Particles::determine_point_ownership");
  
  MPI_Comm comm = mesh.comm();

  dolfinx::common::Timer timer1("leopart::Particles::determine_point_ownership global collisions");
  // Create a global bounding-box tree to find candidate processes with
  // cells that could collide with the points
  const int tdim = mesh.topology()->dim();
  auto cell_map = mesh.topology()->index_map(tdim);
  const std::int32_t num_cells = cell_map->size_local();
  // NOTE: Should we send the cells in as input?
  std::vector<std::int32_t> cells(num_cells, 0);
  std::iota(cells.begin(), cells.end(), 0);
  BoundingBoxTree<T> bb(
    mesh, tdim, padding, std::span<const std::int32_t>(cells));
  BoundingBoxTree<T> global_bbtree = bb.create_global_tree(comm);

  // Get positions of required pidxs, padded to 3 components
  const std::vector<T> points = positions_padded(pidxs);

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

  timer1.stop();
  timer1.flush();
  // -----------------------------------------------------------------------
  // Send geometry to global process colliding ranks
  // -----------------------------------------------------------------------
  dolfinx::common::Timer timer2("leopart::Particles::determine_point_ownership Send geometry to global process colliding ranks");

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
      for (const auto& p : collisions.links(i))
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
      for (const auto& p : collisions.links(i))
      {
        const int neighbor = rank_to_neighbor[p];
        const int pos = send_offsets[neighbor] + counter[neighbor];
        std::copy_n(std::next(data.begin(), i * value_size), value_size,
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
        dolfinx::MPI::mpi_t<T>, received_data.data(), recv_sizes.data(),
        recv_offsets.data(), dolfinx::MPI::mpi_t<T>, forward_comm);

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

  std::map<std::string, const forward_comm_result> field_comm_data;
  for (const auto& [field_name, field] : _fields)
  {
    std::span<const T> field_data = field.data();
    const std::size_t field_vs = field.value_size();

    std::vector<T> field_sub_data(pidxs.size() * field_vs, 0.0);
    for (std::size_t i = 0; i < pidxs.size(); ++i)
      std::copy_n(field.data(pidxs[i]).begin(), field_vs,
                  field_sub_data.begin() + i * field_vs);
    
    const forward_comm_result res = send_data_foward(
      field_sub_data, field_vs);
    field_comm_data.emplace(field_name, std::move(res));
  }

  timer2.stop();
  timer2.flush();
  // -----------------------------------------------------------------------
  // Compute closest entities on the owning ranks
  // -----------------------------------------------------------------------
  dolfinx::common::Timer timer3("leopart::Particles::determine_point_ownership Compute closest entities on owning ranks");

  // Get mesh geometry for closest entity
  const mesh::Geometry<T>& geometry = mesh.geometry();
  if (geometry.cmaps().size() > 1)
    throw std::runtime_error("Mixed topology not supported");
  std::span<const T> geom_dofs = geometry.x();
  auto x_dofmap = geometry.dofmaps().front();

  // Compute candidate cells for collisions (and extrapolation)
  const graph::AdjacencyList<std::int32_t> candidate_collisions
      = compute_collisions(bb, std::span<const T>(res_x.received_data.data(),
                                                  res_x.received_data.size()));

  // Each process checks which local cell is closest and computes the squared
  // distance to the cell
  const int rank = dolfinx::MPI::rank(comm);
  std::vector<std::int32_t> cell_indicator(res_x.received_data.size() / 3);
  std::vector<std::int32_t> closest_cells(res_x.received_data.size() / 3);
  for (std::size_t p = 0; p < res_x.received_data.size(); p += 3)
  {
    std::array<T, 3> point;
    std::copy_n(std::next(res_x.received_data.begin(), p), 3, point.begin());
    // Find first collding cell among the cells with colliding bounding boxes
    const int colliding_cell = geometry::compute_first_colliding_cell(
        mesh, candidate_collisions.links(p / 3), point,
        10 * std::numeric_limits<T>::epsilon());
    // If a collding cell is found, store the rank of the current process
    // which will be sent back to the owner of the point
    cell_indicator[p / 3] = (colliding_cell >= 0) ? rank : -1;
    // Store the cell index for lookup once the owning processes has determined
    // the ownership of the point
    closest_cells[p / 3] = colliding_cell;
  }

  timer3.stop();
  timer3.flush();
  // -----------------------------------------------------------------------
  // Communicate reverse the squared differences
  // -----------------------------------------------------------------------
  dolfinx::common::Timer timer4("leopart::Particles::determine_point_ownership Communicate reverse the squared differences");

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

  std::vector<std::int32_t> recv_ranks(res_x.recv_offsets.back());
  MPI_Neighbor_alltoallv(cell_indicator.data(), res_x.send_sizes.data(),
                         res_x.send_offsets.data(), MPI_INT32_T, recv_ranks.data(),
                         res_x.recv_sizes.data(), res_x.recv_offsets.data(), MPI_INT32_T,
                         reverse_comm);

  timer4.stop();
  timer4.flush();
  // -----------------------------------------------------------------------
  // For each point find the owning process which minimises the square
  // distance to a cell
  // -----------------------------------------------------------------------
  dolfinx::common::Timer timer5("leopart::Particles::determine_point_ownership Find owning process minimise square dist");

  std::vector<std::int32_t> point_owners(points.size() / 3, -1);
  for (std::size_t i = 0; i < res_x.unpack_map.size(); i++)
  {
    const std::int32_t pos = res_x.unpack_map[i];
    // Only insert new owner if no owner has previously been found
    if (recv_ranks[i] >= 0 && point_owners[pos] == -1)
      point_owners[pos] = recv_ranks[i];
  }

  // Create extrapolation marker for those points already sent to other
  // process
  std::vector<std::uint8_t> send_extrapolate(res_x.recv_offsets.back());
  for (std::int32_t i = 0; i < res_x.recv_offsets.back(); i++)
  {
    const std::int32_t pos = res_x.unpack_map[i];
    send_extrapolate[i] = point_owners[pos] == -1;
  }

  timer5.stop();
  timer5.flush();
  // -----------------------------------------------------------------------
  // Communicate forward the destination ranks for each point
  // -----------------------------------------------------------------------
  dolfinx::common::Timer timer6("leopart::Particles::determine_point_ownership Communicate forward destination ranks");

  // Swap communication direction, to send extrapolation marker to other
  // processes
  std::swap(res_x.send_sizes, res_x.recv_sizes);
  std::swap(res_x.send_offsets, res_x.recv_offsets);
  std::vector<std::uint8_t> dest_extrapolate(res_x.recv_offsets.back());
  MPI_Neighbor_alltoallv(send_extrapolate.data(), res_x.send_sizes.data(),
                         res_x.send_offsets.data(), MPI_UINT8_T,
                         dest_extrapolate.data(), res_x.recv_sizes.data(),
                         res_x.recv_offsets.data(), MPI_UINT8_T, forward_comm);
  
  std::vector<T> squared_distances(res_x.received_data.size() / 3, -1);

  for (std::size_t i = 0; i < dest_extrapolate.size(); i++)
  {
    if (dest_extrapolate[i] == 1)
    {
      assert(closest_cells[i] == -1);
      std::array<T, 3> point;
      std::copy_n(std::next(res_x.received_data.begin(), 3 * i), 3, point.begin());

      // Find shortest distance among cells with colldiing bounding box
      T shortest_distance = std::numeric_limits<T>::max();
      std::int32_t closest_cell = -1;
      for (auto cell : candidate_collisions.links(i))
      {
        auto dofs = md::submdspan(x_dofmap, cell, md::full_extent);
        std::vector<T> nodes(3 * dofs.size());
        for (std::size_t j = 0; j < dofs.size(); ++j)
        {
          const std::size_t pos = 3 * dofs[j];
          for (std::size_t k = 0; k < 3; ++k)
            nodes[3 * j + k] = geom_dofs[pos + k];
        }
        const std::array<T, 3> d = compute_distance_gjk<T>(
            std::span<const T>(point.data(), point.size()),
            std::span<const T>(nodes));
        if (T current_distance = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            current_distance < shortest_distance)
        {
          shortest_distance = current_distance;
          closest_cell = cell;
        }
      }
      closest_cells[i] = closest_cell;
      squared_distances[i] = shortest_distance;
    }
  }

  std::swap(res_x.recv_sizes, res_x.send_sizes);
  std::swap(res_x.recv_offsets, res_x.send_offsets);

  // Get distances from closest entity of points that were on the other process
  std::vector<T> recv_distances(res_x.recv_offsets.back());
  MPI_Neighbor_alltoallv(
      squared_distances.data(), res_x.send_sizes.data(), res_x.send_offsets.data(),
      dolfinx::MPI::mpi_t<T>, recv_distances.data(), res_x.recv_sizes.data(),
      res_x.recv_offsets.data(), dolfinx::MPI::mpi_t<T>, reverse_comm);

  // Update point ownership with extrapolation information
  std::vector<T> closest_distance(res_x.unpack_map.size(),
                                  std::numeric_limits<T>::max());
  for (std::size_t i = 0; i < out_ranks.size(); i++)
  {
    for (std::int32_t j = res_x.recv_offsets[i]; j < res_x.recv_offsets[i + 1]; j++)
    {
      const std::int32_t pos = res_x.unpack_map[j];
      auto current_dist = recv_distances[j];
      // Update if closer than previous guess and was found
      if (auto d = closest_distance[pos];
          (current_dist > 0) and (current_dist < d))
      {
        point_owners[pos] = out_ranks[i];
        closest_distance[pos] = current_dist;
      }
    }
  }

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
      dolfinx::MPI::mpi_t<std::int32_t>, dest_ranks.data(),
      res_x.recv_sizes.data(), res_x.recv_offsets.data(),
      dolfinx::MPI::mpi_t<std::int32_t>, forward_comm);

  timer6.stop();
  timer6.flush();
  // -----------------------------------------------------------------------
  // On the destination ranks select the appropriate data and return
  // -----------------------------------------------------------------------
  dolfinx::common::Timer timer7("leopart::Particles::determine_point_ownership On the destination ranks select the appropriate data and return");

  // Unpack dest ranks if point owner is this rank
  std::vector<std::int32_t> owned_recv_ranks;
  owned_recv_ranks.reserve(res_x.recv_offsets.back());
  std::vector<T> owned_recv_points;
  std::vector<std::int32_t> owned_recv_cells;

  std::map<std::string, std::vector<T>> owned_recv_data;

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

        for (const auto& [field_name, field_comm_res] : field_comm_data)
        {
          const std::vector<T>& field_data = field_comm_res.received_data;
          std::vector<T>& recv_data = owned_recv_data[field_name];
          recv_data.insert(recv_data.end(),
            std::next(field_data.cbegin(), field_comm_res.value_size * j),
            std::next(field_data.cbegin(), field_comm_res.value_size * (j + 1)));
        }
      }
    }
  }

  MPI_Comm_free(&forward_comm);
  MPI_Comm_free(&reverse_comm);

  timer7.stop();
  timer7.flush();
  return std::make_tuple(
    std::move(point_owners), std::move(owned_recv_ranks),
    std::move(owned_recv_points), std::move(owned_recv_cells),
    std::move(owned_recv_data));
}
//------------------------------------------------------------------------
template <std::floating_point T>
void Particles<T>::generate_minimum_particles_per_cell(
  const dolfinx::mesh::Mesh<T>& mesh,
  const std::size_t np_per_cell)
{
  const std::int32_t num_cells = mesh.topology()->index_map(
    mesh.topology()->dim())->size_local();

  std::vector<std::int32_t> cells_to_populate;
  std::vector<std::size_t> np_per_cell_vec;
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    const std::size_t np_cell = (static_cast<std::size_t>(c) < this->num_cells())
      ? _cell_offsets[c + 1] - _cell_offsets[c] : 0;
    if (np_cell >= np_per_cell)
      continue;

    cells_to_populate.push_back(c);
    np_per_cell_vec.push_back(np_per_cell - np_cell);
  }

  const auto [xp_new, p2cell_new] = leopart::generation::mesh_fill(
    mesh, np_per_cell_vec, cells_to_populate);

  // Generated points have gdim components, whereas the particles may
  // store a different number (e.g. 2D positions padded to 3 components)
  const std::size_t gdim = mesh.geometry().dim();
  const std::size_t xdim = _x.value_size();
  const std::size_t ncopy = std::min(gdim, xdim);
  const std::size_t first = append(p2cell_new.size());
  std::span<T> x_data = _x.data();
  for (std::size_t i = 0; i < p2cell_new.size(); ++i)
  {
    std::copy_n(xp_new.begin() + i * gdim, ncopy,
                x_data.begin() + (first + i) * xdim);
    _particle_to_cell[first + i] = p2cell_new[i];
  }
  sort_by_cell(num_cells);
}
//------------------------------------------------------------------------
template class leopart::Particles<double>;
//------------------------------------------------------------------------
