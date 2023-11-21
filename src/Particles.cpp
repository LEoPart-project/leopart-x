// Copyright: (c) 2020-2023 Chris Richardson, Jakob Maljaars and Nathan Sime
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <cassert>
#include <dolfinx.h>

#include "generation.h"
#include "Particles.h"

using namespace leopart;

template <std::floating_point T>
Particles<T>::Particles(const std::vector<T>& x,
                        const std::vector<std::int32_t>& cells,
                        const std::size_t gdim) : _particle_to_cell(cells)
{
  // Find max cell index, and create cell->particle map
  std::int32_t max_cell = 0;
  if (!cells.empty())
  {
    auto max_cell_it = std::max_element(cells.begin(), cells.end());
    max_cell = *max_cell_it;
  }
  _cell_to_particle.resize(max_cell + 1);
  for (std::size_t p = 0; p < cells.size(); ++p)
    _cell_to_particle[cells[p]].push_back(p);

  // Create position data field
  const std::size_t rows = x.size() / gdim;
  Field<T> fx(_posname, {gdim}, rows);
  std::copy(x.cbegin(), x.cend(), fx.data().begin());
  _fields.emplace(std::make_pair(_posname, std::move(fx)));
}
//------------------------------------------------------------------------
template <std::floating_point T>
std::size_t Particles<T>::add_particle(
  std::span<const T> x, std::int32_t cell)
{
  assert(cell < _cell_to_particle.size());
  assert(x.size() == _fields.at(_posname).value_shape()[0]);
  std::size_t pidx;
  if (_free_list.empty())
  {
    // Need to create a new particle, and extend associated fields
    // Get new particle index from size of _posname field
    // (which must exist)
    pidx = _fields.at(_posname).size();
    // Resize all fields
    for (auto& [f_name, f] : _fields)
      f.resize(f.size() + 1);
    _particle_to_cell.resize(_particle_to_cell.size() + 1);
  }
  else
  {
    pidx = _free_list.back();
    _free_list.pop_back();
  }

  _cell_to_particle[cell].push_back(pidx);
  _particle_to_cell[pidx] = cell;
  std::copy_n(x.begin(), x.size(), _fields.at(_posname).data(pidx).begin());
  return pidx;
}
//------------------------------------------------------------------------
template <std::floating_point T>
void Particles<T>::delete_particle(std::int32_t cell, std::size_t p_local)
{
  // delete cell to particle entry
  assert(cell < _cell_to_particle.size());
  std::vector<std::size_t>& cp = _cell_to_particle[cell];
  assert(p_local < cp.size());
  std::size_t pidx = cp[p_local];
  cp.erase(cp.begin() + p_local);

  assert(pidx < _particle_to_cell.size());
  _particle_to_cell[pidx] = INVALID_CELL;

  _free_list.push_back(pidx);
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
  Field<T> f(name, shape, _fields.at(_posname).size());
  _fields.emplace(std::make_pair(name, std::move(f)));
}
//------------------------------------------------------------------------
template <std::floating_point T>
void Particles<T>::relocate_bbox_on_proc(
  const dolfinx::mesh::Mesh<T>& mesh, std::span<const std::size_t> pidxs)
{
  dolfinx::common::Timer timer("leopart::Particles::relocate_bbox_on_proc");

  // Resize member if required, TODO: Handle ghosts
  std::shared_ptr<const dolfinx::common::IndexMap> map =
    mesh.topology()->index_map(mesh.topology()->dim());
  const std::size_t total_cells = map->size_local();
  if (_cell_to_particle.size() < total_cells)
    _cell_to_particle.resize(total_cells);

  // Create bbox tree for mesh
  std::vector<std::int32_t> cells(total_cells);
  std::iota(cells.begin(), cells.end(), 0);
  dolfinx::geometry::BoundingBoxTree<T> tree(
    mesh, mesh.topology()->dim(), cells);

  // Get positions of required pidxs
  std::span<const T> xp_all = field(_posname).data();
  const std::size_t gdim = field(_posname).value_shape()[0];
  std::vector<T> xp(pidxs.size() * gdim, 0.0);
  for (std::size_t i = 0; i < pidxs.size(); ++i)
  {
    std::copy_n(xp_all.begin() + pidxs[i] * gdim, gdim, xp.begin() + i * gdim);
  }

  dolfinx::common::Timer timer1("leopart::Particles::relocate_bbox_on_proc collisions");
  dolfinx::graph::AdjacencyList<std::int32_t> cell_candidates =
    dolfinx::geometry::compute_collisions<T>(tree, xp);
  dolfinx::graph::AdjacencyList<std::int32_t> cells_collided =
    dolfinx::geometry::compute_colliding_cells<T>(mesh, cell_candidates, xp);
  timer1.stop();

  dolfinx::common::Timer timer2("leopart::Particles::relocate_bbox_on_proc post process");
  std::vector<std::size_t> lost;
  for (std::size_t l = 0; l < cells_collided.num_nodes(); ++l)
  {
    if (cells_collided.links(l).empty())
    {
      lost.push_back(pidxs[l]);
    }
    else
    {
      const std::size_t pidx = pidxs[l];
      const std::int32_t new_cell = cells_collided.links(l)[0];
      if (_particle_to_cell[pidx] == new_cell)
        continue;

      // Update old and new cells' particles
      const auto [old_cell, local_pidx] = global_to_local(pidx);
      if (old_cell != INVALID_CELL)
      {
        std::vector<std::size_t>& cps = _cell_to_particle[old_cell];
        cps.erase(cps.begin() + local_pidx);
      }
      _cell_to_particle[new_cell].push_back(pidx);
      
      // Update particle's cell
      _particle_to_cell[pidx] = new_cell;
    }
  }

  for (const std::size_t pidx : lost)
  {
    const auto [old_cell, local_pidx] = global_to_local(pidx);
    delete_particle(old_cell, local_pidx);
  }
  timer2.stop();
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
  const std::size_t gdim = field(_posname).value_shape()[0];

  // Resize member if required, TODO: Handle ghosts
  std::shared_ptr<const dolfinx::common::IndexMap> map =
    mesh.topology()->index_map(mesh.topology()->dim());
  const std::size_t total_cells = map->size_local();
  if (_cell_to_particle.size() < total_cells)
    _cell_to_particle.resize(total_cells);

  // Find ownership of the geometry points
  const auto [src_owner, dest_owner, dest_points, dest_cells, dest_data] =
    determine_point_ownership(mesh, pidxs);
  std::span<const T> dest_points_span(dest_points);

  // Find lost particles (outside of the geometry)
  std::vector<std::size_t> lost;

  // Mark particles located outside the domain as lost.
  // Delete the local particles which are now off process to make room
  // for (potentially) incoming particles
  std::vector<std::size_t> pidxs_on_proc;
  for (std::size_t i = 0; i < src_owner.size(); ++i)
  {
    if (src_owner[i] != rank)
    {
      if (src_owner[i] < 0)
        lost.push_back(pidxs[i]);
      const auto [cell, local_pidx] = global_to_local(pidxs[i]);
      delete_particle(cell, local_pidx);
    }
    else
      pidxs_on_proc.push_back(pidxs[i]);
  }

  // Curate particles which are still local or those coming
  // from another process
  std::size_t on_proc_offset = 0;
  for (std::size_t i = 0; i < dest_owner.size(); ++i)
  {
    const std::int32_t new_cell = dest_cells[i];
    if (dest_owner[i] == rank)
    {
      // Particle is already on process
      const std::size_t pidx = pidxs_on_proc[on_proc_offset++];
      if (new_cell == _particle_to_cell[pidx])
        continue;

      // Particle changed cell: update old and new cells' particles
      const auto [old_cell, local_pidx] = global_to_local(pidx);
      std::vector<std::size_t>& cps = _cell_to_particle[old_cell];
      cps.erase(cps.begin() + local_pidx);
      _cell_to_particle[new_cell].push_back(pidx);

      // Update particle's cell
      _particle_to_cell[pidx] = new_cell;
    }
    else
    {
      // Particle came from another process
      const std::size_t new_pidx = add_particle(
        dest_points_span.subspan(i * gdim, gdim), new_cell);
      for (const auto& [field_name, field_data] : dest_data)
      {
        const std::size_t field_vs = field(field_name).value_size();
        std::copy_n(
          field_data.begin() + i * field_vs, field_vs,
          field(field_name).data(new_pidx).begin());
      }
    }
  }
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
  BoundingBoxTree bb(mesh, tdim, cells, padding);
  BoundingBoxTree midpoint_tree = create_midpoint_tree(mesh, tdim, cells);
  BoundingBoxTree global_bbtree = bb.create_global_tree(comm);

  // Get positions of required pidxs
  std::span<const T> xp_all = field("x").data();
  const std::size_t gdim = field("x").value_shape()[0];
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

  timer1.stop();
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
        std::copy(std::next(data.begin(), i * value_size),
                  std::next(data.begin(), (i + 1) * value_size),
                  std::next(send_data.begin(), pos)); // todo: copy_n
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

  std::map<std::string, const forward_comm_result> field_comm_data;
  for (const auto& [field_name, field] : _fields)
  {
    if (field_name == _posname)
      continue; // position data is handled separately

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
  // -----------------------------------------------------------------------
  // Compute closest entities on the owning ranks
  // -----------------------------------------------------------------------
  dolfinx::common::Timer timer3("leopart::Particles::determine_point_ownership Compute closest entities on owning ranks");

  // Get mesh geometry for closest entity
  const mesh::Geometry<T>& geometry = mesh.geometry();
  if (geometry.cmaps().size() > 1)
    throw std::runtime_error("Mixed topology not supported");
  std::span<const T> geom_dofs = geometry.x();
  auto x_dofmap = geometry.dofmap();

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
        auto dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::
            MDSPAN_IMPL_PROPOSED_NAMESPACE::submdspan(
                x_dofmap, cell, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        std::vector<T> nodes(3 * dofs.size());
        for (std::size_t j = 0; j < dofs.size(); ++j)
        {
          const int pos = 3 * dofs[j];
          for (std::size_t k = 0; k < 3; ++k)
            nodes[3 * j + k] = geom_dofs[pos + k];
        }
        const std::array<T, 3> d = compute_distance_gjk<T>(
            std::span<const T>(point.data(), point.size()), nodes);
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
      dolfinx::MPI::mpi_type<T>(), recv_distances.data(), res_x.recv_sizes.data(),
      res_x.recv_offsets.data(), dolfinx::MPI::mpi_type<T>(), reverse_comm);

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
      dolfinx::MPI::mpi_type<std::int32_t>(), dest_ranks.data(),
      res_x.recv_sizes.data(), res_x.recv_offsets.data(),
      dolfinx::MPI::mpi_type<std::int32_t>(), forward_comm);

  timer6.stop();
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
  for (std::int32_t i = 0; i < num_cells; ++i)
  {
    if (_cell_to_particle[i].size() >= np_per_cell)
      continue;

    cells_to_populate.push_back(i);
    np_per_cell_vec.push_back(np_per_cell - _cell_to_particle[i].size());
  }

  const auto [xp_new, p2cell_new] = leopart::generation::mesh_fill(
    mesh, np_per_cell_vec, cells_to_populate);
  
  const int gdim = mesh.geometry().dim();
  const std::size_t num_new_p = p2cell_new.size();
  for (std::size_t pidx = 0; pidx < num_new_p; ++pidx)
  {
    std::span<const T> x(xp_new.data() + pidx * gdim, gdim);
    add_particle(x, p2cell_new[pidx]);
  }
}
//------------------------------------------------------------------------
template class Particles<double>;
//------------------------------------------------------------------------