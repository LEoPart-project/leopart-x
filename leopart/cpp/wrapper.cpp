// Copyright (c) 2020-2023 Chris Richardson, Jakob Maljaars and Nathan Sime
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Field.h"
#include "Particles.h"
#include "advect.h"
#include "generation.h"
#include "transfer.h"

#include <array>
#include <concepts>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <dolfinx.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

using leopart::Field;
using leopart::Particles;

using dtype = double;      // Particle dtype
using dtype_geom = double; // Geometry dtype

namespace
{
/// Create a numpy array which takes ownership of the data of a
/// std::vector.
template <typename V>
auto as_nbarray(V&& x, std::size_t ndim, const std::size_t* shape)
{
  using _V = std::decay_t<V>;
  _V* ptr = new _V(std::forward<V>(x));
  return nb::ndarray<typename _V::value_type, nb::numpy>(
      ptr->data(), ndim, shape,
      nb::capsule(ptr, [](void* p) noexcept { delete (_V*)p; }));
}

template <typename V>
auto as_nbarray(V&& x, std::initializer_list<std::size_t> shape)
{
  return as_nbarray(std::forward<V>(x), shape.size(), shape.begin());
}

/// Return generated points as a 2D numpy array and the associated cells
template <std::floating_point T>
auto points_and_cells(std::vector<T>&& x, std::vector<std::int32_t>&& cells,
                      const std::size_t xdim)
{
  const std::size_t n = (xdim == 0) ? 0 : x.size() / xdim;
  return std::make_tuple(as_nbarray(std::move(x), {n, xdim}),
                         std::move(cells));
}
} // namespace

NB_MODULE(cpp, m)
{
  m.doc() = "LEoPart-x C++ interface";

  // Class Field
  nb::class_<Field<dtype>>(m, "Field")
      .def("data",
          [](Field<dtype>& self, std::size_t p)
          {
            std::span<dtype> array = self.data(p);
            return nb::ndarray<dtype, nb::numpy>(
              array.data(), {array.size()});
          },
          nb::rv_policy::reference_internal, nb::arg("p"))
      .def("data",
          [](Field<dtype>& self)
          {
            const std::size_t value_size = self.value_size();
            std::span<dtype> array = self.data();
            return nb::ndarray<dtype, nb::numpy>(
              array.data(), {array.size() / value_size, value_size});
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro("value_shape", &Field<dtype>::value_shape)
      .def_prop_ro("value_size", &Field<dtype>::value_size)
      .def_ro("name", &Field<dtype>::name)
      .def("resize", &Field<dtype>::resize, nb::arg("n"));

  // Class Particles
  nb::class_<Particles<dtype>>(m, "Particles")
      .def(nb::init<const std::vector<dtype>&, const std::vector<std::int32_t>&,
           const std::size_t>(),
           nb::arg("x"), nb::arg("cells"), nb::arg("gdim"))
      .def("__init__",
           [](Particles<dtype>* self,
              nb::ndarray<const dtype, nb::c_contig> px,
              const std::vector<std::int32_t>& p_cells)
           {
             // Number of components stored per particle position
             const std::size_t xdim = (px.ndim() == 2) ? px.shape(1) : 3;

             if (p_cells.empty())
             {
               if (px.size() != 0)
                 throw std::invalid_argument(
                   "Number of particles and particle cells should be "
                   "equivalent");
               new (self) Particles<dtype>({}, {}, xdim);
               return;
             }

             if (px.ndim() != 2)
               throw std::invalid_argument(
                 "Particle positions must be a 2D array");
             if (px.shape(0) != p_cells.size())
               throw std::invalid_argument(
                 "Number of particles and particle cells should be "
                 "equivalent");
             if ((xdim != 2) and (xdim != 3))
               throw std::invalid_argument(
                 "Particle position value size expected to be 2 or 3");

             std::vector<dtype> p_data(px.data(), px.data() + px.size());
             new (self) Particles<dtype>(p_data, p_cells, xdim);
           },
           nb::arg("x"), nb::arg("cells"))
      .def("x", nb::overload_cast<>(&Particles<dtype>::x),
           nb::rv_policy::reference_internal)
      .def("add_field", &Particles<dtype>::add_field,
           nb::arg("name"), nb::arg("shape"))
      .def("add_particle",
          [](Particles<dtype>& self, const std::vector<dtype>& px,
             std::int32_t cell)
          {
            return self.add_particle(px, cell);
          },
          nb::arg("x"), nb::arg("cell"))
      .def("delete_particle", &Particles<dtype>::delete_particle,
           nb::arg("cell"), nb::arg("p_local"))
      .def("field",
           nb::overload_cast<std::string>(&Particles<dtype>::field),
           nb::rv_policy::reference_internal, nb::arg("name"))
      .def("global_to_local",
           [](const Particles<dtype>& self, std::size_t p)
           {
             const auto [cell, p_local] = self.global_to_local(p);
             return std::make_pair(cell, p_local);
           },
           nb::arg("p"))
      .def("sort_by_cell", &Particles<dtype>::sort_by_cell)
      .def("cell_to_particle", &Particles<dtype>::cell_to_particle)
      .def("particle_to_cell", &Particles<dtype>::particle_to_cell)
      .def("relocate_bbox",
           [](Particles<dtype>& self,
              const dolfinx::mesh::Mesh<dtype_geom>& mesh,
              const std::vector<std::size_t>& pidxs)
           {
             self.relocate_bbox(mesh, pidxs);
           },
           nb::arg("mesh"), nb::arg("pidxs"))
      .def("relocate_bbox_on_proc",
           [](Particles<dtype>& self,
              const dolfinx::mesh::Mesh<dtype_geom>& mesh,
              const std::vector<std::size_t>& pidxs)
           {
             self.relocate_bbox_on_proc(mesh, pidxs);
           },
           nb::arg("mesh"), nb::arg("pidxs"))
      .def("generate_minimum_particles_per_cell",
           [](Particles<dtype>& self,
              const dolfinx::mesh::Mesh<dtype_geom>& mesh,
              const std::size_t np_per_cell)
           {
             self.generate_minimum_particles_per_cell(mesh, np_per_cell);
           },
           nb::arg("mesh"), nb::arg("np_per_cell"))
      .def("active_pidxs",
           [](Particles<dtype>& self)
           {
             std::vector<std::size_t> pidxs = self.active_pidxs();
             const std::size_t n = pidxs.size();
             return as_nbarray(std::move(pidxs), {n});
           })
      .def("field_exists", &Particles<dtype>::field_exists, nb::arg("name"));

  // Generation functions
  m.def("random_reference_tetrahedron",
        [](const std::size_t n, const std::size_t seed)
        {
          std::vector<dtype> x
            = leopart::generation::random_reference_tetrahedron<dtype>(
              n, seed);
          return as_nbarray(std::move(x), {n, 3});
        },
        nb::arg("n"), nb::arg("seed"));
  m.def("random_reference_triangle",
        [](const std::size_t n, const std::size_t seed)
        {
          std::vector<dtype> x
            = leopart::generation::random_reference_triangle<dtype>(n, seed);
          return as_nbarray(std::move(x), {n, 2});
        },
        nb::arg("n"), nb::arg("seed"));
  m.def("mesh_fill",
        [](const dolfinx::mesh::Mesh<dtype_geom>& mesh,
           std::size_t np_per_cell, const std::optional<std::size_t>& seed)
        {
          auto [xp_all, np_cells]
            = leopart::generation::mesh_fill(mesh, np_per_cell, seed);
          return points_and_cells(std::move(xp_all), std::move(np_cells),
                                  mesh.geometry().dim());
        },
        nb::arg("mesh"), nb::arg("np_per_cell"),
        nb::arg("seed").none() = nb::none());
  m.def("mesh_fill",
        [](const dolfinx::mesh::Mesh<dtype_geom>& mesh,
           const std::vector<std::size_t>& np_per_cell,
           const std::vector<std::int32_t>& cells,
           const std::optional<std::size_t>& seed)
        {
          auto [xp_all, np_cells] = leopart::generation::mesh_fill(
            mesh, std::span<const std::size_t>(np_per_cell),
            std::span<const std::int32_t>(cells), seed);
          return points_and_cells(std::move(xp_all), std::move(np_cells),
                                  mesh.geometry().dim());
        },
        nb::arg("mesh"), nb::arg("np_per_cell"), nb::arg("cells"),
        nb::arg("seed").none() = nb::none());
  m.def("generate_at_dof_coords",
        [](const dolfinx::fem::FunctionSpace<dtype_geom>& V)
        {
          auto [xp_all, np_cells]
            = leopart::generation::generate_at_dof_coords(V);
          return points_and_cells(std::move(xp_all), std::move(np_cells), 3);
        },
        nb::arg("V"));

  // Transfer functions
  m.def("find_deficient_cells",
        [](std::shared_ptr<dolfinx::fem::Function<dtype>> f,
           const Particles<dtype>& pax)
        {
          return leopart::transfer::find_deficient_cells<dtype, dtype_geom>(
            f, pax);
        },
        nb::arg("f"), nb::arg("particles"));
  m.def("transfer_to_particles",
        [](const Particles<dtype>& pax, Field<dtype>& field,
           std::shared_ptr<dolfinx::fem::Function<dtype>> f)
        {
          leopart::transfer::transfer_to_particles<dtype>(pax, field, f);
        },
        nb::arg("particles"), nb::arg("field"), nb::arg("f"));
  m.def("transfer_to_function",
        [](std::shared_ptr<dolfinx::fem::Function<dtype>> f,
           const Particles<dtype>& pax,
           const Field<dtype>& field)
        {
          leopart::transfer::transfer_to_function<dtype, dtype_geom>(
            f, pax, field);
        },
        nb::arg("f"), nb::arg("particles"), nb::arg("field"));
  m.def("transfer_to_function_constrained",
        [](std::shared_ptr<dolfinx::fem::Function<dtype>> f,
           const Particles<dtype>& pax,
           const Field<dtype>& field,
           const dtype lb, const dtype ub)
        {
          leopart::transfer::transfer_to_function_constrained<
            dtype, dtype_geom>(f, pax, field, lb, ub);
        },
        nb::arg("f"), nb::arg("particles"), nb::arg("field"),
        nb::arg("lb"), nb::arg("ub"));

  // Advection functions
  m.def("rk",
        [](Particles<dtype>& ptcls,
           const leopart::advect::Tableau<dtype>& tableau,
           std::function<std::shared_ptr<dolfinx::fem::Function<dtype>>(dtype)>
             velocity_callback,
           nb::callable relocate_callback,
           const dtype t, const dtype dt)
        {
          // Pass the particles to Python by reference rather than by copy
          auto relocate = [&relocate_callback](Particles<dtype>& p)
          {
            relocate_callback(nb::cast(&p, nb::rv_policy::reference));
          };
          leopart::advect::rk<dtype>(
            ptcls, tableau, velocity_callback, relocate, t, dt);
        },
        nb::arg("particles"), nb::arg("tableau"),
        nb::arg("velocity_callback"), nb::arg("relocation_callback"),
        nb::arg("t"), nb::arg("dt"));
  m.def("rk",
        [](const dolfinx::mesh::Mesh<dtype_geom>& mesh,
           Particles<dtype>& ptcls,
           const leopart::advect::Tableau<dtype>& tableau,
           std::function<std::shared_ptr<dolfinx::fem::Function<dtype>>(dtype)>
             velocity_callback,
           const dtype t, const dtype dt)
        {
          leopart::advect::rk<dtype>(
            mesh, ptcls, tableau, velocity_callback, t, dt);
        },
        nb::arg("mesh"), nb::arg("particles"), nb::arg("tableau"),
        nb::arg("velocity_callback"), nb::arg("t"), nb::arg("dt"));

  // Advection Butcher Tableau
  using Tableau = leopart::advect::Tableau<dtype>;
  nb::class_<Tableau>(m, "ButcherTableau")
      .def(nb::init<std::size_t, std::vector<dtype>, std::vector<dtype>,
                    std::vector<dtype>>(),
           nb::arg("order"), nb::arg("a"), nb::arg("b"), nb::arg("c"))
      .def("__init__",
           [](Tableau* self,
              nb::ndarray<const dtype, nb::ndim<2>, nb::c_contig> a,
              const std::vector<dtype>& b,
              const std::vector<dtype>& c)
           {
             const auto arg_shape_check = [](
               const std::size_t i, const std::string iname,
               const std::size_t j, const std::string jname) {
                 if (i != j)
                   throw std::runtime_error(iname + " shape does not match "
                                            + jname + " shape");
               };
             arg_shape_check(a.shape(0), "a rows", a.shape(1), "a columns");
             arg_shape_check(a.shape(0), "a rows", b.size(), "b length");
             arg_shape_check(a.shape(0), "a rows", c.size(), "c length");

             std::vector<dtype> a_vec(a.data(), a.data() + a.size());
             const std::size_t order = b.size();
             new (self) Tableau(order, a_vec, b, c);
           },
           nb::arg("a"), nb::arg("b"), nb::arg("c"))
      .def_prop_ro("a",
          [](const Tableau& self)
          {
            return nb::ndarray<const dtype, nb::numpy>(
              self.a.data(), {self.order, self.order});
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro("b",
          [](const Tableau& self)
          {
            return nb::ndarray<const dtype, nb::numpy>(
              self.b.data(), {self.b.size()});
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro("c",
          [](const Tableau& self)
          {
            return nb::ndarray<const dtype, nb::numpy>(
              self.c.data(), {self.c.size()});
          },
          nb::rv_policy::reference_internal)
      .def_ro("order", &Tableau::order)
      .def("field_name_xn", &Tableau::field_name_xn)
      .def("field_name_substep", &Tableau::field_name_substep,
           nb::arg("substep_n"))
      .def("check_and_create_fields", &Tableau::check_and_create_fields,
           nb::arg("particles"));

  // Predefined tableaus
  namespace tableaus = leopart::advect::tableaus;
  nb::module_ tableaus_module = m.def_submodule("tableaus");
  tableaus_module.def_submodule("order1")
      .def("forward_euler", &tableaus::order1::forward_euler<dtype>);
  tableaus_module.def_submodule("order2")
      .def("generic_alpha", &tableaus::order2::generic_alpha<dtype>,
           nb::arg("alpha"))
      .def("explicit_midpoint", &tableaus::order2::explicit_midpoint<dtype>)
      .def("heun", &tableaus::order2::heun<dtype>)
      .def("ralston", &tableaus::order2::ralston<dtype>);
  tableaus_module.def_submodule("order3")
      .def("generic_alpha", &tableaus::order3::generic_alpha<dtype>,
           nb::arg("alpha"))
      .def("heun", &tableaus::order3::heun<dtype>)
      .def("wray", &tableaus::order3::wray<dtype>)
      .def("ralston", &tableaus::order3::ralston<dtype>)
      .def("ssp", &tableaus::order3::ssp<dtype>);
  tableaus_module.def_submodule("order4")
      .def("classic", &tableaus::order4::classic<dtype>)
      .def("kutta1901", &tableaus::order4::kutta1901<dtype>)
      .def("ralston", &tableaus::order4::ralston<dtype>);

  // Utility functions
  m.def("evaluate_basis_functions",
        [](const dolfinx::fem::FunctionSpace<dtype_geom>& V,
           nb::ndarray<const dtype_geom, nb::ndim<2>, nb::c_contig> x,
           const std::vector<std::int32_t>& cells)
        {
          auto [basis, shape] = leopart::utils::evaluate_basis_functions<
            dtype_geom>(V, std::span<const dtype_geom>(x.data(), x.size()),
                        x.shape(1), cells);
          return as_nbarray(std::move(basis), shape.size(), shape.data());
        },
        nb::arg("V"), nb::arg("x"), nb::arg("cells"));

  m.def("create_bbox_pointcloud",
        [](std::vector<std::pair<std::array<dtype_geom, 3>, std::int32_t>> points)
        {
          return dolfinx::geometry::BoundingBoxTree<dtype_geom>(
            std::move(points));
        },
        nb::arg("points"));
}
