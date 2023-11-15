// Copyright: (c) 2020-2023 Chris Richardson, Jakob Maljaars and Nathan Sime
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Field.h"
#include "Particles.h"
#include "advect.h"
#include "generation.h"
#include "transfer.h"

#include <cstddef>
#include <concepts>
#include <vector>
#include <tuple>
#include <dolfinx.h>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;

using leopart::Field;
using leopart::Particles;

using dtype = double;      // Particle dtype
using dtype_geom = double; // Geometry dtype


PYBIND11_MODULE(cpp, m)
{
  // Class Field
  py::class_<Field<dtype>>(m, "Field")
      .def("data",
          [](Field<dtype>& self, int p)
          {
            std::span<dtype> array = self.data(p);
            return py::array_t<dtype, py::array::c_style>(
              array.size(), array.data(), py::cast(self));
          })
      .def("data",
          [](Field<dtype>& self)
          {
            const std::size_t value_size = self.value_size();
            std::span<dtype> array = self.data();
            const std::array<std::size_t, 2> data_shape{array.size() / value_size, value_size};
            return py::array_t<dtype, py::array::c_style>(
              data_shape, array.data(), py::cast(self));
          })
      .def_property_readonly("value_shape", &Field<dtype>::Field::value_shape)
      .def_property_readonly("value_size", &Field<dtype>::Field::value_size)
      .def_readonly("name", &Field<dtype>::Field::name)
      .def("resize", &Field<dtype>::resize);

  // Class Particles
  py::class_<Particles<dtype>>(m, "Particles")
      .def(py::init<std::vector<dtype>&, const std::vector<std::int32_t>&,
           const std::size_t>())
      .def(py::init(
            [](const py::array_t<dtype, py::array::c_style>& px,
               const std::vector<std::int32_t>& p_cells) {
                if (px.shape(0) != p_cells.size())
                  throw std::invalid_argument(
                    "Number of particles and particle cells should be equivalent");

                if (p_cells.empty())
                  return Particles<dtype>({}, {}, 3);

                if ((px.shape(1) != 2) and (px.shape(1) != 3))
                  throw std::invalid_argument(
                    "Particle position value size expected to be 3");
                std::vector<dtype> p_data = std::vector<dtype>(
                  px.data(), px.data() + px.size());
                return Particles(p_data, p_cells, px.shape()[1]);
               }))
      .def("add_field", &Particles<dtype>::Particles::add_field)
      .def("add_particle",
          [](Particles<dtype>& self, std::vector<dtype>& px, std::int32_t cell) {
            return self.add_particle(px, cell);
          })
      .def("delete_particle", &Particles<dtype>::Particles::delete_particle)
      .def("field",
           py::overload_cast<std::string>(&Particles<dtype>::Particles::field),
           py::return_value_policy::reference_internal)
      .def("global_to_local", &Particles<dtype>::Particles::global_to_local)
      .def("cell_to_particle", &Particles<dtype>::Particles::cell_to_particle)
      .def("particle_to_cell", &Particles<dtype>::Particles::particle_to_cell)
      .def("relocate_bbox",
           [](Particles<dtype>& self, const dolfinx::mesh::Mesh<dtype_geom>& mesh,
              const std::vector<std::size_t>& pidxs) {
            self.relocate_bbox(mesh, pidxs);
           })
      .def("relocate_bbox_on_proc",
           [](Particles<dtype>& self, const dolfinx::mesh::Mesh<dtype_geom>& mesh,
              const std::vector<std::size_t>& pidxs) {
            self.relocate_bbox_on_proc(mesh, pidxs);
           });

  // Generation functions
  m.def("random_reference_tetrahedron",
        [](const std::size_t n) {
          const int gdim = 3;
          std::vector<dtype> array = leopart::generation::random_reference_tetrahedron<dtype>(n);
          std::array<std::size_t, 2> shape = {array.size() / gdim, gdim};
          return py::array_t<dtype, py::array::c_style>(shape, array.data());
        },
        py::return_value_policy::move);
  m.def("random_reference_triangle",
        [](const std::size_t n) {
          const int gdim = 2;
          std::vector<dtype> array = leopart::generation::random_reference_triangle<dtype>(n);
          std::array<std::size_t, 2> shape = {array.size() / gdim, gdim};
          return py::array_t<dtype, py::array::c_style>(shape, array.data());
        },
        py::return_value_policy::move);
  m.def("mesh_fill",
        [](std::shared_ptr<dolfinx::mesh::Mesh<dtype_geom>> mesh, dtype density) {
          auto [xp_all, np_cells] = leopart::generation::mesh_fill(*mesh, density);
          const std::size_t gdim = mesh->geometry().dim();
          std::array<std::size_t, 2> shape = {xp_all.size() / gdim, gdim};
          auto ret_val = std::make_tuple<py::array_t<dtype, py::array::c_style>, std::vector<std::int32_t>>(
            py::array_t<dtype, py::array::c_style>(shape, xp_all.data()), std::move(np_cells));
          return ret_val;
        }, py::return_value_policy::move);
  m.def("generate_at_dof_coords",
        [](std::shared_ptr<dolfinx::fem::FunctionSpace<dtype_geom>> V) {
          auto [xp_all, np_cells] = leopart::generation::generate_at_dof_coords(*V);
          const std::size_t gdim = 3;
          std::array<std::size_t, 2> shape = {xp_all.size() / gdim, gdim};
          auto ret_val = std::make_tuple<py::array_t<dtype, py::array::c_style>, std::vector<std::int32_t>>(
            py::array_t<dtype, py::array::c_style>(shape, xp_all.data()), std::move(np_cells));
          return ret_val;
        }, py::return_value_policy::move);

  // Transfer functions
  m.def("transfer_to_particles", &leopart::transfer::transfer_to_particles<dtype>);
  m.def("transfer_to_function",
        [](std::shared_ptr<dolfinx::fem::Function<dtype>> f,
        const Particles<dtype>& pax,
        const Field<dtype>& field) {
          return leopart::transfer::transfer_to_function<dtype, dtype_geom>(
            f, pax, field);
        });
  m.def("transfer_to_function_constrained",
        [](std::shared_ptr<dolfinx::fem::Function<dtype>> f,
        const Particles<dtype>& pax,
        const Field<dtype>& field,
        const dtype lb, const dtype ub) {
          return leopart::transfer::transfer_to_function_constrained<dtype, dtype_geom>(
            f, pax, field, lb, ub);
        });

  // Advection functions
  m.def("rk",
        [](
          const dolfinx::mesh::Mesh<dtype_geom>& mesh,
          Particles<dtype>& ptcls,
          const leopart::advect::Tableau<dtype>& tableau,
          std::function<std::shared_ptr<dolfinx::fem::Function<dtype>>(dtype)> velocity_callback,
          const dtype t, const dtype dt) {
          leopart::advect::rk(mesh, ptcls, tableau, velocity_callback, t, dt);
        });

  // Advection Butcher Tableau
  py::class_<leopart::advect::Tableau<dtype>>(m, "ButcherTableau")
      .def(py::init<std::size_t, std::vector<dtype>&, std::vector<dtype>&, std::vector<dtype>&>())
      .def(py::init(
            [](const py::array_t<dtype, py::array::c_style>& a,
               const std::vector<dtype>& b,
               const std::vector<dtype>& c) {
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

                const std::vector<dtype> a_vec(a.data(), a.data() + a.size());
                const std::size_t order = b.size();
                return leopart::advect::Tableau<dtype>(order, a_vec, b, c);
               }))
      .def_property_readonly("a",
          [](leopart::advect::Tableau<dtype>& self)
          {
            leopart::utils::mdspan_t<const dtype, 2> a_md = self.a_md();
            const std::array<std::size_t, 2> data_shape{a_md.extent(0), a_md.extent(1)};
            return py::array_t<dtype, py::array::c_style>(
              data_shape, self.a.data(), py::cast(self));
          })
      .def_property_readonly("b",
          [](leopart::advect::Tableau<dtype>& self)
          {
            return py::array_t<dtype, py::array::c_style>(
              self.b.size(), self.b.data(), py::cast(self));
          })
      .def_property_readonly("c",
          [](leopart::advect::Tableau<dtype>& self)
          {
            return py::array_t<dtype, py::array::c_style>(
              self.c.size(), self.c.data(), py::cast(self));
          })
      .def_readonly("order", &leopart::advect::Tableau<dtype>::order);

  // Predefined tableaus
  auto tableaus_module = m.def_submodule("tableaus");
  tableaus_module.def_submodule("order1")
      .def("forward_euler", &leopart::advect::tableaus::order1::forward_euler<dtype>);
  tableaus_module.def_submodule("order2")
      .def("generic_alpha", &leopart::advect::tableaus::order2::generic_alpha<dtype>)
      .def("explicit_midpoint", &leopart::advect::tableaus::order2::explicit_midpoint<dtype>)
      .def("heun", &leopart::advect::tableaus::order2::heun<dtype>)
      .def("ralston", &leopart::advect::tableaus::order2::ralston<dtype>);
  tableaus_module.def_submodule("order3")
      .def("generic_alpha", &leopart::advect::tableaus::order3::generic_alpha<dtype>)
      .def("heun", &leopart::advect::tableaus::order3::heun<dtype>)
      .def("wray", &leopart::advect::tableaus::order3::wray<dtype>)
      .def("ralston", &leopart::advect::tableaus::order3::ralston<dtype>)
      .def("ssp", &leopart::advect::tableaus::order3::ssp<dtype>);
  tableaus_module.def_submodule("order4")
      .def("classic", &leopart::advect::tableaus::order4::classic<dtype>)
      .def("kutta1901", &leopart::advect::tableaus::order4::kutta1901<dtype>)
      .def("ralston", &leopart::advect::tableaus::order4::ralston<dtype>);

  // Utility functions
  m.def("evaluate_basis_functions",
        [](std::shared_ptr<dolfinx::fem::FunctionSpace<dtype>> V,
           std::vector<dtype>& x,
           std::vector<std::int32_t>& cells) {
          return leopart::utils::evaluate_basis_functions<dtype>(*V, x, cells);
        });
}