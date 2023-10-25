// Copyright: (c) 2020-2023 Chris Richardson, Jakob Maljaars and Nathan Sime
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Field.h"
#include "Particles.h"
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
            return py::array_t<dtype>(array.size(), array.data(), py::cast(self));
          })
      .def("data",
          [](Field<dtype>& self)
          {
            const std::size_t value_size = self.value_size();
            std::span<dtype> array = self.data();
            const std::array<std::size_t, 2> data_shape{array.size() / value_size, value_size};
            return py::array_t<dtype>(data_shape, array.data(), py::cast(self));
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
                if ((px.shape()[1] != 2) and (px.shape()[1] != 3))
                  throw std::invalid_argument(
                    "Particle position value size expected to be 2 or 3");
                std::vector<dtype> p_data = std::vector<dtype>(
                  px.data(), px.data() + px.size());
                return Particles(p_data, p_cells, px.shape()[1]);
               }))
      .def("add_field", &Particles<dtype>::Particles::add_field)
      .def("add_particle",
          [](Particles<dtype>& self, std::vector<dtype>& px, std::int32_t cell) {
            self.add_particle(px, cell);
          })
      .def("delete_particle", &Particles<dtype>::Particles::delete_particle)
      .def("field",
           py::overload_cast<std::string>(&Particles<dtype>::Particles::field),
           py::return_value_policy::reference_internal)
      .def("cell_to_particle",
           py::overload_cast<>(&Particles<dtype>::Particles::cell_to_particle,
                               py::const_))
      .def("particle_to_cell",
           py::overload_cast<>(&Particles<dtype>::Particles::particle_to_cell,
                               py::const_));;

  // Generation functions
  m.def("random_reference_tetrahedron",
        [](const std::size_t n) {
          const int gdim = 3;
          std::vector<dtype> array = leopart::generation::random_reference_tetrahedron<dtype>(n);
          std::array<std::size_t, 2> shape = {array.size() / gdim, gdim};
          return py::array_t<dtype>(shape, array.data());
        },
        py::return_value_policy::move);
  m.def("random_reference_triangle",
        [](const std::size_t n) {
          const int gdim = 2;
          std::vector<dtype> array = leopart::generation::random_reference_triangle<dtype>(n);
          std::array<std::size_t, 2> shape = {array.size() / gdim, gdim};
          return py::array_t<dtype>(shape, array.data());
        },
        py::return_value_policy::move);
  m.def("mesh_fill",
        [](std::shared_ptr<dolfinx::mesh::Mesh<dtype_geom>> mesh, dtype density) {
          auto [xp_all, np_cells] = leopart::generation::mesh_fill(*mesh, density);
          const std::size_t gdim = mesh->geometry().dim();
          std::array<std::size_t, 2> shape = {xp_all.size() / gdim, gdim};
          auto ret_val = std::make_tuple<py::array_t<dtype>, std::vector<std::int32_t>>(
            py::array_t<dtype>(shape, xp_all.data()), std::move(np_cells));
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

  // Utility functions
  m.def("evaluate_basis_functions",
        [](std::shared_ptr<dolfinx::fem::FunctionSpace<dtype>> V,
           std::vector<dtype>& x,
           std::vector<std::int32_t>& cells) {
          return leopart::utils::evaluate_basis_functions<dtype>(*V, x, cells);
        });
}