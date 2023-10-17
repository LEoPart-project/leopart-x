// Copyright: (c) 2020 Chris Richardson and Jakob Maljaars
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Field.h"
#include "Particles.h"
#include "generation.h"
#include "project/l2project.h"
#include "transfer.h"

#include <cstddef>
#include <concepts>
#include <vector>
#include <tuple>
#include <dolfinx.h>
#include <memory>
// #include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using leopart::Field;
using leopart::Particles;
// using leopart::project::L2Project;

// using leopart::generation::mesh_fill;
// using leopart::generation::random_reference_tetrahedron;
// using leopart::generation::random_reference_triangle;

// using leopart::transfer::get_particle_contributions;
// using leopart::transfer::transfer_to_function;
// using leopart::transfer::transfer_to_particles;

PYBIND11_MODULE(pyleopart, m)
{
  py::class_<Field<double>>(m, "Field")
      .def("data",
          [](Field<double>& self, int p)
          {
            std::span<double> array = self.data(p);
            return py::array_t<double>(array.size(), array.data(), py::cast(self));
          })
      .def("data",
          [](Field<double>& self)
          {
            const std::size_t value_size = self.value_size();
            std::span<double> array = self.data();
            const std::array<std::size_t, 2> data_shape{array.size() / value_size, value_size};
            return py::array_t<double>(data_shape, array.data(), py::cast(self));
          })
      .def_property_readonly("value_shape", &Field<double>::Field::value_shape)
      .def_property_readonly("value_size", &Field<double>::Field::value_size)
      .def_readonly("name", &Field<double>::Field::name)
      .def("resize", &Field<double>::resize);

  py::class_<Particles<double>>(m, "Particles")
      .def(py::init<std::vector<double>&, const std::vector<std::int32_t>&,
           const std::size_t>())
      .def(py::init(
            [](const py::array_t<double, py::array::c_style>& px,
               const std::vector<std::int32_t>& p_cells) {
                if ((px.shape()[1] != 2) and (px.shape()[1] != 3))
                  throw std::invalid_argument(
                    "Particle position value size expected to be 2 or 3");
                std::vector<double> p_data = std::vector<double>(
                  px.data(), px.data() + px.size());
                return Particles(p_data, p_cells, px.shape()[1]);
               }))
      .def("add_field", &Particles<double>::Particles::add_field)
      .def("add_particle",
          [](Particles<double>& self, std::vector<double>& px, std::int32_t cell) {
            self.add_particle(px, cell);
          })
      .def("delete_particle", &Particles<double>::Particles::delete_particle)
      .def("field",
           py::overload_cast<std::string>(&Particles<double>::Particles::field),
           py::return_value_policy::reference_internal)
      .def("cell_particles",
           py::overload_cast<>(&Particles<double>::Particles::cell_particles,
                               py::const_));

  // // Projection classes
  // py::class_<L2Project>(m, "L2Project")
  //     .def(py::init<Particles&,
  //                   std::shared_ptr<dolfinx::function::Function<PetscScalar>>,
  //                   std::string>())
  //     .def("solve", py::overload_cast<>(&L2Project::L2Project::solve),
  //          "l2 projection")
  //     .def("solve",
  //          py::overload_cast<double, double>(&L2Project::L2Project::solve),
  //          "Bounded l2 projection");

  // Generation functions
  m.def("random_reference_tetrahedron",
        [](const std::size_t n) {
          const int gdim = 3;
          std::vector<double> array = leopart::generation::random_reference_tetrahedron<double>(n);
          std::array<std::size_t, 2> shape = {array.size() / gdim, gdim};
          return py::array_t<double>(shape, array.data());
        },
        py::return_value_policy::move);
  m.def("random_reference_triangle",
        [](const std::size_t n) {
          const int gdim = 2;
          std::vector<double> array = leopart::generation::random_reference_triangle<double>(n);
          std::array<std::size_t, 2> shape = {array.size() / gdim, gdim};
          return py::array_t<double>(shape, array.data());
        },
        py::return_value_policy::move);
  m.def("mesh_fill",
        [](std::shared_ptr<dolfinx::mesh::Mesh<double>> mesh, double density) {
          auto [xp_all, np_cells] = leopart::generation::mesh_fill(*mesh, density);
          const std::size_t gdim = mesh->geometry().dim();
          std::array<std::size_t, 2> shape = {xp_all.size() / gdim, gdim};
          auto ret_val = std::make_tuple<py::array_t<double>, std::vector<std::int32_t>>(
            py::array_t<double>(shape, xp_all.data()), std::move(np_cells));
          return ret_val;
        }, py::return_value_policy::move);

  // // Transfer functions (probably shouldn't be interfaced)
  // m.def("get_particle_contributions", &get_particle_contributions);
  // m.def("transfer_to_particles",
  //       py::overload_cast<
  //           Particles&, Field&,
  //           std::shared_ptr<const dolfinx::function::Function<PetscScalar>>,
  //           const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
  //                              Eigen::RowMajor>&>(
  //           &transfer_to_particles<PetscScalar>));
  // m.def("transfer_to_function",
  //       py::overload_cast<
  //           std::shared_ptr<dolfinx::function::Function<PetscScalar>>,
  //           const Particles&, const Field&,
  //           const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
  //                              Eigen::RowMajor>&>(
  //           &transfer_to_function<PetscScalar>));
}