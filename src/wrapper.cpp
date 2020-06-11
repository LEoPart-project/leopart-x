// Copyright: (c) 2020 Chris Richardson and Jakob Maljaars
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Particles.h"
#include "generation.h"
#include "transfer.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace leopart;
PYBIND11_MODULE(pyleopart, m)
{
  py::class_<Field>(m, "Field")
      .def("data", py::overload_cast<int>(&Field::Field::data),
           py::return_value_policy::reference_internal)
      .def_property_readonly("value_shape", &Field::Field::value_shape)
      .def_property_readonly("value_size", &Field::Field::value_size)
      .def_readonly("name", &Field::Field::name);

  py::class_<Particles>(m, "Particles")
      .def(py::init<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>&,
                    const std::vector<int>&>())
      .def("add_field", &Particles::Particles::add_field)
      .def("add_particle", &Particles::Particles::add_particle)
      .def("delete_particle", &Particles::Particles::delete_particle)
      .def("data",
           [](Particles& self, int p, int f) { return self.field(f).data(p); })
      .def("field",
           py::overload_cast<std::string>(&Particles::Particles::field),
           py::return_value_policy::reference_internal)
      .def("cell_particles", &Particles::Particles::cell_particles);

  // Generation functions
  m.def("random_tet", &generation::random_reference_tetrahedron);
  m.def("random_tri", &generation::random_reference_triangle);
  m.def("mesh_fill", &generation::mesh_fill);

  // Transfer functions
  m.def("get_particle_contributions", &transfer::get_particle_contributions);
  m.def("transfer_to_particles", &transfer::transfer_to_particles);
  m.def("transfer_to_function", &transfer::transfer_to_function);
}