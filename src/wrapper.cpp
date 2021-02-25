// Copyright: (c) 2020 Chris Richardson and Jakob Maljaars
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Field.h"
#include "Particles.h"
#include "advect/advect.h"
#include "generation.h"
#include "project/l2project.h"
#include "transfer.h"

#include <dolfinx.h>
#include <memory>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using leopart::Field;
using leopart::Particles;
using leopart::project::L2Project;

using leopart::generation::mesh_fill;
using leopart::generation::random_reference_tetrahedron;
using leopart::generation::random_reference_triangle;

using leopart::transfer::get_particle_contributions;
using leopart::transfer::transfer_to_function;
using leopart::transfer::transfer_to_particles;

using leopart::advect::Advect;

PYBIND11_MODULE(pyleopart, m)
{
  py::class_<Field>(m, "Field")
      .def("data", py::overload_cast<int>(&Field::Field::data),
           py::return_value_policy::reference_internal)
      .def_property_readonly("value_shape", &Field::Field::value_shape)
      .def_property_readonly("value_size", &Field::Field::value_size)
      .def_readonly("name", &Field::Field::name);

  py::class_<Particles, std::shared_ptr<Particles>>(m, "Particles")
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
      .def("cell_particles",
           py::overload_cast<>(&Particles::Particles::cell_particles,
                               py::const_));

  // Projection classes
  py::class_<L2Project>(m, "L2Project")
      .def(py::init<Particles&,
                    std::shared_ptr<dolfinx::fem::Function<PetscScalar>>,
                    std::string>())
      .def("solve", py::overload_cast<>(&L2Project::L2Project::solve),
           "l2 projection")
      .def("solve",
           py::overload_cast<double, double>(&L2Project::L2Project::solve),
           "Bounded l2 projection");

  // Advection class(es)
  py::class_<Advect, std::shared_ptr<Advect>>(m, "Advect")
      //    .def(py::init<int>())
      // //    .def(py::init<
      // //         std::shared_ptr<const Particles>>())
      //    .def(py::init<
      //         const std::shared_ptr<dolfinx::mesh::Mesh>&>())
      .def(py::init<std::shared_ptr<Particles>&,
                    const std::shared_ptr<dolfinx::mesh::Mesh>&>());

  // Generation functions
  m.def("random_tet", &random_reference_tetrahedron);
  m.def("random_tri", &random_reference_triangle);
  m.def("mesh_fill", &mesh_fill);

  // Transfer functions (probably shouldn't be interfaced)
  m.def("get_particle_contributions", &get_particle_contributions);
  m.def("transfer_to_particles",
        py::overload_cast<
            Particles&, Field&,
            std::shared_ptr<const dolfinx::fem::Function<PetscScalar>>,
            const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>&>(
            &transfer_to_particles<PetscScalar>));
  m.def("transfer_to_function",
        py::overload_cast<std::shared_ptr<dolfinx::fem::Function<PetscScalar>>,
                          const Particles&, const Field&,
                          const Eigen::Array<double, Eigen::Dynamic,
                                             Eigen::Dynamic, Eigen::RowMajor>&>(
            &transfer_to_function<PetscScalar>));
}