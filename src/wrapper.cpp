
#include "generation.h"
#include "particles.h"
#include "transfer.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace leopart;
PYBIND11_MODULE(pyleopart, m)
{
  py::class_<particles>(m, "particles")
      .def(py::init<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>&,
                    const std::vector<int>&>())
      .def("add_field", &particles::particles::add_field)
      .def("data", py::overload_cast<int, int>(&particles::particles::data))
      .def("cell_particles", &particles::particles::cell_particles);

  // Generation functions
  m.def("random_tet", &generation::random_reference_tetrahedron);
  m.def("random_tri", &generation::random_reference_triangle);

  // Transfer functions
  m.def("get_particle_contributions", &transfer::get_particle_contributions);
  m.def("transfer_to_particles", &transfer::transfer_to_particles);
  m.def("transfer_to_function", &transfer::transfer_to_function);
}