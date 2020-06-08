
#include "generation.h"
#include "particles.h"

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
      .def("data", py::overload_cast<int, int>(&particles::particles::data))
      .def("cell_particles", &particles::particles::cell_particles);

  m.def("random_tet", &generation::random_reference_tetrahedron);
  m.def("random_tri", &generation::random_reference_triangle);
}