#include "particles.h"
#include <dolfinx.h>

using namespace leopart;

particles::particles(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>& x,
                     const std::vector<int>& cells)
    : _field_name({"x"}), _field_shape({{(int)x.cols()}}),
      _field_data({std::vector<double>(x.rows() * x.cols())})
{
  // Find max cell index, and create cell->particle map
  auto max_cell_it = std::max_element(cells.begin(), cells.end());
  if (max_cell_it == cells.end())
    throw std::runtime_error("Error in cells data");
  const int max_cell = *max_cell_it;
  _cell_particles.resize(max_cell + 1);
  for (std::size_t p = 0; p < cells.size(); ++p)
    _cell_particles[cells[p]].push_back(p);

  // Copy position data to first field
  std::copy(x.data(), x.data() + x.rows() * x.cols(), _field_data[0].begin());
}
