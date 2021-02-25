#include "advect.h"

#include <dolfinx.h>

namespace leopart
{
namespace advect
{

Advect::Advect(std::shared_ptr<Particles>& particles,
               const std::shared_ptr<dolfinx::mesh::Mesh>& mesh)
    : _particles(particles), _mesh(mesh)
{
  set_facet_info();
}

void Advect::set_facet_info()
{
  const dolfinx::mesh::Topology& topology = _mesh->topology();

  const dolfinx::mesh::CellType celltype
      = topology.cell_type(); // _mesh->topology().cell_type();
  const std::size_t tdim = topology.dim();
  // Number of facets per cell
  const int num_facets_per_cell
      = dolfinx::mesh::cell_num_entities(celltype, tdim - 1);
  std::cout << "Num facets per cell " << num_facets_per_cell << std::endl;

  // Information for each facet of the mesh
  // facets_info.resize(mesh->num_entities(tdim - 1));
  // _facet_info.resize(_mesh->topology_mutable().create_entities(tdim - 1));
  // std::cout << "Looking for number of facets
  // "<<_mesh->topology_mutable().create_entities(tdim - 1)<<std::endl;

  // Get number of facets owned by this process
  // Facet-to-cell connectivity
  _mesh->topology_mutable().create_connectivity(tdim - 1, tdim);
  // Cell-to-facet connectivity
  _mesh->topology_mutable().create_connectivity(tdim, tdim - 1);
  // Guess we need to make this a member?
  auto f_to_c = topology.connectivity(tdim - 1, tdim);
  auto c_to_f = topology.connectivity(tdim, tdim - 1);
  // assert(topology.index_map(tdim - 1));
  // std::set<std::int32_t> fwd_shared_facets;

  // TODO: might be incomplete in case of ghost facets?
  const int num_facets = topology.index_map(tdim - 1)->size_local();
  _facet_info.resize(num_facets);
  std::cout << "Num facets local " << num_facets << std::endl;

  std::cout << f_to_c->str() << std::endl;
  std::cout << c_to_f->str() << std::endl;

  // Now Eigen, should be dolfinx::common::array2d
  const auto midpoints
      = dolfinx::mesh::midpoints(*_mesh, tdim - 1, f_to_c->links(0));
  std::cout << "Midpoints " << midpoints.size() << std::endl;

  for (int f = 0; f < num_facets; ++f)
  {
    // std::cout << "Connections "<<f<<" "<<f_to_c->links(f)<<std::endl;
    // if (f_to_c->num_links(f) == 1
    //     and fwd_shared_facets.find(f) == fwd_shared_facets.end())
    // {
    //   active_entities.push_back(f);
    // }
  }

  // std::int32_t create_entities(int dim);

  // Cache midpoint, and normal of each facet in mesh
  // Note that in DOLFIN simplicial cells, Facet f_i is opposite Vertex v_i,
  // etc.

  //   const Mesh* mesh = _P->mesh();

  //   const std::size_t num_cell_facets = mesh->type().num_entities(tdim - 1);

  // Information for each facet of the mesh
  // facets_info.resize(mesh->num_entities(tdim - 1));

  // for (FacetIterator fi(*mesh); !fi.end(); ++fi)
  // {
  //   // Get and store facet normal and facet midpoint
  //   Point facet_n = fi->normal();
  //   Point facet_mp = fi->midpoint();
  //   std::vector<bool> outward_normal;

  //   // FIXME: could just look at first cell only, simplifies code

  //   int i = 0;
  //   for (CellIterator ci(*fi); !ci.end(); ++ci)
  //   {
  //     const unsigned int* cell_facets = ci->entities(tdim - 1);

  //     // Find which facet this is in the cell
  //     const std::size_t local_index
  //         = std::find(cell_facets, cell_facets + num_cell_facets,
  //         fi->index())
  //           - cell_facets;
  //     assert(local_index < num_cell_facets);

  //     // Get cell vertex opposite facet
  //     Vertex v(*mesh, ci->entities(0)[local_index]);

  //     // Take vector from facet midpoint to opposite vertex
  //     // and compare to facet normal.
  //     const Point q = v.point() - facet_mp;
  //     const double dir = q.dot(facet_n);
  //     assert(std::abs(dir) > 1e-10);
  //     bool outward_pointing = (dir < 0);

  //     // Make sure that the facet normal is always outward pointing
  //     // from Cell 0.
  //     if (!outward_pointing and i == 0)
  //     {
  //       facet_n *= -1.0;
  //       outward_pointing = true;
  //     }

  //     // Store outward normal bool for safety check (below)
  //     outward_normal.push_back(outward_pointing);
  //     ++i;
  //   }

  //   // Safety check
  //   if (fi->num_entities(tdim) == 2)
  //   {
  //     if (outward_normal[0] == outward_normal[1])
  //     {
  //       dolfin_error(
  //           "advect_particles.cpp::update_facets_info",
  //           "get correct facet normal direction",
  //           "The normal cannot be of same direction for neighboring cells");
  //     }
  //   }

  //   // Store info in facets_info array
  //   const std::size_t index = fi->index();
  //   facets_info[index].midpoint = facet_mp;
  //   facets_info[index].normal = facet_n;
  // } // End facet iterator
}

} // namespace advect
} // namespace leopart