
#include "generation.h"
#include <Eigen/Dense>

using namespace leopart;

Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>
generation::random_reference_triangle(int n)
{
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> p(n, 3);

  for (int i = 0; i < n; ++i)
  {
    Eigen::Vector3d x;
    x.head(2) = Eigen::Vector2d::Random();
    x[2] = 0.0;
    if (x[0] + x[1] > 0.0)
    {
      x[0] = (1 - x[0]);
      x[1] = (1 - x[1]);
    }
    else
    {
      x[0] = (1 + x[0]);
      x[1] = (1 + x[1]);
    }
    p.row(i) = x;
  }

  p /= 2.0;
  return p;
}
//------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>
generation::random_reference_tetrahedron(int n)
{
  Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> p(n, 3);
  p.fill(1.0);

  for (int i = 0; i < n; ++i)
  {
    Eigen::RowVector3d r = Eigen::Vector3d::Random();
    double& x = r[0];
    double& y = r[1];
    double& z = r[2];

    // Fold cube into tetrahedron
    if ((x + z) > 0)
    {
      x = -x;
      z = -z;
    }

    if ((y + z) > 0)
    {
      z = -z - x - 1;
      y = -y;
    }
    else if ((x + y + z) > -1)
    {
      x = -x - z - 1;
      y = -y - z - 1;
    }

    p.row(i) += r;
  }
  p /= 2.0;

  return p;
}
