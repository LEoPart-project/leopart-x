// Copyright: (c) 2020-2023 Jakob Maljaars, Chris Richardson and Nathan Sime
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <iostream>
#include <ranges>
#include <dolfinx.h>

#include "Particles.h"
#include "transfer.h"
#include "utils.h"


namespace leopart
{
namespace advect
{

using leopart::utils::mdspan_t;

template <std::floating_point T>
class Tableau
{
public:
  /// Butcher Tableau for use with time integration by
  /// (explicit) Runge-Kutta (RK) methods.
  ///
  /// The tableau for an RK method of order s is defined in the format
  ///
  /// c | a
  /// ------
  ///   | b
  ///
  /// where a is an s x s matrix of coefficients and b and c are arrays 
  /// of coefficients of length s.
  ///
  /// @tparam T The unknown scalar type.
  /// @param order Method order
  /// @param a Coefficient matrix, unrolled row major
  /// @param b Coefficient vector
  /// @param c Coefficient vector
  Tableau(
    const std::size_t order,
    const std::vector<T> a,
    const std::vector<T> b,
    const std::vector<T> c)
  : order(order), a(a), b(b), c(c)
  {
    const auto arg_shape_check = [&order](
      const std::size_t i, const std::string iname,
       const std::size_t expected) {
        if (i != expected)
          throw std::runtime_error(
            iname + " size " + std::to_string(i)
            + " does not match expected size " + std::to_string(expected)
            + " determined by order " + std::to_string(order));
      };
    arg_shape_check(a.size(), "a", order * order);
    arg_shape_check(b.size(), "b", order);
    arg_shape_check(c.size(), "c", order);
  };

  // Copy constructor
  Tableau(const Tableau& ptcls) = delete;

  /// Move constructor
  Tableau(Tableau&& ptcls) = default;

  /// Destructor
  ~Tableau() = default;

  // Field name generator for each RK substep index
  std::string field_name_substep(const std::size_t substep_n) const
  {
    return std::string("k") + std::to_string(substep_n);
  };

  constexpr std::string field_name_xn() const { return "xn"; }

  void check_and_create_fields(Particles<T> particles) const
  {
    const std::size_t gdim = particles.x().value_size();

    if (!particles.field_exists(field_name_xn()))
      particles.add_field(field_name_xn(), {gdim});
    
    for (std::size_t i = 0; i < order; ++i)
      if (!particles.field_exists(field_name_substep(i)))
        particles.add_field(field_name_substep(i), {gdim});
  }

  constexpr mdspan_t<const T, 2> a_md() const
  {
    return mdspan_t<const T, 2>(a.data(), order, order);
  }

  const std::size_t order;
  const std::vector<T> a;
  const std::vector<T> b;
  const std::vector<T> c;
};

namespace tableaus
{

namespace order1
{
template <std::floating_point T>
Tableau<T> forward_euler() { return Tableau<T>(1, {0.0}, {1.0}, {0.0}); }
} // end namespace order1

namespace order2
{
template <std::floating_point T>
Tableau<T> generic_alpha(const T alpha)
{
  return Tableau<T>(2,
    {0.0,   0.0,
     alpha, 0.0},
    {1.0 - 1.0 / (2.0 * alpha), 1.0 / (2.0 * alpha)},
    {0.0, alpha});
}

template <std::floating_point T>
Tableau<T> explicit_midpoint() { return generic_alpha<T>(0.5); }

template <std::floating_point T>
Tableau<T> heun() { return generic_alpha<T>(1.0); }

template <std::floating_point T>
Tableau<T> ralston() { return generic_alpha<T>(2.0 / 3.0); }
} // end namespace order2

namespace order3
{
template <std::floating_point T>
Tableau<T> generic_alpha(const T alpha)
{
  return Tableau<T>(3,
    {0.0,   0.0, 0.0,
     alpha, 0.0, 0.0,
     1.0 + (1.0 - alpha) / (alpha * (3.0 * alpha - 2.0)), - (1.0 - alpha) / (alpha * (3.0 * alpha - 2.0)), 0.0},
    {0.5 - 1.0 / (6.0 * alpha),
     1.0 / (6.0 * alpha * (1.0 - alpha)),
     (2.0 - 3.0 * alpha) / (6.0 * (1.0 - alpha))},
    {0.0, alpha, 1.0});
}

template <std::floating_point T>
Tableau<T> heun()
{
  return Tableau<T>(3,
    {0.0,       0.0,       0.0,
     1.0 / 3.0, 0.0,       0.0,
     0.0,       2.0 / 3.0, 0.0},
    {0.25,      0.0,       0.75},
    {0.0,       1.0 / 3.0, 2.0 / 3.0});
}

template <std::floating_point T>
Tableau<T> wray()
{
  return Tableau<T>(3,
    {0.0,        0.0,        0.0,
     8.0 / 15.0, 0.0,        0.0,
     0.25,       5.0 / 12.0, 0.0},
    {0.25,       0.0,        0.75},
    {0.0,        8.0 / 15.0, 2.0 / 3.0});
}

template <std::floating_point T>
Tableau<T> ralston()
{
  return Tableau<T>(3,
    {0.0, 0.0,  0.0,
     0.5, 0.0,  0.0,
     0.0, 0.75, 0.0},
    {2.0/9.0, 1.0/3.0, 4.0/9.0},
    {0.0, 0.5, 0.75});
}

template <std::floating_point T>
Tableau<T> ssp()
{
  return Tableau<T>(3,
    {0.0,  0.0,  0.0,
     1.0,  0.0,  0.0,
     0.25, 0.25, 0.0},
    {1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0},
    {0.0, 1.0, 0.5});
}
} // end namespace order3

namespace order4
{
template <std::floating_point T>
Tableau<T> classic()
{
  return Tableau<T>(4,
    {0.0, 0.0, 0.0, 0.0,
     0.5, 0.0, 0.0, 0.0,
     0.0, 0.5, 0.0, 0.0,
     0.0, 0.0, 1.0, 0.0},
    {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0},
    {0.0, 0.5, 0.5, 1.0});
}

template <std::floating_point T>
Tableau<T> kutta1901()
{
  return Tableau<T>(4,
    {0.0,         0.0,       0.0,       0.0,
     1.0 / 3.0,   0.0,       0.0,       0.0,
     -1.0 / 3.0,  1.0,       0.0,       0.0,
     1.0,        -1.0,       1.0,       0.0},
    {1.0 / 8.0,   3.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0},
    {0.0,         1.0 / 3.0, 2.0 / 3.0, 1.0});
}

template <std::floating_point T>
Tableau<T> ralston()
{
  return Tableau<T>(4,
    {0.0,         0.0,        0.0,        0.0,
     0.4,         0.0,        0.0,        0.0,
     0.29697761,  0.15875964, 0.0,        0.0,
     0.21810040, -3.05096516, 3.83286476, 0.0},
    {0.17476028, -0.55148066, 1.20553560, 0.17118478},
    {0.0,         0.4,        0.45573725, 1.0});
}
} // end namespace order4

} // end namespace tableaus


// Explicit Runge-Kutta method
template <std::floating_point T>
void rk(
  const dolfinx::mesh::Mesh<T>& mesh, // TODO: The mesh could vary with time
  leopart::Particles<T>& ptcls,
  const Tableau<T>& tableau,
  std::function<std::shared_ptr<dolfinx::fem::Function<T>>(T)> velocity_callback,
  const T t, const T dt)
{
  dolfinx::common::Timer timer("leopart::advect::rk");

  const std::string xn_name = tableau.field_name_xn();
  const int num_steps = tableau.order;
  mdspan_t<const T, 2> a = tableau.a_md();
  const std::vector<T>& b = tableau.b;
  const std::vector<T>& c = tableau.c;

  // Store initial position
  std::copy(ptcls.x().data().begin(), ptcls.x().data().end(),
            ptcls.field(xn_name).data().begin());

  for (std::size_t s = 0; s < num_steps; ++s)
  {
    // Compute k_s = u(t_n + c_s h, x_n + sum_{i=1}^{s-1} a_{si} k_i h)
    if (s != 0)
    {
      std::span<const T> xn = ptcls.field(xn_name).data();
      std::vector<T> suffix(xn.size(), 0.0);
      for (std::size_t i = 0; i < s; ++i)
      {
        std::span<const T> ks_data = ptcls.field(
          tableau.field_name_substep(i)).data();
        const T a_si = a(s, i);
        for (std::size_t j = 0; j < ks_data.size(); ++j)
          suffix[j] += a_si * ks_data[j] * dt;
      }
      
      std::span<T> xp = ptcls.x().data();
      for (std::size_t j = 0; j < xp.size(); ++j)
        xp[j] = xn[j] + suffix[j];
      
      ptcls.relocate_bbox(mesh, ptcls.active_pidxs());
    }

    std::shared_ptr<dolfinx::fem::Function<T>> uh_t = velocity_callback(t + c[s]);
    leopart::Field<T>& substep_field = ptcls.field(tableau.field_name_substep(s));
    leopart::transfer::transfer_to_particles<T>(ptcls, substep_field, uh_t);
  }

  std::span<const T> xn = ptcls.field(xn_name).data();
  std::vector<T> suffix(xn.size(), 0.0);
  for (std::size_t s = 0; s < num_steps; ++s)
  {
    std::span<const T> ks_data = ptcls.field(tableau.field_name_substep(s)).data();
    const T b_s = b[s];
    for (std::size_t i = 0; i < suffix.size(); ++i)
      suffix[i] += b_s * ks_data[i];
  }

  std::span<T> xp = ptcls.x().data();
  for (std::size_t i = 0; i < suffix.size(); ++i)
    xp[i] = xn[i] + dt * suffix[i];
  ptcls.relocate_bbox(mesh, ptcls.active_pidxs());
};

} // namespace advect
} // namespace leopart