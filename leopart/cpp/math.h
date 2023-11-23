// Copyright: (c) 2020-2023 Chris Richardson, Jakob Maljaars and Nathan Sime
// This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <basix/math.h>
#include "utils.h"


namespace leopart::math
{
using leopart::utils::mdspan_t;

template <dolfinx::scalar T>
void transpose(mdspan_t<const T, 2> A, mdspan_t<T, 2> A_T)
{
  for (std::size_t i = 0; i < A_T.extent(0); ++i)
    for (std::size_t j = 0; j < A_T.extent(1); ++j)
      A_T(i, j) = A(j, i);
}

template <dolfinx::scalar T>
void matmult(mdspan_t<const T, 2> A, mdspan_t<const T, 2> B, mdspan_t<T, 2> C)
{
  for (std::size_t i = 0; i < A.extent(0); ++i)
    for (std::size_t j = 0; j < B.extent(1); ++j)
    {
      T sum{0};
      for (std::size_t k = 0; k < A.extent(1); ++k)
        sum += A(i, k) * B(k, j);
      C(i, j) = sum;
    }
}

} // namespace leopart::math