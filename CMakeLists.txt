# Copyright (c) 2020-2023 Chris Richardson, Jakob Maljaars and Nathan Sime
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

cmake_minimum_required(VERSION 3.21)

project(leopart VERSION "0.11.0" LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LEOPART_WERROR "Treat compiler warnings as errors" OFF)

if(WIN32)
  # Windows requires all symbols to be manually exported.
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Locate Basix via its Python package (required by DOLFINx's CMake config)
execute_process(
  COMMAND ${Python_EXECUTABLE} -c
    "import os, sys, basix; sys.stdout.write(os.path.dirname(basix.__file__))"
  OUTPUT_VARIABLE BASIX_PY_DIR
  RESULT_VARIABLE BASIX_PY_COMMAND_RESULT
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(Basix REQUIRED CONFIG HINTS ${BASIX_PY_DIR})
find_package(DOLFINX 0.11 REQUIRED CONFIG)
message(STATUS "Found DOLFINx at ${DOLFINX_DIR} (version ${DOLFINX_VERSION})")

# Detect the installed nanobind package and import it into CMake.
# nanobind must match the version DOLFINx's Python interface was built
# with, so that DOLFINx objects can be passed to LEoPart-x.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
  cpp
  NB_SUPPRESS_WARNINGS
  NOMINSIZE
  leopart/cpp/wrapper.cpp
  leopart/cpp/Particles.cpp
  leopart/cpp/external/quadprog_mdspan/QuadProg++.cc)

if(DOLFINX_VERSION VERSION_GREATER_EQUAL "0.12.0.0")
  target_link_libraries(cpp PRIVATE dolfinx::dolfinx)
else()
  target_link_libraries(cpp PRIVATE dolfinx)
endif()

# Note: leopart/cpp is deliberately *not* added as an include directory.
# Its headers are included with quotes relative to each other, and adding it
# would let leopart/cpp/math.h shadow the system <math.h>.

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(cpp PRIVATE -Wall -Wextra -pedantic -Wno-comment)
  if(LEOPART_WERROR)
    target_compile_options(cpp PRIVATE -Werror -Wfatal-errors)
  endif()
endif()

set_target_properties(cpp PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS cpp LIBRARY DESTINATION leopart)
