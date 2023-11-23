# Copyright: (c) 2020 Chris Richardson and Jakob Maljaars
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

cmake_minimum_required(VERSION 3.19)

set(PROJECT_NAME leopart)
project(${PROJECT_NAME} VERSION "0.7.0")

set(PY_PROJECT_NAME cpp)

# Get DOLFINX configuration data (DOLFINXConfig.cmake must be in
# DOLFINX_CMAKE_CONFIG_PATH)
if (NOT TARGET dolfinx)
  find_package(DOLFINX REQUIRED)
endif()

# Source directories
set(LEOPART_DIRS
    leopart/cpp/external)

# Executable
add_library(
  ${PROJECT_NAME} SHARED
  leopart/cpp/Particles.cpp)

# Add source to leopart target, and get sets of header files
foreach(DIR ${LEOPART_DIRS})
  add_subdirectory(${DIR})
endforeach()

target_include_directories(
  ${PROJECT_NAME} PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>/leopart/cpp")

# Target libraries and compiler options
target_link_libraries(${PROJECT_NAME} dolfinx)
target_compile_options(
  ${PROJECT_NAME} PRIVATE
  -Wno-comment
  -Wall
  -Wextra
  -pedantic
  -Werror
  -Wfatal-errors)

# Headers are not being installed
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(${PY_PROJECT_NAME} MODULE leopart/cpp/wrapper.cpp)

# Add DOLFINx libraries.
target_link_libraries(${PY_PROJECT_NAME} PUBLIC dolfinx ${PROJECT_NAME})

# Set C++20 standard
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_23)
target_compile_features(${PY_PROJECT_NAME} PRIVATE cxx_std_23)