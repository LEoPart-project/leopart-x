# Copyright (c) 2020-2023 Chris Richardson, Jakob Maljaars and Nathan Sime
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

cmake_minimum_required(VERSION 3.19)

project(cpp VERSION "0.7.2")

find_package(DOLFINX REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
  ${PROJECT_NAME} MODULE
  leopart/cpp/wrapper.cpp
  leopart/cpp/Particles.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC dolfinx)
target_compile_options(
  ${PROJECT_NAME} PRIVATE
  # -Wno-comment
  # -Wall
  # -Wextra
  # -pedantic
  # -Werror
  -Wfatal-errors
  )

target_include_directories(
  ${PROJECT_NAME} PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>/leopart/cpp")

set(LEOPART_DIRS
    leopart/cpp/external)
foreach(DIR ${LEOPART_DIRS})
  add_subdirectory(${DIR})
endforeach()

# Set C++23 standard
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_23)

set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS ${PROJECT_NAME} DESTINATION leopart)