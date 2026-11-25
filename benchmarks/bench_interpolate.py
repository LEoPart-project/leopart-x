# Copyright (c) 2026 Nathan Sime
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Interpolate a vector CG function at the particles
(transfer_to_particles), as each Runge-Kutta stage does."""

import common

import leopart.cpp as pyleopart

DEFAULT_PPC = 10


class Interpolate(common.Benchmark):
    name = "interpolate"

    def setup(self, mesh, args, state):
        state["u"] = common.velocity_function(mesh, args.degree)

    def create_particles(self, mesh, args, state):
        ptcls = common.fill_mesh(
            mesh, common.default_np_per_cell(args, DEFAULT_PPC))
        ptcls.add_field("u", [mesh.geometry.dim])
        return ptcls

    def run(self, mesh, args, state):
        ptcls = state["ptcls"]
        pyleopart.transfer_to_particles(
            ptcls, ptcls.field("u"), state["u"]._cpp_object)

    def config(self, args, state):
        return {"ppc": common.default_np_per_cell(args, DEFAULT_PPC),
                "space": "CG", "value_shape": "gdim"}


if __name__ == "__main__":
    common.main(Interpolate())
