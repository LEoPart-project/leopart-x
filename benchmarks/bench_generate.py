# Copyright (c) 2026 Nathan Sime
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Fill the mesh with particles: mesh_fill plus constructing Particles.

The kernel creates the particles, so its peak memory includes their
storage."""

import common

DEFAULT_PPC = 10


class Generate(common.Benchmark):
    name = "generate"

    def prepare(self, mesh, args, state):
        # Free the previous repeat's particles before measuring
        state["ptcls"] = None

    def run(self, mesh, args, state):
        state["ptcls"] = common.fill_mesh(
            mesh, common.default_np_per_cell(args, DEFAULT_PPC))

    def config(self, args, state):
        return {"ppc": common.default_np_per_cell(args, DEFAULT_PPC),
                "kernel": "mesh_fill + Particles"}


if __name__ == "__main__":
    common.main(Generate())
