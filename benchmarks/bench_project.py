# Copyright (c) 2026 Nathan Sime
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Project scalar particle data onto a DG function by local least
squares (transfer_to_function)."""

import common
import numpy as np

import dolfinx
import leopart.cpp as pyleopart


class Project(common.Benchmark):
    name = "project"

    def setup(self, mesh, args, state):
        Q = dolfinx.fem.functionspace(mesh, ("DG", args.degree))
        num_dofs = Q.dofmap.dof_layout.num_dofs
        ppc = common.default_np_per_cell(args, 2 * num_dofs)
        if ppc < num_dofs:
            msg = (f"Projection onto DG{args.degree} needs at least "
                   f"{num_dofs} particles per cell, got {ppc}")
            raise ValueError(msg)
        state["ppc"] = ppc
        state["uh"] = dolfinx.fem.Function(Q)

    def create_particles(self, mesh, args, state):
        ptcls = common.fill_mesh(mesh, state["ppc"])
        ptcls.add_field("v", [1])
        x = ptcls.x().data()
        ptcls.field("v").data()[:, 0] = (
            np.sin(2.0 * np.pi * x[:, 0]) * np.cos(2.0 * np.pi * x[:, 1]))
        return ptcls

    def run(self, mesh, args, state):
        ptcls = state["ptcls"]
        pyleopart.transfer_to_function(
            state["uh"]._cpp_object, ptcls, ptcls.field("v"))

    def config(self, args, state):
        return {"ppc": state["ppc"], "space": "DG", "value_shape": "scalar"}


if __name__ == "__main__":
    common.main(Project())
