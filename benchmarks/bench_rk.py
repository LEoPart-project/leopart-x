# Copyright (c) 2026 Nathan Sime
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""One explicit Runge-Kutta step (leopart.cpp.rk) in a steady cellular
flow, with a time step of half a cell. Each stage interpolates the
velocity and relocates every particle."""

import common

import leopart.cpp as pyleopart

DEFAULT_PPC = 10

TABLEAUS = {
    1: pyleopart.tableaus.order1.forward_euler,
    2: pyleopart.tableaus.order2.heun,
    3: pyleopart.tableaus.order3.ssp,
    4: pyleopart.tableaus.order4.classic,
}


class RungeKutta(common.Benchmark):
    name = "rk"

    def add_arguments(self, parser):
        parser.add_argument("--rk-order", type=int, choices=TABLEAUS,
                            default=4)

    def setup(self, mesh, args, state):
        state["u"] = common.velocity_function(mesh, args.degree)
        state["tableau"] = TABLEAUS[args.rk_order]()
        # The flow speed is at most 1
        state["dt"] = 0.5 / state["n"]
        state["t"] = 0.0

    def create_particles(self, mesh, args, state):
        ptcls = common.fill_mesh(
            mesh, common.default_np_per_cell(args, DEFAULT_PPC))
        state["tableau"].check_and_create_fields(ptcls)
        return ptcls

    def run(self, mesh, args, state):
        u = state["u"]._cpp_object
        pyleopart.rk(mesh._cpp_object, state["ptcls"], state["tableau"],
                     lambda t: u, state["t"], state["dt"])
        state["t"] += state["dt"]

    def config(self, args, state):
        return {"ppc": common.default_np_per_cell(args, DEFAULT_PPC),
                "rk_order": args.rk_order, "space": "CG",
                "value_shape": "gdim"}


if __name__ == "__main__":
    common.main(RungeKutta())
