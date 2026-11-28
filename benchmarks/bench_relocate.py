# Copyright (c) 2026 Nathan Sime
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Move every particle, then relocate them all (relocate_bbox).

Motions:
  small:   up to half a cell, as one Runge-Kutta stage typically moves
           particles. Most stay in their cell or move to a neighbour.
  reflect: through the centre of the domain, so most particles change
           rank. This exercises the global search and communication.

The particles carry --num-fields scalar fields, which relocation
communicates along with the positions."""

import common
import numpy as np

DEFAULT_PPC = 10


def small_step(x, gdim, h):
    # Up to h/2 in each component, vanishing on the boundary of the unit
    # square or cube, and monotone, so particles stay in the domain
    a = 0.5 * h
    y = x.copy()
    for i in range(gdim):
        j = (i + 1) % gdim
        y[:, i] += (a * 4.0 * x[:, i] * (1.0 - x[:, i])
                    * np.sin(2.0 * np.pi * x[:, j]))
    return y


def reflect(x, gdim, h):
    return 1.0 - x


MOTIONS = {"small": small_step, "reflect": reflect}


class Relocate(common.Benchmark):
    name = "relocate"

    def add_arguments(self, parser):
        parser.add_argument("--motion", choices=MOTIONS, default="small")
        parser.add_argument("--num-fields", type=int, default=2,
                            help="Scalar fields carried by each particle")

    def create_particles(self, mesh, args, state):
        ptcls = common.fill_mesh(
            mesh, common.default_np_per_cell(args, DEFAULT_PPC))
        for i in range(args.num_fields):
            ptcls.add_field(f"f{i}", [1])
            ptcls.field(f"f{i}").data()[:] = float(i)
        return ptcls

    def prepare(self, mesh, args, state):
        ptcls = state["ptcls"]
        x = ptcls.x().data()
        x[:] = MOTIONS[args.motion](x, mesh.geometry.dim, 1.0 / state["n"])

    def run(self, mesh, args, state):
        ptcls = state["ptcls"]
        ptcls.relocate_bbox(mesh._cpp_object)

    def config(self, args, state):
        return {"ppc": common.default_np_per_cell(args, DEFAULT_PPC),
                "motion": args.motion, "num_fields": args.num_fields}


if __name__ == "__main__":
    common.main(Relocate())
