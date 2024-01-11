# Copyright (c) 2023 Jørgen Dokken and Nathan Sime
# This file is part of LEoPart-X, a particle-in-cell package for DOLFIN-X
# License: GNU Lesser GPL version 3 or any later version
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pathlib
import typing
import xml.etree.ElementTree as ET

from mpi4py import MPI

import adios2
import numpy as np
import numpy.typing

import dolfinx
import leopart.cpp


class XDMFParticlesFile:

    def __init__(self, comm: MPI.Intracomm, filename: str, mode: adios2.Mode):
        """
        Class for writing particle data to binary hdf5 data files with XDMF
        front end index.

        Args:
            comm: MPI communicator
            filename: Output file path
            mode: adios2 enum mode (e.g., read, write append)
        """
        self.comm = comm
        self.filename = pathlib.Path(filename)
        self.mode = mode

        self.adios = adios2.ADIOS(self.comm)
        self.io_name = f"XDMFParticlesFile_{hash(self)}"
        self.io = self.adios.DeclareIO(self.io_name)
        self.io.SetEngine("HDF5")

        self.xml_doc = None
        if mode is adios2.Mode.Write:
            xdmf = ET.Element("Xdmf")
            xdmf.attrib["Version"] = "3.0"
            xdmf.attrib["xmlns:xi"] = "http://www.w3.org/2001/XInclude"
            domain = ET.SubElement(xdmf, "Domain")
            temporal_grid = ET.SubElement(domain, "Grid")
            temporal_grid.attrib["Name"] = "CellTime"
            temporal_grid.attrib["GridType"] = "Collection"
            temporal_grid.attrib["CollectionType"] = "Temporal"
            self.xml_doc = xdmf
            self.domain = domain
            self.temporal_grid = temporal_grid
            self.step_count = 0

            self.outfile = self.io.Open(
                str(self.filename.with_suffix(".h5")), adios2.Mode.Write)
        elif mode is adios2.Mode.Append or mode is adios2.Mode.Read:
            raise NotImplementedError

    def __del__(self):
        if hasattr(self, "outfile"):
            self.outfile.Close()
        assert self.adios.RemoveIO(self.io_name)

    def _write_xml(self):
        ET.indent(self.xml_doc, space="\t", level=0)
        if self.comm.rank == 0:
            with self.filename.open("w") as outfile:
                outfile.write(
                    '<?xml version="1.0"?>\n'
                    '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
                outfile.write(ET.tostring(self.xml_doc, encoding="unicode"))

    def _append_xml_node(
            self, nglobal: int, gdim_pts: int, t: float, t_str: str,
            data_map: typing.Dict[str, np.typing.NDArray]):
        domain = self.temporal_grid
        grid = ET.SubElement(domain, "Grid")
        time = ET.SubElement(grid, "Time")
        time.attrib["Value"] = f"{t:.12e}"
        grid.attrib["GridType"] = "Uniform"
        grid.attrib["Name"] = "Point Cloud"
        topology = ET.SubElement(grid, "Topology")
        topology.attrib["NumberOfElements"] = str(nglobal)
        topology.attrib["TopologyType"] = "PolyVertex"
        topology.attrib["NodesPerElement"] = "1"
        geometry = ET.SubElement(grid, "Geometry")
        geometry.attrib["GeometryType"] = "XY" if gdim_pts == 2 else "XYZ"
        it0 = ET.SubElement(geometry, "DataItem")
        it0.attrib["Dimensions"] = f"{nglobal} {gdim_pts}"
        it0.attrib["Format"] = "HDF"
        it0.text = self.filename.stem + f".h5:/Step0/Points_{t_str}"

        for data_name, data in data_map.items():
            attrib = ET.SubElement(grid, "Attribute")
            attrib.attrib["Name"] = data_name
            attribute_type = "Scalar" if data.shape[1] == 1 else "Vector"
            attrib.attrib["AttributeType"] = attribute_type
            attrib.attrib["Center"] = "Node"
            it1 = ET.SubElement(attrib, "DataItem")
            it1.attrib["Dimensions"] = f"{nglobal} {data.shape[1]}"
            it1.attrib["Format"] = "HDF"
            it1.text = self.filename.stem + f".h5:/Step0/{data_name}_{t_str}"

    def write_particles(
            self, particles: leopart.cpp.Particles, t: float,
            field_names: typing.Optional[typing.Sequence[str]] = None):
        """
        Write the particle data to file along with the provided fields' data.

        Args:
            particles: Particles to write to file
            t: Time step
            field_names: Names of particles' field data to write
        """
        if field_names is None:
            field_names = []
        active_pidxs = particles.active_pidxs()
        data_map = dict(
            zip(field_names,
                (particles.field(field_name).data()[active_pidxs]
                 for field_name in field_names)))
        self.write_points(particles.x().data()[active_pidxs], data_map, t)

    def write_points(
            self, points: np.typing.NDArray,
            data_map: typing.Dict[str, np.typing.NDArray],
            t: float):
        """
        Write a point cloud with associated data to file.

        Args:
            points: Point cloud
            data_map: Map from name to data for each point in the point cloud
            t: Time step
        """
        nlocal = points.shape[0]
        im_data = dolfinx.common.IndexMap(MPI.COMM_WORLD, nlocal)
        nglobal = im_data.size_global
        local_range = im_data.local_range

        # Update xml data
        t_str = f"{t:.12e}".replace(".", "_").replace("-", "_")
        self._append_xml_node(nglobal, points.shape[1], t, t_str,
                              data_map)

        # ADIOS2 write binary data
        outfile = self.outfile
        io = self.io

        pointvar = io.DefineVariable(
            f"Points_{t_str}", points, shape=[nglobal, points.shape[1]],
            start=[local_range[0], 0], count=[nlocal, points.shape[1]])
        outfile.Put(pointvar, points)

        for data_name, data in data_map.items():
            assert data.shape[0] == nlocal
            valuevar = io.DefineVariable(
                f"{data_name}_{t_str}", data, shape=[nglobal, data.shape[1]],
                start=[local_range[0], 0], count=[nlocal, data.shape[1]])
            outfile.Put(valuevar, data)

        outfile.PerformPuts()

        self._write_xml()
