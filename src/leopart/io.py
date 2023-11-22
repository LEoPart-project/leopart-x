from mpi4py import MPI
import pathlib
import numpy as np
import numpy.typing
import adios2
import xml.etree.ElementTree as ET


def compute_local_range(comm: MPI.Comm, N: int):
    rank = comm.rank
    size = comm.size
    n = N // size
    r = N % size
    # First r processes has one extra value
    if rank < r:
        return [rank * (n + 1), (rank + 1) * (n + 1)]
    else:
        return [rank * n + r, (rank + 1) * n + r]


class XDMFParticlesFile:

    def __init__(self, comm: MPI.Intracomm, filename: str, mode: adios2.Mode):
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
        elif mode is adios2.Mode.Append:
            raise NotImplementedError
        elif mode is adios2.Mode.Read:
            raise NotImplementedError

    def __del__(self):
        if hasattr(self, "outfile"):
            self.outfile.Close()
        assert self.adios.RemoveIO(self.io_name)

    def _write_xml(self):
        ET.indent(self.xml_doc, space="\t", level=0)
        if self.comm.rank == 0:
            with open(self.filename, "w") as outfile:
                outfile.write(
                    '<?xml version="1.0"?>\n<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
                outfile.write(ET.tostring(self.xml_doc, encoding="unicode"))

    def _append_xml_node(
            self, nglobal: int, gdim_pts: int, t: float, t_str: str):
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
        attrib = ET.SubElement(grid, "Attribute")
        attrib.attrib["Name"] = "Values"
        attrib.attrib["AttributeType"] = "Scalar"
        attrib.attrib["Center"] = "Node"
        it1 = ET.SubElement(attrib, "DataItem")
        it1.attrib["Dimensions"] = f"{nglobal} 1"
        it1.attrib["Format"] = "HDF"
        it1.text = self.filename.stem + f".h5:/Step0/Values_{t_str}"

    def write(self, points: np.typing.NDArray, t: float):
        nlocal = points.shape[0]
        nglobal = self.comm.allreduce(nlocal, op=MPI.SUM)
        local_range = compute_local_range(self.comm, nglobal)

        # Update xml data
        t_str = f"{t:.12e}".replace(".", "_").replace("-", "_")
        self._append_xml_node(nglobal, points.shape[1], t, t_str)

        # ADIOS2 write binary data
        outfile = self.outfile
        io = self.io

        pointvar = io.DefineVariable(
            f"Points_{t_str}", points, shape=[nglobal, points.shape[1]],
            start=[local_range[0], 0], count=[nlocal, points.shape[1]])
        outfile.Put(pointvar, points)

        values = np.arange(nlocal) + 5
        valuevar = io.DefineVariable(
            f"Values_{t_str}", values, shape=[nglobal, 1],
            start=[local_range[0], 0], count=[nlocal, 1])
        outfile.Put(valuevar, values)

        outfile.PerformPuts()

        self._write_xml()
