# LEoPart-x

![](https://github.com/LEoPart-X/LEoPart/workflows/C/C++%20CI/badge.svg)

LEoPart-x (**L**agrangian-**E**ulerian **o**n **Part**icles-x) integrates
particle functionality into the open-source finite element library
[DOLFINx](https://github.com/FEniCS/dolfinx), a component of
[the FEniCS project](www.fenicsproject.org).

LEoPart-x is developed and tested against
[DOLFINx `v0.7.0`](https://github.com/FEniCS/dolfinx/releases/tag/v0.7.0).

## Documentation
A doxygen page can be generated by navigating to the `doc/` directory and typing

```bash
doxygen
```

## Dependencies

- [Python3](https://www.python.org/)
- [Pybind11](https://github.com/pybind/pybind11)
- [DOLFINx](https://github.com/FEniCS/dolfinx)
- [Basix](https://github.com/FEniCS/basix)

## Installation

#### C++ library

```bash
mkdir src/build
cd src/build
cmake .. && make
```

#### C++ & Python interface via pip

```bash
cd src && pip3 install .
```

#### Docker

Inside a `DOLFINx` Docker container

```bash
docker run -ti dolfinx/dolfinx:v0.7.0
pip3 install pybind11[global]
git clone https://github.com/LEoPart-project/leopart-x.git
cd leopart-x/src && pip3 install .
```

## Citing

LEoPart has been described and developed in a series of papers:

```
@article{Maljaars2020,
    author = {Maljaars, Jakob M. and Richardson, Chris N. and Sime, Nathan},
    doi = {10.1016/j.camwa.2020.04.023},
    issn = {08981221},
    journal = {Computers and Mathematics with Applications},
    number = {xxxx},
    pages = {1--27},
    title = {{LEOPART: A particle library for FENICS}},
    year = {2020}
}

@article{Maljaars2019,
  author = {Maljaars, Jakob M. and Labeur, Robert Jan and Trask, Nathaniel and Sulsky, Deborah},
  doi = {10.1016/J.CMA.2019.01.028},
  issn = {0045-7825},
  journal = {Comput. Methods Appl. Mech. Eng.},
  month = {jan},
  pages = {443--465},
  publisher = {North-Holland},
  title = {{Conservative, high-order particle-mesh scheme with applications to advection-dominated flows}},
  volume = {348},
  year = {2019}
}
```
