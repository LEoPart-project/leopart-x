# Copyright (C) 2023 The DOLFINX authors
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified 2023 by Nathan Sime

import gc
import pathlib
import shutil
import time
from collections import defaultdict

from mpi4py import MPI

import pytest


def pytest_runtest_teardown(item):
    """Collect garbage after every test to force calling
    destructors which might be collective"""

    # Do the normal teardown
    item.teardown()

    # Collect the garbage (call destructors collectively)
    del item
    # NOTE: How are we sure that 'item' does not hold references to
    # temporaries and someone else does not hold a reference to 'item'?!
    # Well, it seems that it works...
    gc.collect()
    comm = MPI.COMM_WORLD
    comm.Barrier()


# Add 'skip_in_parallel' skip
def pytest_runtest_setup(item):
    marker = item.get_closest_marker("skip_in_parallel")
    if marker and MPI.COMM_WORLD.size > 1:
        pytest.skip("This test should only be run in serial")


def _worker_id(request):
    """Returns thread id when running with pytest-xdist in parallel."""
    try:
        return request.config.workerinput["workerid"]
    except AttributeError:
        return "master"


def _create_tempdir(request):
    # Get directory name of test_foo.py file
    testfile = request.module.__file__
    testfiledir = pathlib.Path(testfile).resolve().parent

    # Construct name test_foo_tempdir from name test_foo.py
    testfilename = pathlib.Path(testfile).name
    outputname = testfilename.replace(".py", f"_tempdir_{_worker_id(request)}")

    # Get function name test_something from test_foo.py
    function = request.function.__name__

    # Join all of these to make a unique path for this test function
    basepath = testfiledir / outputname
    path = basepath / function

    # Add a sequence number to avoid collisions when tests are otherwise
    # parameterized
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        _create_tempdir._sequencenumber[path] += 1
        sequencenumber = _create_tempdir._sequencenumber[path]
    else:
        sequencenumber = None

    sequencenumber = comm.bcast(sequencenumber)
    path = path.parent / (path.name + "__" + str(sequencenumber))

    # Delete and re-create directory on root node
    if comm.rank == 0:
        # First time visiting this basepath, delete the old and create a
        # new
        if basepath not in _create_tempdir._basepaths:
            _create_tempdir._basepaths.add(basepath)
            if basepath.exists():
                shutil.rmtree(basepath)
            # Make sure we have the base path test_foo_tempdir for this
            # test_foo.py file
            if not basepath.exists():
                basepath.mkdir()

        # Delete path from old test run
        if path.exists():
            shutil.rmtree(path)
        # Make sure we have the path for this test execution: e.g.
        # test_foo_tempdir/test_something__3
        if not path.exists():
            path.mkdir()

        # Wait until the above created the directory
        waited = 0
        while not path.exists():
            time.sleep(0.1)
            waited += 0.1
            if waited > 1:
                msg = f"Unable to create test directory {path}"
                raise RuntimeError(msg)

    comm.Barrier()

    return path


# Assigning a function member variables is a bit of a nasty hack
_create_tempdir._sequencenumber = defaultdict(int)
_create_tempdir._basepaths = set()


@pytest.fixture()
def tempdir(request):
    """Return a unique directory name for this test function instance.

    Deletes and re-creates directory from previous test runs but lets
    the directory stay after the test run for eventual inspection.

    Returns the directory name, derived from the test file and
    function plus a sequence number to work with parameterized tests.

    Does NOT change the current directory.

    MPI safe (assuming MPI.COMM_WORLD context).

    """
    return _create_tempdir(request)
