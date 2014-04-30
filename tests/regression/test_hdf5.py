import numpy as np
import pytest
import os

from firedrake import *
from tests.common import *

try:
    import h5py
except ImportError:
    h5py = None

_h5file = os.path.join(os.curdir, "test_hdf5.h5")
_xmffile = os.path.join(os.curdir, "test_hdf5.xmf")


@pytest.fixture
def filepath():
    if os.path.exists(_xmffile):
        os.remove(_xmffile)
    if os.path.exists(_h5file):
        os.remove(_h5file)
    return _h5file


@pytest.mark.skipif("h5py is None", reason='h5py not available')
def test_hdf5_scalar(mesh, filepath):
    fs = FunctionSpace(mesh, "CG", 1)
    x = Function(fs, name='xcoord')
    x.interpolate(Expression("x[0]"))

    h5file = File(filepath)
    h5file << x

    h5out = h5py.File(filepath, 'r')
    xval = h5out['fields']['xcoord'][0, :, 0]
    assert np.max(np.abs(xval - x.dat.data)) < 1e-6


@pytest.mark.skipif("h5py is None", reason='h5py not available')
def test_hdf5_xdmf_header(mesh, filepath):
    fs = FunctionSpace(mesh, "CG", 1)
    x = Function(fs, name='xcoord')
    x.interpolate(Expression("x[0]"))

    h5file = File(filepath)
    h5file << x
    del h5file  # Close output file
    assert os.path.exists(_xmffile)
