from firedrake import *
import pytest
import numpy as np
from mpi4py import MPI


# Utility Functions and Fixtures

# NOTE we don't include interval mesh since many of the function spaces
# here are not defined on it.
@pytest.fixture(params=["square",
                        pytest.param("extruded", marks=pytest.mark.xfail(reason="extruded meshes not supported")),
                        "cube",
                        "tetrahedron",
                        pytest.param("immersedsphere", marks=pytest.mark.xfail(reason="immersed parent meshes not supported")),
                        pytest.param("periodicrectangle", marks=pytest.mark.xfail(reason="meshes made from coordinate fields are not supported"))])
def parentmesh(request):
    if request.param == "interval":
        return UnitIntervalMesh(1)
    elif request.param == "square":
        return UnitSquareMesh(1, 1)
    elif request.param == "extruded":
        return ExtrudedMesh(UnitSquareMesh(1, 1), 1)
    elif request.param == "cube":
        return UnitCubeMesh(1, 1, 1)
    elif request.param == "tetrahedron":
        return UnitTetrahedronMesh()
    elif request.param == "immersedsphere":
        return UnitIcosahedralSphereMesh()
    elif request.param == "periodicrectangle":
        return PeriodicRectangleMesh(3, 3, 1, 1)


@pytest.fixture(params=[0, 1, 100], ids=lambda x: f"{x}-coords")
def vertexcoords(request, parentmesh):
    size = (request.param, parentmesh.geometric_dimension())
    return pseudo_random_coords(size)


@pytest.fixture(params=[("CG", 2, FunctionSpace),
                        ("DG", 2, FunctionSpace)],
                ids=lambda x: "%s(%s%s)" % (x[2].__name__, x[0], x[1]))
def fs(request):
    return request.param


@pytest.fixture(params=[("CG", 2, VectorFunctionSpace),
                        ("N1curl", 2, FunctionSpace),
                        ("N2curl", 2, FunctionSpace),
                        ("N1div", 2, FunctionSpace),
                        ("N2div", 2, FunctionSpace),
                        ("N1curl", 2, VectorFunctionSpace),
                        ("N1div", 2, VectorFunctionSpace)],
                ids=lambda x: "%s(%s%s)" % (x[2].__name__, x[0], x[1]))
def vfs(request):
    return request.param


@pytest.fixture(params=[("CG", 2, TensorFunctionSpace),
                        ("BDM", 2, VectorFunctionSpace),
                        ("Regge", 1, FunctionSpace)],
                ids=lambda x: "%s(%s%s)" % (x[2].__name__, x[0], x[1]))
def tfs(request):
    return request.param


def pseudo_random_coords(size):
    """
    Get an array of pseudo random coordinates with coordinate elements
    between -0.5 and 1.5. The random numbers are consistent for any
    given `size` since `numpy.random.seed(0)` is called each time this
    is used.
    """
    np.random.seed(0)
    a, b = -0.5, 1.5
    return (b - a) * np.random.random_sample(size=size) + a


# Tests

# NOTE: these _spatialcoordinate tests should be equivalent to some kind of
# interpolation from a CG1 VectorFunctionSpace (I think)
@pytest.mark.xfail
def test_scalar_spatialcoordinate_interpolation(parentmesh, vertexcoords):
    vm = VertexOnlyMesh(parentmesh, vertexcoords)
    W = FunctionSpace(vm, "DG", 0)
    from functools import reduce
    from operator import mul
    expr = reduce(mul, SpatialCoordinate(parentmesh))
    w_expr = interpolate(expr, W)
    assert np.allclose(w_expr.dat.data_ro, np.prod(vertexcoords, axis = 1))


@pytest.mark.xfail
def test_scalar_function_interpolation(parentmesh, vertexcoords, fs):
    vm = VertexOnlyMesh(parentmesh, vertexcoords)
    fs_fam, fs_deg, fs_typ = fs
    V = fs_typ(parentmesh, fs_fam, fs_deg)
    W = FunctionSpace(vm, "DG", 0)
    from functools import reduce
    from operator import mul
    expr = reduce(mul, SpatialCoordinate(parentmesh))
    v = Function(V).project(expr)
    w_v = interpolate(v, W)
    assert np.allclose(w_v.dat.data_ro, np.prod(vertexcoords, axis = 1))


@pytest.mark.xfail
def test_vector_spatialcoordinate_interpolation(parentmesh, vertexcoords):
    vm = VertexOnlyMesh(parentmesh, vertexcoords)
    W = VectorFunctionSpace(vm, "DG", 0)
    expr = 2 * SpatialCoordinate(parentmesh)
    w_expr = interpolate(expr, W)
    assert np.allclose(w_expr.dat.data_ro, 2*np.asarray(vertexcoords))
    # assert np.allclose(w_expr.dat.data_ro, 2*vertexcoords)


@pytest.mark.xfail
def test_vector_function_interpolation(parentmesh, vertexcoords, vfs):
    vm = VertexOnlyMesh(parentmesh, vertexcoords)
    vfs_fam, vfs_deg, vfs_typ = vfs
    V = vfs_typ(parentmesh, vfs_fam, vfs_deg)
    W = VectorFunctionSpace(vm, "DG", 0)
    expr = 2 * SpatialCoordinate(parentmesh)
    v = Function(V).project(expr)
    w_v = interpolate(v, W)
    assert np.allclose(w_v.dat.data_ro, 2*np.asarray(vertexcoords))


@pytest.mark.xfail
def test_tensor_spatialcoordinate_interpolation(parentmesh, vertexcoords):
    vm = VertexOnlyMesh(parentmesh, vertexcoords)
    W = TensorFunctionSpace(vm, "DG", 0)
    x = SpatialCoordinate(parentmesh)
    expr = 2 * as_tensor([x,x])
    w_expr = interpolate(expr, W)
    result = 2 * np.asarray([[vertexcoords[i], vertexcoords[i]] for i in range(len(vertexcoords))])
    assert np.allclose(w_expr.dat.data_ro, result)


@pytest.mark.xfail
def test_tensor_function_interpolation(parentmesh, vertexcoords, tfs):
    vm = VertexOnlyMesh(parentmesh, vertexcoords)
    tfs_fam, tfs_deg, tfs_typ = tfs
    V = tfs_typ(parentmesh, tfs_fam, tfs_deg)
    W = TensorFunctionSpace(vm, "DG", 0)
    x = SpatialCoordinate(parentmesh)
    expr = 2 * as_tensor([x,x])
    v = Function(V).project(expr)
    result = 2 * np.asarray([[vertexcoords[i], vertexcoords[i]] for i in range(len(vertexcoords))])
    w_v = interpolate(v, W)
    assert np.allclose(w_v.dat.data_ro, result)

# TODO: Add parallel tests
# @pytest.mark.parallel
# def test_scalar_spatialcoordinate_interpolation_parallel(parentmesh, vertexcoords):
#     test_scalar_spatialcoordinate_interpolation(parentmesh, vertexcoords)
