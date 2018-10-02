import numpy as np
import os

from libcpp cimport bool

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np

DTYPE = np.float64

ctypedef np.float64_t DTYPE_t



cdef extern from "capi.h":
    ctypedef void  zkdtree;
    ctypedef void zkdtree_result;

    zkdtree * zkdtree_load(const char * filename, char ** errout);

    void zkdtree_delete(zkdtree *);

    size_t zkdtree_nfeatures(zkdtree *);

    double zkdtree_interpolate(zkdtree *, double * pt);

    zkdtree_result * zkdtree_result_allocate();

    void zkdtree_result_delete(zkdtree_result *);

    void zkdtree_point_info(zkdtree *, double * pt, zkdtree_result * output);

    bool zkdtree_result_interpolable(zkdtree_result *);

    double zkdtree_result_central_value(zkdtree_result *);

    double zkdtree_result_variance(zkdtree_result *);


    void zkdtree_test_point(zkdtree *tree, size_t index, double *features_out,
		double *true_value_out, double *prediction_out);

    size_t zkdtree_nsamples(zkdtree *);

def quick_test(name):
    cdef char* err
    cdef zkdtree* obj
    cdef  np.ndarray[DTYPE_t, ndim=1] inp
    obj = zkdtree_load(os.fsencode(name), &err)
    if not obj:
        raise OSError(err)
    inp = np.zeros(zkdtree_nfeatures(obj))
    print(zkdtree_interpolate(obj, &inp[0]))
    zkdtree_delete(obj)

def loo_test(name):
    cdef char* err
    cdef zkdtree* obj
    cdef  np.ndarray[DTYPE_t, ndim=2] inp
    obj = zkdtree_load(os.fsencode(name), &err)
    if not obj:
        raise OSError(err)
    inp = np.empty((zkdtree_nsamples(obj), 2+zkdtree_nfeatures(obj)))
    for j in range(inp.shape[0]):
        zkdtree_test_point(obj, j, &inp[j,2], &inp[j,0], &inp[j,1])
    return inp
