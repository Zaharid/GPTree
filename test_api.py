import math

import cffi

ffi = cffi.FFI()

ffi.cdef("typedef struct zkdtree  zkdtree;")
ffi.cdef("zkdtree * zkdtree_load(const char * filename, char** errout);")
ffi.cdef("void zkdtree_delete(zkdtree *);")
ffi.cdef("size_t zkdtree_nfeatures(zkdtree *);")
ffi.cdef("double zkdtree_interpolate(zkdtree * tree, double * pt);")
ffi.cdef("void free(void *);")


C = ffi.dlopen("libcapi.so")

err = ffi.new("char **")

def test_basics():
    tree = C.zkdtree_load(b"r.tree", err)

    assert tree != ffi.NULL, ffi.string(err[0])

    assert C.zkdtree_nfeatures(tree) == 1

    val = C.zkdtree_interpolate(tree, [5.])

    assert math.isfinite(val)

    C.zkdtree_delete(tree)

    bad_tree = C.zkdtree_load(b"meson.build", err)

    assert bad_tree == ffi.NULL

    s = ffi.string(err[0])

    assert s

    C.free(err[0])
