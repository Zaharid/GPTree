import pathlib

import cffi

ffi = cffi.FFI()

ffi.cdef("typedef struct zkdtree  zkdtree;")
ffi.cdef("zkdtree * zkdtree_load(const char * filename, char** errout);")
ffi.cdef("void zkdtree_delete(zkdtree *);")
ffi.cdef("size_t zkdtree_nfeatures(zkdtree *);")
ffi.cdef("double zkdtree_interpolate(zkdtree * tree, double * pt);")
ffi.cdef("void free(void *);")


C = ffi.dlopen(str(pathlib.Path("bld/libcapi.so").absolute()))

err = ffi.new("char **")

def test_basics():
    tree = C.zkdtree_load(b"bld/tree.cereal", err)

    assert C.zkdtree_nfeatures(tree) == 3

    print(C.zkdtree_interpolate(tree, [25., 25., 0.]))

    C.zkdtree_delete(tree)

    bad_tree = C.zkdtree_load(b"meson.build", err)

    assert bad_tree == ffi.NULL

    s = ffi.string(err[0])

    assert(s)

    C.free(err[0])


