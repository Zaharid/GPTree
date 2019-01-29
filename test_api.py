import pathlib

import cffi

ffi = cffi.FFI()

ffi.cdef("typedef struct zkdtree  zkdtree;")
ffi.cdef("zkdtree * zkdtree_load(const char * filename);")
ffi.cdef("void zkdtree_delete(zkdtree *);")
ffi.cdef("size_t zkdtree_nfeatures(zkdtree *);")
ffi.cdef("double zkdtree_interpolate(zkdtree * tree, double * pt);")


C = ffi.dlopen(str(pathlib.Path("bld/libcapi.so").absolute()))

tree = C.zkdtree_load(b"bld/tree.cereal")

assert C.zkdtree_nfeatures(tree) == 3

print(C.zkdtree_interpolate(tree, [25., 25., 0.]))

C.zkdtree_delete(tree)
