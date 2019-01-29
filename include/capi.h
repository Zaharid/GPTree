#pragma once
#include <stddef.h>

typedef struct zkdtree  zkdtree;

#ifdef __cplusplus
extern "C"{
#endif
zkdtree * zkdtree_load(const char * filename);

size_t zkdtree_nfeatures(zkdtree *);

void zkdtree_delete(zkdtree *);

double zkdtree_interpolate(zkdtree * tree, double * pt);


#ifdef __cplusplus
}
#endif
