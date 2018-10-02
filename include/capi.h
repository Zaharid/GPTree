#pragma once
#include <stdbool.h>
#include <stddef.h>

typedef struct zkdtree  zkdtree;
typedef struct zkdtree_result zkdtree_result;

#ifdef __cplusplus
extern "C"{
#endif
zkdtree * zkdtree_load(const char * filename, char ** errout);

void zkdtree_delete(zkdtree *);

size_t zkdtree_nfeatures(zkdtree *);

double zkdtree_interpolate(zkdtree *, double * pt);

double zkdtree_interpolate_index(zkdtree *, size_t index);

zkdtree_result * zkdtree_result_allocate(void);

void zkdtree_result_delete(zkdtree_result *);

void zkdtree_point_info(zkdtree *, double * pt, zkdtree_result * output);

bool zkdtree_result_interpolable(zkdtree_result *);

double zkdtree_result_central_value(zkdtree_result *);

double zkdtree_result_variance(zkdtree_result *);

//TODO: Move these away

size_t zkdtree_nsamples(zkdtree *);

void zkdtree_test_point(zkdtree *tree, size_t index, double *features_out,
		double *true_value_out, double *prediction_out);

#ifdef __cplusplus
}
#endif
