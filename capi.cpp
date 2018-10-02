#include "capi.h"
#include "tree.hpp"
#include <string.h>

zkdtree *zkdtree_load(const char *filename, char **error) {
  auto loaded = ZKDTree::load_tree(filename);
  if (std::holds_alternative<ZKDTree::KDTree>(loaded)) {
    auto *t = new ZKDTree::KDTree(std::move(std::get<ZKDTree::KDTree>(loaded)));
    return reinterpret_cast<zkdtree *>(t);
  } else {
    auto error_str = std::get<std::string>(loaded);
    auto size = sizeof(*error) * (error_str.size() + 1);
    *error = (char *)malloc(size);
    if (*error != nullptr) {
      strcpy(*error, error_str.c_str());
    }
    return nullptr;
  }
}

void zkdtree_delete(zkdtree *t) {
  delete reinterpret_cast<ZKDTree::KDTree *>(t);
}

size_t zkdtree_nfeatures(zkdtree *tree) {
  auto *t = reinterpret_cast<ZKDTree::KDTree *>(tree);
  return t->nfeatures();
}

double zkdtree_interpolate(zkdtree *tree, double *pt) {
  auto *t = reinterpret_cast<ZKDTree::KDTree *>(tree);
  auto val = ZKDTree::point_type(pt, t->nfeatures());
  return t->interpolate_single(val);
}

double zkdtree_interpolate_index(zkdtree *tree, size_t index) {
  auto *t = reinterpret_cast<ZKDTree::KDTree *>(tree);
  return t->interpolate_index(index);
}

zkdtree_result *zkdtree_result_allocate() {
  auto *r = new ZKDTree::interpolation_result();
  return reinterpret_cast<zkdtree_result *>(r);
}

void zkdtree_result_delete(zkdtree_result *result) {
  delete reinterpret_cast<ZKDTree::interpolation_result *>(result);
}

void zkdtree_point_info(zkdtree *tree, double *pt, zkdtree_result *output) {
  auto *t = reinterpret_cast<ZKDTree::KDTree *>(tree);
  auto *r = reinterpret_cast<ZKDTree::interpolation_result *>(output);
  auto val = ZKDTree::point_type(pt, t->nfeatures());
  *r = t->interpolate_single_result(val);
}

double zkdtree_result_central_value(zkdtree_result *result) {
  auto *r = reinterpret_cast<ZKDTree::interpolation_result *>(result);
  return r->central_value;
}

double zkdtree_result_variance(zkdtree_result *result) {
  auto *r = reinterpret_cast<ZKDTree::interpolation_result *>(result);
  return r->variance;
}

bool zkdtree_result_interpolable(zkdtree_result *result) {
  auto *r = reinterpret_cast<ZKDTree::interpolation_result *>(result);
  return r->interpolable;
}

//TODO: Move this to some internal header
void zkdtree_test_point(zkdtree *tree, size_t index, double *features_out,
		double *true_value_out, double *prediction_out) {
  auto *t = reinterpret_cast<ZKDTree::KDTree *>(tree);
  for (size_t i = 0; i <= t->nfeatures(); i++) {
    features_out[i] = t->data.at(index, i);
  }
  *true_value_out = t->responses[index];
  *prediction_out = t->interpolate_index(index);
}


size_t zkdtree_nsamples(zkdtree *tree) {
  auto *t = reinterpret_cast<ZKDTree::KDTree *>(tree);
  return t->nsamples();
}
