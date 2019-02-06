#include "capi.h"
#include "tree.hpp"

zkdtree * zkdtree_load(const char * filename){
	auto *t = new ZKDTree::KDTree(ZKDTree::load_tree(filename));
	return reinterpret_cast<zkdtree*>(t);
}

void zkdtree_delete(zkdtree * t){
	delete reinterpret_cast<ZKDTree::KDTree*>(t);
}

size_t zkdtree_nfeatures(zkdtree* tree){
	auto *t = reinterpret_cast<ZKDTree::KDTree*>(tree);
	return t->nfeatures();
}

double zkdtree_interpolate(zkdtree* tree, double * pt){
	auto *t = reinterpret_cast<ZKDTree::KDTree*>(tree);
	auto val = ZKDTree::point_type(pt, t->nfeatures());
	return t->interpolate_single(val);
}

zkdtree_result * zkdtree_result_allocate(){
	auto *r = new  ZKDTree::interpolation_result();
	return reinterpret_cast<zkdtree_result*>(r);
}

void zkdtree_result_delete(zkdtree_result * result){
	delete reinterpret_cast<ZKDTree::interpolation_result*>(result);
}

void zkdtree_point_info(zkdtree * tree, double * pt, zkdtree_result *output){
	auto *t = reinterpret_cast<ZKDTree::KDTree*>(tree);
	auto *r = reinterpret_cast<ZKDTree::interpolation_result*>(output);
	auto val = ZKDTree::point_type(pt, t->nfeatures());
	*r = t->interpolate_single_result(val);
}

double zkdtree_result_central_value(zkdtree_result * result){
	auto *r = reinterpret_cast<ZKDTree::interpolation_result*>(result);
	return r->central_value;
}

double zkdtree_result_variance(zkdtree_result * result){
	auto *r = reinterpret_cast<ZKDTree::interpolation_result*>(result);
	return r->variance;
}

bool zkdtree_result_interpolable(zkdtree_result * result){
	auto *r = reinterpret_cast<ZKDTree::interpolation_result*>(result);
	return r->interpolable;
}
