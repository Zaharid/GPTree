#include "capi.h"
#include "tree.hpp"

zkdtree * zkdtree_load(const char * filename){
	auto *t = new ZKDTree::KDTree(ZKDTree::load_tree(filename));
	return reinterpret_cast<zkdtree*>(t);
}

size_t zkdtree_nfeatures(zkdtree* tree){
	auto *t = reinterpret_cast<ZKDTree::KDTree*>(tree);
	return t->nfeatures();
}

void zkdtree_delete(zkdtree * t){
	delete reinterpret_cast<ZKDTree::KDTree*>(t);
}

double zkdtree_interpolate(zkdtree* tree, double * pt){
	auto *t = reinterpret_cast<ZKDTree::KDTree*>(tree);
	auto val = ZKDTree::point_type(pt, t->nfeatures());
	return t->interpolate_single_bruteforce(val);
}


