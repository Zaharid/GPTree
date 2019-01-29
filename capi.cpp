#include "capi.h"
#include "tree.hpp"


#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>

#include <fstream>
#include <memory>

zkdtree * zkdtree_load(const char * filename){
	//The unqie_ptr is required by cereal
	auto *t = new ZKDTree::KDTree();
	std::ifstream is(filename, std::ios::binary);
	cereal::PortableBinaryInputArchive ar(is);
	//ar(t);
	ar(*t);
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


