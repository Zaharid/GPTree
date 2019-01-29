#include "tree.hpp"

#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/vector.hpp>

#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <random>
#include <functional>
#include <cmath>


#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using namespace ZKDTree;


bool is_close(double a, double b, double rtol=1e-5, double atol=1e-10){
    return std::abs(a - b) <= (atol + rtol*std::abs(b));
}

TEST_CASE("Test distance", "[Data2D]"){
	auto dt = Data2D(2,2);
    REQUIRE(is_close(reduced_distance(dt, 0,1), 0));
	dt = Data2D(2,2,{1,2,3,4});
    auto dt2 = dt;
    REQUIRE(is_close(reduced_distance(dt, 0,0), 0));
    REQUIRE(is_close(reduced_distance(dt, 0,1), 8));
    REQUIRE(is_close(reduced_distance(dt, 0, dt2, 1), 8));
    std::cout << dt2.at(1)[0] << "\n";
    std::cout << dt2.at(1)[1] << "\n";
    REQUIRE(is_close(reduced_distance(dt, 0, dt2.at(1)), 8));
    double pt[2] = {0,1};
    REQUIRE(is_close(reduced_distance(dt,0,pt), 2));


}

TEST_CASE("Test 3d indexing", "[Data3D]"){
    auto dt = Data3D{2,3,4,
                            {0  ,  1  , 2,  20,
                             3  ,  4  , 5,  50,
                             30 ,  40 , 50, 500,
                             //
                             6  , 7  , 8,  80,
                             9  , 10 , 11, 110,
                             90 , 100, 110,1100
                             }};
	REQUIRE(dt(0,1,2)==5);
	REQUIRE(dt(1,1,1)==10);
}

TEST_CASE("Test kernel matrix", "[GP]"){
	auto dt = Data2D{2, 3, {1,0,0,
		                    0,0,1}};
	auto lower = rbf_lo_matrix(dt, 2);
	std::cout << lower(0,0) << "\t";
	std::cout << lower(0,1) << "\n";
	std::cout << lower(1,0) << "\t";
	std::cout << lower(1,1) << "\n";
	REQUIRE(is_close(lower(0,0), 1.));
	//REQUIRE(is_close(lower(0,1), 0));
	REQUIRE(is_close(lower(1,1), 1.));
	REQUIRE(is_close(lower(1,0), std::exp(-1./4.)));
	add_noise(lower, 0.1);
	std::cout << lower(0,0) << "\t";
	std::cout << lower(0,1) << "\n";
	std::cout << lower(1,0) << "\t";
	std::cout << lower(1,1) << "\n";
	REQUIRE(is_close(lower(0,0), 1. + 0.01));
	auto chol = lapack_cholesky_factor(lower);
	//TODO??
	//REQUIRE(is_close(lower(0,0), chol(0,0)*chol(0,0) + chol(0,1)*chol(0,1)));
	auto y = std::vector<double>{5,6};
	auto sol = lapack_cholesky_solve(chol, y);
	REQUIRE(is_close(lower(0,0)*sol[0] + lower(0,1)*sol[1], 5));
	REQUIRE(is_close(lower(0,1)*sol[0] + lower(1,1)*sol[1], 6));

	//auto tree = KDTree(dt, y, 2, 0.1);
	auto training_pivots = compute_training_pivots(dt, y, 2., 0.1);
	//Need the lambda because it doesn't fill the default arguments of is_close.
	REQUIRE(std::equal(sol.begin(), sol.end(), training_pivots.begin(),
				[](auto x, auto y){return is_close(x, y);}));
}

TEST_CASE("Test tuple", "[Tuple]"){
	auto dis = std::vector<DistanceIndex>{{0.4, 1},{0.2,2}};
	std::sort(dis.begin(), dis.end());
	REQUIRE(dis.front().index==2);
}

TEST_CASE("Test apply permutation", "[Permutation]"){
	auto v = std::vector<double>{0,1,2,3,4,5};
	auto indexes = std::vector<size_t>{4,0,3,2,4,1};
	auto res = apply_permutation(v, indexes);
	REQUIRE(std::equal(res.begin(), res.end(), indexes.begin(), indexes.end()));
}

TEST_CASE("Test tree invariants", "[KDTree]"){
    auto dt = Data2D(2,4);
	auto y = std::vector<double>{1,3};
    auto tree = KDTree(dt, y);
    REQUIRE(tree.nsamples() == 2);
    REQUIRE(tree.nfeatures() == 4);
}

TEST_CASE("Test min_rdist", "[KDTree]"){
    auto dt = Data2D(4,2, {0,0
                          ,0,1
                          ,1,0,
                           1,1});
	auto y = std::vector<double>(4);
    auto tree = KDTree(dt, y);
    std::cout << "\n-----------\n";
    for (size_t inode = 0; inode < tree.nnodes(); inode++){
        std::cout << "(" << tree.getNodeBounds().at(0,inode,0) << "-"<< tree.getNodeBounds().at(1,inode,0) << ")";
        std::cout << " (" << tree.getNodeBounds().at(0,inode,1) << "-"<< tree.getNodeBounds().at(1,inode,1) << ")";
        std::cout << "\n";
    }
    std::cout << "Indexes:\n";
    std::cout << tree.print_tree();
    std::cout << "\n----\n";
    double pt[] = {0.25,0.25};
    REQUIRE(tree.min_rdist(0,pt)==0);

}

TEST_CASE("Test tree construction", "[KDTree]"){
	std::mt19937 g(43);
	std::uniform_real_distribution<double> dist {0,100};
    size_t nsamples = 10833;
	auto gen = [&dist, &g](){return dist(g);};
    auto points = std::vector<double>(nsamples*3);
	std::generate(std::begin(points), std::end(points), gen);
    Data2D dt{nsamples, 3, points};
	auto y = std::vector<double>{};
	auto f = [](auto pt){return std::sin(pt[0]/100.*2*pi)*std::cos(pt[1]/200.*2*pi)*exp(-pt[2]/100);};
	for(size_t i=0; i<nsamples; i++){
		y.push_back(f(dt.at(i)));
	}
	KDTree tree {dt, y, 10, 1e-10, 1e-10};
    REQUIRE(tree.nlevels() == 9);
    std::cout << tree.print_tree();
	auto pt = std::vector<double>{25, 25, 25};
	auto val = f(pt);
	auto interp2 = tree.interpolate_single_bruteforce(pt);
	auto interp = tree.interpolate_single(pt);
	std::cout << val << "\n";
	std::cout << interp << "\n";
	std::cout << interp2 << "\n";

	REQUIRE(is_close(interp, val, 1e-1));
	{
	std::ofstream os ("tree.cereal", std::ios::binary);
	cereal::PortableBinaryOutputArchive ar(os);
	ar(tree);
	}
	{
		std::ifstream is("tree.cereal", std::ios::binary);
		cereal::PortableBinaryInputArchive ar(is);
		KDTree loaded_tree;
		ar(loaded_tree);
		auto newinterp = loaded_tree.interpolate_single_bruteforce(pt);
		REQUIRE(interp2 == newinterp);
		auto nis = loaded_tree.interpolate_single(pt);
		REQUIRE(interp == nis);
	}




}

/*
TEST_CASE("Test interpolation", "[KDTree]"){
	std::mt19937 g(43);
	std::uniform_real_distribution<double> dist {0,100};
    size_t nsamples = 1000;
	auto gen = [&dist, &g](){return dist(g);};
    auto points = std::vector<double>(nsamples*3);
	std::generate(std::begin(points), std::end(points), gen);
    Data2D dt{nsamples, 3, points};
	auto y = std::vector<double>{};
	auto f = [](auto pt){return std::sin(pt[0]/100.*2*pi)*std::cos(pt[1]/200.*2*pi)*exp(-pt[2]/100);};
	for(size_t i=0; i<nsamples; i++){
		y.push_back(f(dt.at(i)));
	}
	KDTree tree {dt, y, 3, 1e-10, 1e-2};
    std::cout << tree.print_tree();
	auto pt = std::vector<double>{25, 25, 25};
	auto val = f(pt);
	auto interp2 = tree.interpolate_single_bruteforce(pt);
	auto interp = tree.interpolate_single(pt);
	std::cout << val << "\n";
	std::cout << interp << "\n";
	std::cout << interp2 << "\n";

	REQUIRE(is_close(interp, val, 1e-1));

}

TEST_CASE("Test max_k_heap_push", "[max_heap]"){
	std::vector<DistanceIndex> heap;
	size_t k = 3;
	max_k_heap_push({1, 0}, heap, k);
	REQUIRE(heap.size() == 1);
	max_k_heap_push({3, 0}, heap, k);
	REQUIRE(heap.size() == 2);
	max_k_heap_push({2, 0}, heap, k);
	REQUIRE(heap.size() == 3);
	max_k_heap_push({4, 0}, heap, k);
	REQUIRE(heap.size() == 3);
	REQUIRE(heap.front().rdistance == 3);
	max_k_heap_push({-1, 0}, heap, k);
	REQUIRE(heap.size() == 3);
	REQUIRE(heap.front().rdistance == 2);
	std::sort_heap(heap.begin(), heap.end());
	REQUIRE(heap.front().rdistance == -1);

}

TEST_CASE("Test query", "[KDTree]"){
	std::mt19937 g(47);
	std::uniform_real_distribution<double> dist {0,100};
    size_t nsamples = 1000;
	auto gen = [&dist, &g](){return dist(g);};
    auto points = std::vector<double>(nsamples*3);
	std::generate(std::begin(points), std::end(points), gen);
    Data2D dt{nsamples, 3, points};
	auto y = std::vector<double>{};
	auto f = [](auto pt){return std::sin(pt[0]/100.*2*pi)*std::cos(pt[1]/200.*2*pi)*exp(-pt[2]/100);};
	for(size_t i=0; i<nsamples; i++){
		y.push_back(f(dt.at(i)));
	}
	KDTree tree {dt, y, 10, 1e-8, 1e-7};
	auto pt = std::vector<double>{50., 50., 50.};
	size_t nele = 50;
	auto res = tree.query(pt, nele);
	for (auto & it : res){
		std::cout << it.rdistance << "\n";

	}
	auto v = std::vector<DistanceIndex>{};
	for(size_t i=0; i < nsamples; i++){
		v.emplace_back(reduced_distance(dt, i, pt), i);
	}
	std::nth_element(v.begin(), v.begin()+nele, v.end());
	std::sort(v.begin(), v.begin()+nele);
	//std::sort(res.begin(), res.end());
	for (size_t i=0; i < nele; i++ ){
		REQUIRE(v[i].rdistance == res[i].rdistance);
		// This is not true because we moved around the internal data
		//REQUIRE(v[i].index == res[i].index);
	}





}
