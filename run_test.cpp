#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <random>
#include <functional>

#include "tree.hpp"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using namespace ZKDTree;

TEST_CASE("Test distance", "[Data2D]"){
	auto dt = Data2D(2,2);
	REQUIRE(reduced_distance(dt, 0,1)==0);
	dt = Data2D(2,2,{1,2,3,4});
	REQUIRE(reduced_distance(dt, 0,1)==11);
}

TEST_CASE("Test 3d indexing", "[Data3D]"){
	auto dt = Data3D{2,2,3,
		                    {0  , 1  , 2,
		                     3  , 4  , 5,
							 //
		                     6  , 7  , 8,
							 9  , 10 , 11}};
	REQUIRE(dt(0,1,2)==5);
	REQUIRE(dt(1,1,1)==10);
}

TEST_CASE("Test tuple", "[Tuple]"){
	auto dis = std::vector<DistanceIndex>{{0.4, 1},{0.2,2}};
	std::sort(dis.begin(), dis.end());
	REQUIRE(dis.front().index==2);
}

TEST_CASE("Test tree invariants", "[KDTree]"){
    auto dt = Data2D(2,4);
    auto tree = KDTree(dt);
    REQUIRE(tree.nsamples() == 2);
    REQUIRE(tree.nfeatures() == 4);
}

TEST_CASE("Test tree construction", "[KDTree]"){
	std::mt19937 g(42);
	std::uniform_real_distribution<double> dist {0,100};
    size_t nsamples = 10833;
	auto gen = [&dist, &g](){return dist(g);};
    auto points = std::vector<double>(nsamples*3);
	std::generate(std::begin(points), std::end(points), gen);
    Data2D dt{nsamples, 3, points};
	KDTree tree {dt};
    REQUIRE(tree.nlevels() == 9);
    std::cout << tree.print_tree();
}
