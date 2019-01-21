#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <random>
#include <functional>
#include <cmath>

#include "tree.hpp"

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

TEST_CASE("Test min_rdist", "[KDTree]"){
    auto dt = Data2D(4,2, {0,0
                          ,0,1
                          ,1,0,
                           1,1});
    auto tree = KDTree(dt);
    std::cout << "\n-----------\n";
    for (size_t inode = 0; inode < tree.nnodes(); inode++){
        std::cout << "(" << tree.getNodeBounds().at(0,inode,0) << "-"<< tree.getNodeBounds().at(1,inode,0) << ")";
        std::cout << " (" << tree.getNodeBounds().at(0,inode,1) << "-"<< tree.getNodeBounds().at(1,inode,1) << ")";
        std::cout << "\n";
    }
    std::cout << "Indexes:\n";
    for (auto i : tree.indexes){
        std::cout << i << "\n";
    }
    std::cout << tree.print_tree();
    std::cout << "\n----\n";
    double pt[] = {0.25,0.25};
    REQUIRE(tree.min_rdist(0,pt)==0);

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
