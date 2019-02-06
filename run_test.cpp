#include "tree.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

constexpr double pi = 3.141592653589793238463;


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

TEST_CASE("Test apply permutation", "[Permutation]"){
	auto v = std::vector<double>{0,1,2,3,4,5};
	auto indexes = std::vector<size_t>{4,0,3,2,4,1};
	auto res = apply_permutation(v, indexes);
	REQUIRE(std::equal(res.begin(), res.end(), indexes.begin(), indexes.end()));
}

TEST_CASE("Test tree invariants", "[KDTree]"){
    auto dt = Data2D(2,4);
	auto y = std::vector<double>{1,3};
    auto tree = KDTree(std::move(dt), std::move(y));
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

TEST_CASE("Test tree construction", "[KDTree]") {
  std::mt19937 g(43);
  std::uniform_real_distribution<double> dist{0, 100};
  size_t nsamples = 10833;
  auto gen = [&dist, &g]() { return dist(g); };
  auto points = std::vector<double>(nsamples * 3);
  std::generate(std::begin(points), std::end(points), gen);
  Data2D dt{nsamples, 3, points};
  auto y = std::vector<double>{};
  auto f = [](auto pt) {
    return std::sin(pt[0] / 100. * 2 * pi) * std::cos(pt[1] / 200. * 2 * pi) *
           exp(-pt[2] / 100);
  };
  for (size_t i = 0; i < nsamples; i++) {
    y.push_back(f(dt.at(i)));
  }
  KDTree tree{dt, y, 20, 1e-1};
  REQUIRE(tree.nlevels() == 9);
  std::cout << tree.print_tree();
  auto pt = std::vector<double>{10, 10, 25};
  auto val = f(pt);
  auto interp = tree.interpolate_single(pt);
  std::cout << val << "\n";
  std::cout << interp << "\n";

  REQUIRE(is_close(interp, val, 1e-1));
  save_tree(tree, "tree.cereal");
  auto loaded_tree = load_tree("tree.cereal");
  auto nis = loaded_tree.interpolate_single(pt);
  REQUIRE(interp == nis);
}
TEST_CASE("Test interpolation", "[KDTree]"){
	std::mt19937 g(44);
	std::uniform_real_distribution<double> dist {0,100};
    size_t nsamples = 100000;
	auto gen = [&dist, &g](){return dist(g);};
    auto points = std::vector<double>(nsamples*3);
	std::generate(std::begin(points), std::end(points), gen);
    Data2D dt{nsamples, 3, points};
	auto y = std::vector<double>{};
	auto f = [](auto pt){return std::sin(pt[0]/100.*2*pi)*std::cos(pt[1]/200.*2*pi)*exp(-pt[2]/100);};
	for(size_t i=0; i<nsamples; i++){
		y.push_back(f(dt.at(i)));
	}
	KDTree tree {dt, y, 5, 0};
    std::cout << tree.print_tree();
	auto pt = std::vector<double>{25, 25, 25};
	auto val = f(pt);
	auto interp = tree.interpolate_single(pt);
	std::cout << val << "\n";
	std::cout << interp << "\n";

	REQUIRE(is_close(interp, val, 1e-1));

	auto obj = tree.interpolate_single_result(pt);
	REQUIRE(obj.central_value == interp);
	REQUIRE(obj.interpolable);

	pt = {1000, 1000, 1000};
	obj = tree.interpolate_single_result(pt);
	REQUIRE(is_close(obj.variance, 1));
	REQUIRE(!obj.interpolable);

	auto pt2 = dt.at(0);
	obj = tree.interpolate_single_result(pt2);
	REQUIRE(is_close(obj.variance, 0));

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
	KDTree tree {dt, y, 10, 1e-8};
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
	std::sort_heap(res.begin(), res.end());
	for (size_t i=0; i < nele; i++ ){
		REQUIRE(v[i].rdistance == res[i].rdistance);
		// This is not true because we moved around the internal data
		//REQUIRE(v[i].index == res[i].index);
	}
}
