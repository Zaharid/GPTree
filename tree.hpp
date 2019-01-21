#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <numeric>
#include <string>
#include <sstream>
#include <functional>
#include <memory>
#include <assert.h>

#include "gsl/span"


namespace ZKDTree {

using point_type = gsl::span<double>;

namespace {
constexpr auto inf = std::numeric_limits<double>::infinity();
constexpr auto neg_inf = -inf;
constexpr double pi = 3.141592653589793238463;
constexpr double sqrt2 = std::sqrt(2);
constexpr size_t default_leaf_size = 40;
}


struct NodeData {
  double radius;
  size_t start;
  size_t end;
  bool is_leaf;
};

struct Data2D {
  size_t nsamples;
  size_t nfeatures;
  std::vector<double> data;

  Data2D(size_t nsamples, size_t nfeatures, std::vector<double> data)
      : nsamples(nsamples), nfeatures(nfeatures), data(std::move(data)) {assert(nsamples*nfeatures == this->data.size());}

  Data2D(size_t nsamples, size_t nfeatures)
      : nsamples(nsamples), nfeatures(nfeatures), data(nsamples * nfeatures) {}

  double & at(size_t i, size_t j) { return data[i * nfeatures + j]; }

  const double & at(size_t i, size_t j) const { return data[i * nfeatures + j]; }

  point_type at(size_t i) {return {&data.data()[i * nfeatures], static_cast<long>(nfeatures)};}

  double & operator()(size_t i, size_t j) { return at(i, j); }
};

struct Data3D {
  size_t kind;
  size_t nnodes;
  size_t nfeatures;
  std::vector<double> data;

  Data3D(size_t kind, size_t nnodes, size_t nfeatures,
         std::vector<double> data)
      : kind(kind), nnodes(nnodes), nfeatures(nfeatures), data(std::move(data)) {assert(this->data.size() == kind*nnodes*nfeatures);}

  Data3D(size_t kind, size_t nnodes, size_t nfeatures)
      : kind(kind), nnodes(nnodes), nfeatures(nfeatures),
        data(nnodes * nfeatures * kind) {}

  double & at(size_t k, size_t i, size_t j)  {
    return data[k * nnodes * nfeatures + i*nfeatures + j];
  }

  double at (size_t k, size_t i, size_t j) const {
    return data[k * nnodes * nfeatures + i*nfeatures + j];
  }
  double & operator()(size_t k, size_t i, size_t j) { return at(k, i, j); }
};

struct IndexComparator{
    Data2D &data;
    size_t split_dim;
    IndexComparator(Data2D & data, size_t split_dim)
        : data(data), split_dim(split_dim) {}
    bool operator()(size_t i, size_t j){
        return data.at(i, split_dim) < data.at(j, split_dim);
    }
};




struct DistanceIndex {
  double rdistance;
  size_t index;
  DistanceIndex(double rdistance, size_t index)
      : rdistance(rdistance), index(index) {}
  bool operator<(DistanceIndex const &other) {
    return this->rdistance < other.rdistance;
  }
};

double reduced_distance(Data2D const &data, size_t i, size_t j) {
  double s = 0;
  for (size_t k = 0; k < data.nfeatures; k++) {
    auto d = data.at(i, k) - data.at(j, k);
    s += d*d;
  }
  return s;
}

double reduced_distance(Data2D const &data, size_t i, Data2D const &otherdata, size_t j){
  double s = 0;
  for (size_t k = 0; k < data.nfeatures; k++) {
    auto d = data.at(i, k) - otherdata.at(j, k);
    s += d*d;
  }
  return s;
}

double reduced_distance(Data2D const &data, size_t i, point_type const &point){
  double s = 0;
  for (size_t k = 0; k < data.nfeatures; k++) {
    auto d = data.at(i, k) - point.at(k);
    s += d*d;
  }
  return s;
}




struct KDTree {
  //NOTE: lead_size must go first because it is needed to initialize node_bounds,
  //and the idiotic initializer list feature needs to mind the order.
  size_t leaf_size = default_leaf_size;
  Data2D data;
  Data3D node_bounds;
  std::vector<NodeData> node_data;
  std::vector<size_t> indexes;
  size_t nsamples() { return data.nsamples; }
  size_t nfeatures() { return data.nfeatures; }
  size_t nlevels() {
    //This is verbose on purpose so the debugger is easy to hook up
    size_t ratio{(nsamples() - 1) / leaf_size};
    if (ratio < 2){
        return 2;
    }else{
        return size_t(std::log2(ratio) + 1);
    }
  }
  size_t nnodes() { return (1 << nlevels()) - 1; }

  KDTree(Data2D &datain, size_t leaf_size=default_leaf_size):leaf_size(leaf_size), data(datain), node_bounds(2, nnodes(), datain.nfeatures) {
    auto halfbound = static_cast<long>(nnodes() * nfeatures());
    auto begin = std::begin(node_bounds.data);
    std::fill(begin, begin + halfbound, inf);
    std::fill(begin + halfbound, begin + 2 * halfbound, neg_inf);
    node_data.assign(nnodes(), NodeData());

    //Fill indexes
    indexes = std::vector<size_t>(nsamples());
    std::iota(indexes.begin(), indexes.end(), 0);

    //Build
    recursive_build(0,0,nsamples());
  }
  void recursive_build(size_t inode, size_t start, size_t end) {
    auto npoints = end - start;
    auto nmid = npoints / 2;
    init_node(inode, start, end);
    if (2*inode + 1 >=nnodes()){
        node_data[inode].is_leaf = true;
    }else{
        node_data[inode].is_leaf = false;
        auto i_max = find_node_split_dim(start, end);
        IndexComparator comp {data, i_max};
        std::nth_element(&indexes[start], &indexes[start+nmid], &indexes[end], comp);
        recursive_build(2*inode +1, start, start + nmid);
        recursive_build(2*inode+2, start+nmid,end);
    }

  }

  /** Because there is no easy way of slicing an std::vector, we take a a pair of indices. This is no worse than a pair of iterators and hides the ugly types.**/
  size_t find_node_split_dim(size_t start_id, size_t end_id) {

      double maxdelta = neg_inf;
      size_t max_split_dim = 0;
      for(size_t j=0; j<nfeatures(); j++){
          double maxval = neg_inf;
          double minval = inf;
          for (auto ind_in_index_array = start_id; ind_in_index_array < end_id; ++ind_in_index_array){
              auto data_ind = indexes[ind_in_index_array];
              auto val = data.at(data_ind,j);
              if (val < minval){
                  minval = val;
              }
              if (val > maxval){
                  maxval = val;
              }
          }
          auto delta = maxval - minval;
          if (delta > maxdelta){
              maxdelta = delta;
              max_split_dim = j;
          }
      }
      return max_split_dim;
  }

  void init_node(size_t inode, size_t idx_start, size_t idx_end){
      for (auto i =idx_start; i<idx_end; i++){
          auto data_index = indexes[i];
          for(size_t j=0; j<nfeatures(); j++){
              auto data_val = data.at(data_index, j);
              auto& lowbound = node_bounds.at(0,inode,j);
              lowbound = std::min(lowbound, data_val);
              auto& highbound = node_bounds.at(1, inode, j);
              highbound = std::max(highbound, data_val);

          }
      }
      double rad = 0;
      for (size_t j=0; j<nfeatures(); j++){
          rad += std::pow(0.5*(node_bounds.at(1,inode,j) - node_bounds.at(0,inode,j)), 2);
      }
      rad = std::sqrt(rad);
      node_data[inode].radius = rad;
      node_data[inode].start = idx_start;
      node_data[inode].end = idx_end;
  }

  std::string print_tree(){
      if (node_data.empty()){
          return "";
      }
      std::stringstream s{};
      //Need all this stuff to make it recursive
      std::function<void(std::stringstream&, size_t, size_t)> add_info_from_node;
      add_info_from_node = [&](std::stringstream & stream, size_t inode, size_t tab)
      {
          auto &node = node_data[inode];
          auto spaces = std::string(tab, ' ');
          if (node.is_leaf){
              stream << spaces << "Leaf node " << inode << " containing " << node.start << "-" << node.end << "\n";
          }else{
              stream << spaces << "Parent node " << inode << "\n";
              add_info_from_node(stream, 2*inode+1, tab+1);
              add_info_from_node(stream, 2*inode+2, tab+1);
          }
      };
      add_info_from_node(s, 0, 0);
      return s.str();
  }

  double min_rdist(size_t inode, const point_type & pt){
      double rdist = 0;
      for(size_t j=0; j < nfeatures(); j++){
          //lo pt hi -> 0 inside for that dimenstion
          //pt lo hi -> lo - pt
          //lo hi pt -> pt - hi
          auto lo = node_bounds(0, inode, j) - pt.at(j);
          auto hi = pt.at(j) - node_bounds(1, inode, j);
          auto d = std::max(lo, 0.) + std::max(hi, 0.);
          rdist += d*d;
      }
      return rdist;
  }

  size_t getLeafSize(){return  leaf_size;}
  const Data3D& getNodeBounds(){return  node_bounds;}





};

} // namespace ZKDTree
