#pragma once

#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include "gsl/span"

namespace ZKDTree {

constexpr const std::uint32_t magic = 0x12ABED;
constexpr const std::uint32_t tree_data_version = 1;

using point_type = gsl::span<double>;

namespace {
constexpr auto inf = std::numeric_limits<double>::infinity();
constexpr auto neg_inf = -inf;
constexpr size_t default_leaf_size = 40;
} // namespace

struct NodeData {
  double radius;
  double training_sum;
  double training_min;
  double training_max;
  size_t start;
  size_t end;
  bool is_leaf;

  template <class Archive> void serialize(Archive &ar) {
    ar(radius, training_sum, training_min, training_max, start, end, is_leaf);
  }
};

struct Data2D {
  size_t nsamples;
  size_t nfeatures;
  std::vector<double> data;

  Data2D(size_t nsamples, size_t nfeatures, std::vector<double> data)
      : nsamples(nsamples), nfeatures(nfeatures), data(std::move(data)) {
    assert(nsamples * nfeatures == this->data.size());
  }

  Data2D(size_t nsamples, size_t nfeatures)
      : nsamples(nsamples), nfeatures(nfeatures), data(nsamples * nfeatures) {}

  // For serialization
  Data2D() {}

  double &at(size_t i, size_t j) { return data[i * nfeatures + j]; }

  const double &at(size_t i, size_t j) const { return data[i * nfeatures + j]; }

  point_type at(size_t i) {
    return {&data.data()[i * nfeatures], static_cast<long>(nfeatures)};
  }

  double &operator()(size_t i, size_t j) { return at(i, j); }

  template <class Archive> void serialize(Archive &ar) {
    ar(nsamples, nfeatures, data);
  }
};

struct Data3D {
  size_t kind;
  size_t nnodes;
  size_t nfeatures;
  std::vector<double> data;

  Data3D(size_t kind, size_t nnodes, size_t nfeatures, std::vector<double> data)
      : kind(kind), nnodes(nnodes), nfeatures(nfeatures),
        data(std::move(data)) {
    assert(this->data.size() == kind * nnodes * nfeatures);
  }

  Data3D(size_t kind, size_t nnodes, size_t nfeatures)
      : kind(kind), nnodes(nnodes), nfeatures(nfeatures),
        data(nnodes * nfeatures * kind) {}

  // For serialization
  Data3D() {}

  double &at(size_t k, size_t i, size_t j) {
    return data[k * nnodes * nfeatures + i * nfeatures + j];
  }

  double at(size_t k, size_t i, size_t j) const {
    return data[k * nnodes * nfeatures + i * nfeatures + j];
  }
  double &operator()(size_t k, size_t i, size_t j) { return at(k, i, j); }

  template <class Archive> void serialize(Archive &ar) {
    ar(kind, nnodes, nfeatures, data);
  }
};

struct IndexComparator {
  Data2D &data;
  size_t split_dim;
  IndexComparator(Data2D &data, size_t split_dim)
      : data(data), split_dim(split_dim) {}
  bool operator()(size_t i, size_t j) {
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

void max_k_heap_push(DistanceIndex element, std::vector<DistanceIndex> &heap,
                     size_t k) {
  if (heap.size() < k) {
    heap.push_back(element);
    std::push_heap(heap.begin(), heap.end());
    return;
  }
  auto largest = heap.front();
  if (element < largest) {
    std::pop_heap(heap.begin(), heap.end());
    heap.back() = element;
    std::push_heap(heap.begin(), heap.end());
  }
}

Data2D apply_permutation(Data2D const &in, std::vector<size_t> const &perm) {
  auto res = Data2D{in.nsamples, in.nfeatures};
  for (size_t i = 0; i < in.nsamples; i++) {
    auto source_index = perm[i];
    for (size_t j = 0; j < in.nfeatures; j++) {
      res.at(i, j) = in.at(source_index, j);
    }
  }
  return res;
}

std::vector<double> apply_permutation(std::vector<double> const &in,
                                      std::vector<size_t> &perm) {
  auto res = std::vector<double>{};
  res.reserve(in.size());
  for (size_t i = 0; i < in.size(); i++) {
    res.push_back(in[perm[i]]);
  }
  return res;
}

template <typename Functor>
void fill_uplo_packed(double *data, size_t n, Functor function) {
  auto k = 0;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < i + 1; j++) {
      data[k] = function(j, i);
      k++;
    }
  }
}

struct interpolation_storage {
  std::vector<double> data;
  double *bstart;
  double *wstart;
  double *ystart;
  double *covmatstart;
  interpolation_storage(size_t nneighbours)
      : data(3 * nneighbours + (nneighbours + nneighbours * nneighbours) / 2),
        bstart(&data[0]), wstart(&data[nneighbours]),
        ystart(&data[2 * nneighbours]), covmatstart(&data[3 * nneighbours]) {}
};

double reduced_distance(Data2D const &data, size_t i, size_t j) {
  double s = 0;
  for (size_t k = 0; k < data.nfeatures; k++) {
    auto d = data.at(i, k) - data.at(j, k);
    s += d * d;
  }
  return s;
}

double reduced_distance(Data2D const &data, size_t i, Data2D const &otherdata,
                        size_t j) {
  double s = 0;
  for (size_t k = 0; k < data.nfeatures; k++) {
    auto d = data.at(i, k) - otherdata.at(j, k);
    s += d * d;
  }
  return s;
}

double reduced_distance(Data2D const &data, size_t i, point_type const &point) {
  double s = 0;
  for (size_t k = 0; k < data.nfeatures; k++) {
    auto d = data.at(i, k) - point.at(k);
    s += d * d;
  }
  return s;
}

double basic_rbf(double rdist, double scale) {
  return std::exp(-rdist / (2. * scale * scale));
}


/** Add noise_scale² to the diagonal of data in place.
 */
void add_noise(Data2D &data, double noise_scale) {
  for (size_t i = 0; i < data.nsamples; i++) {
    data.at(i, i) += noise_scale * noise_scale;
  }
}


extern "C" void dppsv_(char *uplo, int &n, int &nrhs, double *a, double *b,
                       int &ldb, int *info);



//Could we represent interpolable in the sign of variance?
struct interpolation_result{
	double central_value;
	double variance;
	bool interpolable;
};

struct KDTree {
  //
  // NOTE: lead_size must go first because it is needed to initialize
  // node_bounds, and the idiotic initializer list feature needs to mind the
  // order.
  size_t leaf_size = default_leaf_size;
  // This is the spatial information
  Data2D data;
  // The internal node bounds.
  Data3D node_bounds;
  // These are the values obtained by fitting the gp.
  //std::vector<double> training_pivots;
  std::vector<double> responses;
  std::vector<NodeData> node_data;
  // Kernel and noise parameters. Maybe abstract away.
  double rbf_scale;
  double noise_scale;
  size_t nneighbours = 20;
  size_t nsamples() { return data.nsamples; }
  size_t nfeatures() { return data.nfeatures; }
  size_t nlevels() {
    // This is verbose on purpose so the debugger is easy to hook up
    size_t ratio{(nsamples() - 1) / leaf_size};
    if (ratio < 2) {
      return 2;
    } else {
      return size_t(std::log2(ratio) + 1);
    }
  }
  size_t nnodes() { return (1 << nlevels()) - 1; }

  // For serialization
  KDTree() {}

  KDTree(Data2D datain, std::vector<double> const &y, double rbf_scale = 0.1,
         double noise_scale = 1e-7,
         size_t leaf_size = default_leaf_size)
      : leaf_size(leaf_size), data(std::move(datain)),
        node_bounds(2, nnodes(), datain.nfeatures), rbf_scale(rbf_scale),
        noise_scale(noise_scale) {

    assert(y.size() == data.nsamples);

    // Initialize node bounds to infinity
    auto halfbound = static_cast<long>(nnodes() * nfeatures());
    auto begin = std::begin(node_bounds.data);
    std::fill(begin, begin + halfbound, inf);
    std::fill(begin + halfbound, begin + 2 * halfbound, neg_inf);
    node_data.assign(nnodes(), NodeData());

    // Fill indexes
    auto indexes = std::vector<size_t>(nsamples());
    std::iota(indexes.begin(), indexes.end(), 0);

    // Build tree
    recursive_build(indexes, 0, 0, nsamples());

    // TODO: Take data by reference and avoid copying.
    data = apply_permutation(data, indexes);
    // Train
    responses = apply_permutation(y, indexes);
	assert(responses.size() == nsamples());
    // training_pivots =
    //    compute_training_pivots(data, yperm, rbf_scale, noise_scale);
    //compute_training_pivots2(yperm);
  }

  void recursive_build(std::vector<size_t> &indexes, size_t inode, size_t start,
                       size_t end) {
    auto npoints = end - start;
    auto nmid = npoints / 2;
    init_node(indexes, inode, start, end);
    if (2 * inode + 1 >= nnodes()) {
      node_data[inode].is_leaf = true;
    } else {
      node_data[inode].is_leaf = false;
      auto i_max = find_node_split_dim(indexes, start, end);
      IndexComparator comp{data, i_max};
      std::nth_element(&indexes[start], &indexes[start + nmid], &indexes[end],
                       comp);
      recursive_build(indexes, 2 * inode + 1, start, start + nmid);
      recursive_build(indexes, 2 * inode + 2, start + nmid, end);
    }
  }

  /** Because there is no easy way of slicing an std::vector, we take a a pair
   * of indices. This is no worse than a pair of iterators and hides the ugly
   * types.**/
  size_t find_node_split_dim(std::vector<size_t> &indexes, size_t start_id,
                             size_t end_id) {

    double maxdelta = neg_inf;
    size_t max_split_dim = 0;
    for (size_t j = 0; j < nfeatures(); j++) {
      double maxval = neg_inf;
      double minval = inf;
      for (auto ind_in_index_array = start_id; ind_in_index_array < end_id;
           ++ind_in_index_array) {
        auto data_ind = indexes[ind_in_index_array];
        auto val = data.at(data_ind, j);
        if (val < minval) {
          minval = val;
        }
        if (val > maxval) {
          maxval = val;
        }
      }
      auto delta = maxval - minval;
      if (delta > maxdelta) {
        maxdelta = delta;
        max_split_dim = j;
      }
    }
    return max_split_dim;
  }

  void init_node(std::vector<size_t> &indexes, size_t inode, size_t idx_start,
                 size_t idx_end) {
    // double training_sum = 0;
    // double min_pivot = inf;
    // double max_pivot = neg_inf;
    for (auto i = idx_start; i < idx_end; i++) {
      auto data_index = indexes[i];
      // auto pivot = training_pivots[data_index];
      // training_sum += pivot;
      // if (pivot > max_pivot) {
      //  max_pivot = pivot;
      //}
      // if (pivot < min_pivot) {
      //  min_pivot = pivot;
      //}
      for (size_t j = 0; j < nfeatures(); j++) {
        auto data_val = data.at(data_index, j);
        auto &lowbound = node_bounds.at(0, inode, j);
        lowbound = std::min(lowbound, data_val);
        auto &highbound = node_bounds.at(1, inode, j);
        highbound = std::max(highbound, data_val);
      }
    }
    double rad = 0;
    for (size_t j = 0; j < nfeatures(); j++) {
      rad += std::pow(
          0.5 * (node_bounds.at(1, inode, j) - node_bounds.at(0, inode, j)), 2);
    }
    rad = std::sqrt(rad);
    node_data[inode].radius = rad;
    // node_data[inode].training_sum = training_sum;
    // node_data[inode].training_min = min_pivot;
    // node_data[inode].training_max = max_pivot;
    node_data[inode].start = idx_start;
    node_data[inode].end = idx_end;
  }

  std::string print_tree() {
    if (node_data.empty()) {
      return "";
    }
    std::stringstream s{};
    // Need all this stuff to make it recursive
    std::function<void(std::stringstream &, size_t, size_t)> add_info_from_node;
    add_info_from_node = [&](std::stringstream &stream, size_t inode,
                             size_t tab) {
      auto &node = node_data[inode];
      auto spaces = std::string(tab, ' ');
      if (node.is_leaf) {
        stream << spaces << "Leaf node " << inode << " containing "
               << node.start << "-" << node.end << "\n";
      } else {
        stream << spaces << "Parent node " << inode << "\n";
        add_info_from_node(stream, 2 * inode + 1, tab + 1);
        add_info_from_node(stream, 2 * inode + 2, tab + 1);
      }
    };
    add_info_from_node(s, 0, 0);
    return s.str();
  }

  double min_rdist(size_t inode, const point_type &pt) {
    double rdist = 0;
    for (size_t j = 0; j < nfeatures(); j++) {
      // lo pt hi -> 0 inside for that dimenstion
      // pt lo hi -> lo - pt
      // lo hi pt -> pt - hi
      auto lo = node_bounds(0, inode, j) - pt.at(j);
      auto hi = pt.at(j) - node_bounds(1, inode, j);
      auto d = std::max(lo, 0.) + std::max(hi, 0.);
      rdist += d * d;
    }
    return rdist;
  }

  double max_rdist(size_t inode, const point_type &pt) {
    double rdist = 0;
    for (size_t j = 0; j < nfeatures(); j++) {
      auto lo = std::abs(pt.at(j) - node_bounds(0, inode, j));
      auto hi = std::abs(pt.at(j) - node_bounds(1, inode, j));
      auto max = std::max(lo, hi);
      rdist += max * max;
    }
    return rdist;
  }

  double rbf(double rdist) { return basic_rbf(rdist, rbf_scale); }

  interpolation_storage& compute_data_for_point_interpolation(const point_type & pt){
	auto neigh = query(pt, nneighbours);
	return compute_data_for_point_interpolation(neigh);;
  }

  interpolation_storage& compute_data_for_point_interpolation(std::vector<DistanceIndex> const & neigh){
	thread_local auto internal_storage = interpolation_storage(nneighbours);
	auto  bstart = internal_storage.bstart;
	auto  ystart = internal_storage.ystart;
	auto wstart = internal_storage.wstart;
	auto  covmatstart = internal_storage.covmatstart;
	std::transform(std::begin(neigh), std::end(neigh), bstart, [&](auto &n){return rbf(n.rdistance);});
	//We need this copy because one of the two sets of weights is going to be mutated to store the solution
	//of the system, and we need it to estimate the error later.
	std::copy(bstart, bstart+nneighbours, wstart);
	//Note: we could have some approximate nearest neighbour and cache this part
	std::transform(std::begin(neigh), std::end(neigh), ystart, [&](auto &n){return responses[n.index];});
	fill_uplo_packed(covmatstart, nneighbours,
        [&](size_t i, size_t j)
		{return rbf(reduced_distance(data, neigh[i].index, neigh[j].index)) + int(i == j) * noise_scale * noise_scale;});
	char U = 'U';
	int one = 1;
	int intneighbours = nneighbours;
	int info;
	dppsv_(&U, intneighbours, one, covmatstart, bstart, intneighbours, &info);
	assert(info==0);
	return internal_storage;
  }

  double get_central_prediction(interpolation_storage & store){
	return std::inner_product(store.bstart, store.bstart+nneighbours, store.ystart, 0.);
  }


  double get_variance(interpolation_storage & store){
	  return rbf(0) - std::inner_product(store.bstart, store.bstart+nneighbours, store.wstart, 0.);
  }

  double interpolate_single(const point_type &pt) {
	auto store = compute_data_for_point_interpolation(pt);
	return get_central_prediction(store);
  }

  struct interpolation_result interpolate_single_result(const point_type &pt){
	  auto store = compute_data_for_point_interpolation(pt);
	  auto central = get_central_prediction(store);
	  auto variance = get_variance(store);
	  bool interpolable = (min_rdist(0, pt) == 0);
	  return {central, variance, interpolable};
  }


  std::vector<DistanceIndex> query(const point_type &pt, size_t k) {
    auto heap = std::vector<DistanceIndex>{};
    query_single_depthfirst(pt, k, 0, 0., heap);
    // std::sort_heap(heap.begin(), heap.end());
    return heap;
  }

  void query_single_depthfirst(const point_type &pt, size_t k, size_t inode,
                               double minrdist,
                               std::vector<DistanceIndex> &heap) {
    if (!heap.empty() && minrdist > heap.front().rdistance) {
      return;
    }
    auto &ndt = node_data[inode];
    if (ndt.is_leaf) {
      for (size_t idx = ndt.start; idx < ndt.end; idx++) {
        // auto data_index = indexes[idx];
        auto data_index = idx;
        auto rdist = reduced_distance(data, data_index, pt);
        max_k_heap_push({rdist, data_index}, heap, k);
      }

    } else {
      auto child1 = 2 * inode + 1;
      auto child2 = 2 * inode + 2;
      auto d1 = min_rdist(child1, pt);
      auto d2 = min_rdist(child2, pt);
      if (d1 <= d2) {
        query_single_depthfirst(pt, k, child1, d1, heap);
        query_single_depthfirst(pt, k, child2, d2, heap);
      } else {
        query_single_depthfirst(pt, k, child2, d2, heap);
        query_single_depthfirst(pt, k, child1, d1, heap);
      }
    }
  }

  size_t getLeafSize() { return leaf_size; }
  const Data3D &getNodeBounds() { return node_bounds; }

  template <class Archive> void serialize(Archive &ar) {
    // Note we do not save the data and responses on purpose:
    // We save them separatedly as they are likely to remain
    // stable much longer than the tree. If an upgrade of
    // this code causes a change in the tree, we can try to
    // recompute it.
	//
	// clang-format off
    ar(leaf_size,
       // data,
       node_bounds,
       // responses,
       node_data,
	   rbf_scale,
	   noise_scale,
	   nneighbours);
  }
};

std::variant<KDTree, std::string> load_tree(const char *filename) {
  auto res = KDTree();
  std::ifstream is(filename, std::ios::binary);
  if (!is.good()) {
    return std::string("Could not open archive: ") + filename;
  }
  cereal::PortableBinaryInputArchive ar(is);
  std::uint32_t magic_test;
  ar(magic_test);
  if (magic_test != magic) {
    return std::string("Corrupted archive: ") + filename +
           ". Magic numbers do not match";
  }
  std::uint32_t version_test;
  ar(version_test);
  if (version_test != tree_data_version) {
    return std::string("Mismatch between versions.");
  }
  auto dt = Data2D{};
  ar(dt);
  std::vector<double> y;
  ar(y);
  ar(res);
  res.data = std::move(dt);
  res.responses = std::move(y);
  return res;
}

//Note the version parameter is mostly for testing
void save_tree(KDTree &tree, const char *filename, std::uint32_t version=tree_data_version) {
  std::ofstream os(filename, std::ios::binary);
  cereal::PortableBinaryOutputArchive ar(os);
  ar(magic);
  ar(version);
  ar(tree.data);
  ar(tree.responses);
  ar(tree);
}

} // namespace ZKDTree
