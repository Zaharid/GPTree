#pragma once
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "gsl/span"

namespace ZKDTree {

using point_type = gsl::span<double>;

namespace {
constexpr auto inf = std::numeric_limits<double>::infinity();
constexpr auto neg_inf = -inf;
constexpr double pi = 3.141592653589793238463;
constexpr double sqrt2 = std::sqrt(2);
constexpr size_t default_leaf_size = 40;
} // namespace

struct NodeData {
  double radius;
  double training_sum;
  size_t start;
  size_t end;
  bool is_leaf;
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

  double &at(size_t i, size_t j) { return data[i * nfeatures + j]; }

  const double &at(size_t i, size_t j) const { return data[i * nfeatures + j]; }

  point_type at(size_t i) {
    return {&data.data()[i * nfeatures], static_cast<long>(nfeatures)};
  }

  double &operator()(size_t i, size_t j) { return at(i, j); }
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

  double &at(size_t k, size_t i, size_t j) {
    return data[k * nnodes * nfeatures + i * nfeatures + j];
  }

  double at(size_t k, size_t i, size_t j) const {
    return data[k * nnodes * nfeatures + i * nfeatures + j];
  }
  double &operator()(size_t k, size_t i, size_t j) { return at(k, i, j); }
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

/** Return a matrix encoded as Data2D where the upper triangle is zero and
 * the lower triangle is the result of evaluating the RBF kernel on the points
 * in data, with the scale parameter scale. For j<i we have:
 *
 * mat(i,j) = exp(-|xi - xj|²)/(2*scale²)
 *
 */
Data2D rbf_lo_matrix(Data2D const &data, double scale) {
  auto mat = Data2D(data.nsamples, data.nsamples);
  for (size_t i = 0; i < data.nsamples; i++) {
    for (size_t j = 0; j < data.nsamples; j++) {
      auto x = basic_rbf(reduced_distance(data, i, j), scale);
      mat(i, j) = x;
    }
  }
  return mat;
}

/** Add noise_scale² to the diagonal of data in place.
 */
void add_noise(Data2D &data, double noise_scale) {
  for (size_t i = 0; i < data.nsamples; i++) {
    data.at(i, i) += noise_scale * noise_scale;
  }
}

extern "C" int dpotrf_(const char *UPLO, const int &N, double *A,
                       const int &LDA, int &info);

extern "C" int dpotrs_(const char *UPLO, const int &N, const int &NRHS,
                       const double *A, const int &LDA, double *B,
                       const int &LDB, int &info);

extern "C" void dposv_(char *uplo, int *n, int *nrhs, double *a, int *lda,
                       double *b, int *ldb, int *info);

extern "C" void dgesv_(int *n, int *nrhs, double *a, int *lda, int *ipiv,
                       double *b, int *ldb, int *info);

// Probably only need the inplace version below.
Data2D lapack_cholesky_factor(const Data2D &in) {
  auto n = in.nsamples;
  auto copy = in.data;
  auto res = Data2D{n, n, std::move(copy)};
  int info;
  dpotrf_("L", n, res.data.data(), n, info);
  assert(info == 0);
  return res;
}

void lapack_cholesky_factor_inplace(Data2D &in) {
  auto n = in.nsamples;
  int info;
  dpotrf_("L", n, in.data.data(), n, info);
  assert(info == 0);
}

std::vector<double> lapack_cholesky_solve(Data2D const &chol,
                                          const std::vector<double> &b) {
  auto x = b;
  auto n = chol.nsamples;
  int info;
  dpotrs_("L", n, 1, chol.data.data(), n, x.data(), n, info);
  assert(info == 0);
  return x;
}

std::vector<double> compute_training_pivots(Data2D const &data,
                                            std::vector<double> const &y,
                                            double rbf_scale,
                                            double noise_scale) {
  auto mat = rbf_lo_matrix(data, rbf_scale);
  add_noise(mat, noise_scale);
  int n = mat.nsamples;
  auto res = y;
  int one = 1;
  int info;
  std::vector<int> throwaway(n);
  dgesv_(&n, &one, mat.data.data(), &n, throwaway.data(), res.data(), &n,
         &info);
  assert(info == 0);
  */
  lapack_cholesky_factor_inplace(mat);
  auto res = lapack_cholesky_solve(mat, y);

  return res;
}

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
  std::vector<double> training_pivots;
  std::vector<NodeData> node_data;
  std::vector<size_t> indexes;
  // Kernel and noise parameters. Maybe abstract away.
  double rbf_scale;
  double noise_scale;
  double search_threshold;
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

  KDTree(Data2D datain, std::vector<double> const &y, double rbf_scale = 0.1,
         double noise_scale = 1e-7, double search_threshold = 1e-5,
         size_t leaf_size = default_leaf_size)
      : leaf_size(leaf_size), data(std::move(datain)),
        node_bounds(2, nnodes(), datain.nfeatures), rbf_scale(rbf_scale),
        noise_scale(noise_scale), search_threshold(search_threshold) {

    assert(y.size() == data.nsamples);

    // Train
    training_pivots = compute_training_pivots(data, y, rbf_scale, noise_scale);

    // Initialize node bounds to infinity
    auto halfbound = static_cast<long>(nnodes() * nfeatures());
    auto begin = std::begin(node_bounds.data);
    std::fill(begin, begin + halfbound, inf);
    std::fill(begin + halfbound, begin + 2 * halfbound, neg_inf);
    node_data.assign(nnodes(), NodeData());

    // Fill indexes
    indexes = std::vector<size_t>(nsamples());
    std::iota(indexes.begin(), indexes.end(), 0);

    // Build tree
    recursive_build(0, 0, nsamples());
  }

  void recursive_build(size_t inode, size_t start, size_t end) {
    auto npoints = end - start;
    auto nmid = npoints / 2;
    init_node(inode, start, end);
    if (2 * inode + 1 >= nnodes()) {
      node_data[inode].is_leaf = true;
    } else {
      node_data[inode].is_leaf = false;
      auto i_max = find_node_split_dim(start, end);
      IndexComparator comp{data, i_max};
      std::nth_element(&indexes[start], &indexes[start + nmid], &indexes[end],
                       comp);
      recursive_build(2 * inode + 1, start, start + nmid);
      recursive_build(2 * inode + 2, start + nmid, end);
    }
  }

  /** Because there is no easy way of slicing an std::vector, we take a a pair
   * of indices. This is no worse than a pair of iterators and hides the ugly
   * types.**/
  size_t find_node_split_dim(size_t start_id, size_t end_id) {

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

  void init_node(size_t inode, size_t idx_start, size_t idx_end) {
    double training_sum = 0;
    for (auto i = idx_start; i < idx_end; i++) {
      auto data_index = indexes[i];
      training_sum += training_pivots[data_index];
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
    node_data[inode].training_sum = training_sum;
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
  double weight(size_t data_index, const point_type &pt) {
    return rbf(reduced_distance(data, data_index, pt));
  }

  double interpolate_single(const point_type &pt) {
    double wsofar = 0;
    auto res = accumulate_weight(0, pt, wsofar);
    // assert(std::all_of(training_pivots.begin(), training_pivots.end(), []
    // (auto x) {return std::isnan(x);}));
    return res;
  }

  double accumulate_weight(size_t inode, const point_type &pt, double &wsofar) {
    auto wmin = rbf(max_rdist(inode, pt));
    auto wmax = rbf(min_rdist(inode, pt));
    auto &ndt = node_data[inode];
    auto node_size = ndt.end - ndt.start;
    if (node_size * (wmax - wmin) <=
        2 * search_threshold * (wsofar + node_size * wmin)) {
      wsofar += wmin * node_size;

      std::cout << "inode:" << inode << "\n";
      std::cout << "0.5*(wmin + wmax)[" << 0.5 * (wmin + wmax)
                << "] * ndt.training_sum[" << ndt.training_sum << "]\n";
      auto wtest = rbf(reduced_distance(data, indexes[ndt.start], pt));
      assert(wtest < wmax);
      assert(wmin < wtest);

      auto wmean = 0.5 * (wmax + wmin);
      // assert(0);
      double debugsum = 0;
      double realsum = 0;
      for (auto index = ndt.start; index < ndt.end; index++) {
        auto data_index = indexes[index];

        auto wxx = rbf(reduced_distance(data, data_index, pt));
        wsofar += wmin;
        debugsum += wmean * training_pivots[data_index];
        realsum += wxx * training_pivots[data_index];
      }

      std::cout << "Realsum:" << realsum << ";\n";
      std::cout << "Debugsum:" << debugsum << ";\n";
      // assert(0);
      return debugsum;
      // return 0.5 * (wmin + wmax) * ndt.training_sum;
      // return  debugsum * ndt.training_sum;
    } else {
      if (ndt.is_leaf) {
        double res = 0;
        for (auto index = ndt.start; index < ndt.end; index++) {
          auto data_index = indexes[index];
          auto w = rbf(reduced_distance(data, data_index, pt));
          wsofar += w;
          res += training_pivots[data_index] * w;
          // training_pivots[data_index] = NAN;
        }
        return res;
      } else {
        auto child1 = 2 * inode + 1;
        auto child2 = 2 * inode + 2;
        auto d1 = min_rdist(child1, pt);
        auto d2 = min_rdist(child2, pt);

        double nearweight;
        double farweight;
        if (d1 < d2) {
          nearweight = accumulate_weight(child1, pt, wsofar);
          farweight = accumulate_weight(child2, pt, wsofar);
        } else {
          nearweight = accumulate_weight(child2, pt, wsofar);
          farweight = accumulate_weight(child1, pt, wsofar);
        }
        return nearweight + farweight;
      }
    }
  }

  double interpolate_single_bruteforce(const point_type &pt) {
    double res = 0;
    for (size_t i = 0; i < nsamples(); i++) {
      res += rbf(reduced_distance(data, i, pt)) * training_pivots[i];
    }
    return res;
  }

  size_t getLeafSize() { return leaf_size; }
  const Data3D &getNodeBounds() { return node_bounds; }
};

} // namespace ZKDTree
