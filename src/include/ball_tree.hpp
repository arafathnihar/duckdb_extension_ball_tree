#pragma once

// Ball tree over (latitude, longitude) points with the haversine metric.
//
// A port of scikit-learn's BallTree (sklearn/neighbors/_ball_tree.pyx.tp and
// _binary_tree.pxi.tp) specialised to haversine:
//   * the tree is an implicit complete binary tree: node i has children 2i+1 and 2i+2
//   * it is built once, in bulk; leaves hold between leaf_size and 2*leaf_size points
//   * each node stores a centroid and the radius of the smallest ball around it
//     (about that centroid) that contains all of its points
//   * queries prune with the ball bounds: min = max(0, d(q, c) - r), max = d(q, c) + r
//
// Points are [lat, lon] in radians. All distances here are angular (radians);
// multiply by the sphere radius to get a length.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace balltree {

constexpr double EARTH_RADIUS_KM = 6371.0088;
constexpr double PI = 3.14159265358979323846;

// Haversine "reduced distance" is sin^2(d / 2): monotonic in d and cheaper (no asin),
// so the hot loops compare reduced distances and only convert at the edges.
struct Haversine {
	static double RDist(const double *a, const double *b) {
		const double s0 = std::sin(0.5 * (a[0] - b[0]));
		const double s1 = std::sin(0.5 * (a[1] - b[1]));
		return s0 * s0 + std::cos(a[0]) * std::cos(b[0]) * s1 * s1;
	}
	static double RDistToDist(double rdist) {
		return 2.0 * std::asin(std::sqrt(std::min(rdist, 1.0)));
	}
	static double DistToRDist(double dist) {
		const double s = std::sin(0.5 * std::min(dist, PI));
		return s * s;
	}
	static double Dist(const double *a, const double *b) {
		return RDistToDist(RDist(a, b));
	}
};

class BallTree {
public:
	static constexpr int DIM = 2;

	// `data` is n_points rows of [lat, lon] in radians, row-major. It is copied.
	BallTree(const double *data, size_t n_points, size_t leaf_size = 40)
	    : data_(data, data + n_points * DIM), n_points_(n_points), leaf_size_(leaf_size) {
		if (n_points == 0) {
			throw std::invalid_argument("BallTree: no points");
		}
		if (leaf_size < 1) {
			throw std::invalid_argument("BallTree: leaf_size must be >= 1");
		}
		// Levels are chosen so leaves end up with between leaf_size and 2 * leaf_size points.
		const double ratio = std::max(1.0, static_cast<double>(n_points - 1) / static_cast<double>(leaf_size));
		n_levels_ = static_cast<size_t>(std::log2(ratio) + 1);
		n_nodes_ = (size_t(1) << n_levels_) - 1;

		idx_array_.resize(n_points);
		std::iota(idx_array_.begin(), idx_array_.end(), int64_t(0));
		nodes_.assign(n_nodes_, Node());
		centroids_.assign(n_nodes_ * DIM, 0.0);
		Build(0, 0, n_points);
	}

	size_t Size() const {
		return n_points_;
	}
	size_t LeafSize() const {
		return leaf_size_;
	}
	// Approximate heap footprint in bytes.
	size_t MemoryUsage() const {
		return data_.size() * sizeof(double) + idx_array_.size() * sizeof(int64_t) + nodes_.size() * sizeof(Node) +
		       centroids_.size() * sizeof(double);
	}

	// Flat byte encoding of the whole tree (points, ordering, node bounds), so it can be stored and read back
	// without rebuilding. Values are fixed-width and in native byte order; the buffer is meant for the same
	// kind of machine that wrote it.
	void Serialize(std::vector<uint8_t> &out) const {
		out.clear();
		out.reserve(SerializedHeaderSize() + data_.size() * 8 + idx_array_.size() * 8 + nodes_.size() * 32 +
		            centroids_.size() * 8);
		Put<uint32_t>(out, SERIAL_MAGIC);
		Put<uint32_t>(out, SERIAL_VERSION);
		Put<uint64_t>(out, n_points_);
		Put<uint64_t>(out, leaf_size_);
		Put<uint64_t>(out, n_levels_);
		Put<uint64_t>(out, n_nodes_);
		PutArray(out, data_.data(), data_.size() * sizeof(double));
		PutArray(out, idx_array_.data(), idx_array_.size() * sizeof(int64_t));
		for (const Node &node : nodes_) {
			Put<uint64_t>(out, node.idx_start);
			Put<uint64_t>(out, node.idx_end);
			Put<double>(out, node.radius);
			Put<uint64_t>(out, node.is_leaf ? 1 : 0);
		}
		PutArray(out, centroids_.data(), centroids_.size() * sizeof(double));
	}

	// Inverse of Serialize. Throws std::runtime_error if the buffer is not a well-formed tree.
	static BallTree Deserialize(const uint8_t *buf, size_t size) {
		Reader in {buf, size, 0};
		if (in.Get<uint32_t>() != SERIAL_MAGIC) {
			throw std::runtime_error("BallTree: not a serialized ball tree");
		}
		if (in.Get<uint32_t>() != SERIAL_VERSION) {
			throw std::runtime_error("BallTree: unsupported serialization version");
		}
		BallTree tree;
		tree.n_points_ = in.Get<uint64_t>();
		tree.leaf_size_ = in.Get<uint64_t>();
		tree.n_levels_ = in.Get<uint64_t>();
		tree.n_nodes_ = in.Get<uint64_t>();
		if (tree.n_points_ == 0 || tree.leaf_size_ == 0 || tree.n_levels_ == 0 || tree.n_levels_ > 40 ||
		    tree.n_nodes_ != (size_t(1) << tree.n_levels_) - 1 || tree.n_points_ > (size_t(1) << 40)) {
			throw std::runtime_error("BallTree: corrupt serialized header");
		}
		// check the size up front so a corrupt count cannot trigger a huge allocation
		const size_t need = tree.n_points_ * 16 + tree.n_points_ * 8 + tree.n_nodes_ * 32 + tree.n_nodes_ * 16;
		if (in.Remaining() < need) {
			throw std::runtime_error("BallTree: serialized data is truncated");
		}
		tree.data_.resize(tree.n_points_ * DIM);
		in.GetArray(tree.data_.data(), tree.data_.size() * sizeof(double));
		tree.idx_array_.resize(tree.n_points_);
		in.GetArray(tree.idx_array_.data(), tree.idx_array_.size() * sizeof(int64_t));
		for (int64_t id : tree.idx_array_) {
			if (id < 0 || static_cast<size_t>(id) >= tree.n_points_) {
				throw std::runtime_error("BallTree: corrupt point index");
			}
		}
		tree.nodes_.resize(tree.n_nodes_);
		for (Node &node : tree.nodes_) {
			node.idx_start = in.Get<uint64_t>();
			node.idx_end = in.Get<uint64_t>();
			node.radius = in.Get<double>();
			node.is_leaf = in.Get<uint64_t>() != 0;
			if (node.idx_start > node.idx_end || node.idx_end > tree.n_points_) {
				throw std::runtime_error("BallTree: corrupt node range");
			}
		}
		tree.centroids_.resize(tree.n_nodes_ * DIM);
		in.GetArray(tree.centroids_.data(), tree.centroids_.size() * sizeof(double));
		return tree;
	}
	const double *Point(int64_t i) const {
		return &data_[static_cast<size_t>(i) * DIM];
	}

	// Ids of all points within angular distance `r` of `pt`, unsorted. If `dists` is
	// non-null it receives the matching angular distances.
	void QueryRadius(const double *pt, double r, std::vector<int64_t> &ids, std::vector<double> *dists) const {
		ids.clear();
		if (dists) {
			dists->clear();
		}
		QueryRadiusNode(0, pt, r, Haversine::DistToRDist(r), ids, dists);
	}

	// The k points closest to `pt`, nearest first, as (id, angular distance). Includes the
	// point itself if it is in the tree.
	std::vector<std::pair<int64_t, double>> QueryKnn(const double *pt, size_t k) const {
		k = std::min(k, n_points_);
		std::vector<std::pair<double, int64_t>> heap; // max-heap on reduced distance
		heap.reserve(k);
		if (k > 0) {
			KnnNode(0, pt, k, 0.0, heap);
		}
		std::sort_heap(heap.begin(), heap.end());
		std::vector<std::pair<int64_t, double>> out;
		out.reserve(heap.size());
		for (auto &h : heap) {
			out.emplace_back(h.second, Haversine::RDistToDist(h.first));
		}
		return out;
	}

	// Radius count for every point in the tree, indexed by point id. Uses `n_threads`
	// worker threads (0 = hardware concurrency).
	std::vector<int64_t> CountRadiusAll(double r, size_t n_threads = 0) const {
		std::vector<int64_t> counts(n_points_);
		if (n_threads == 0) {
			n_threads = std::max(1u, std::thread::hardware_concurrency());
		}
		n_threads = std::min(n_threads, std::max<size_t>(1, n_points_ / 1024));
		const double rr = Haversine::DistToRDist(r);
		auto work = [&](size_t begin, size_t end) {
			for (size_t i = begin; i < end; i++) {
				const int64_t id = idx_array_[i];
				counts[id] = CountRadiusNode(0, Point(id), r, rr);
			}
		};
		if (n_threads <= 1) {
			work(0, n_points_);
			return counts;
		}
		std::vector<std::thread> threads;
		const size_t chunk = (n_points_ + n_threads - 1) / n_threads;
		for (size_t t = 0; t < n_threads; t++) {
			const size_t begin = t * chunk;
			const size_t end = std::min(n_points_, begin + chunk);
			if (begin < end) {
				threads.emplace_back(work, begin, end);
			}
		}
		for (auto &th : threads) {
			th.join();
		}
		return counts;
	}

private:
	BallTree() = default; // for Deserialize

	static constexpr uint32_t SERIAL_MAGIC = 0x42545245; // "BTRE"
	static constexpr uint32_t SERIAL_VERSION = 1;
	static constexpr size_t SerializedHeaderSize() {
		return 2 * sizeof(uint32_t) + 4 * sizeof(uint64_t);
	}

	template <class T>
	static void Put(std::vector<uint8_t> &out, T value) {
		PutArray(out, &value, sizeof(T));
	}
	static void PutArray(std::vector<uint8_t> &out, const void *src, size_t bytes) {
		const auto *p = static_cast<const uint8_t *>(src);
		out.insert(out.end(), p, p + bytes);
	}

	struct Reader {
		const uint8_t *buf;
		size_t size;
		size_t pos;
		size_t Remaining() const {
			return size - pos;
		}
		void GetArray(void *dst, size_t bytes) {
			if (bytes > Remaining()) {
				throw std::runtime_error("BallTree: serialized data is truncated");
			}
			if (bytes > 0) {
				std::memcpy(dst, buf + pos, bytes);
			}
			pos += bytes;
		}
		template <class T>
		T Get() {
			T value;
			GetArray(&value, sizeof(T));
			return value;
		}
	};

	struct Node {
		size_t idx_start = 0;
		size_t idx_end = 0;
		double radius = 0.0; // angular
		bool is_leaf = false;
	};

	const double *Centroid(size_t node) const {
		return &centroids_[node * DIM];
	}

	// Mean of the node's points (as plain [lat, lon] values) and the largest distance from it to any of them.
	void InitNode(size_t i_node, size_t idx_start, size_t idx_end) {
		double *centroid = &centroids_[i_node * DIM];
		centroid[0] = centroid[1] = 0.0;
		for (size_t i = idx_start; i < idx_end; i++) {
			const double *p = Point(idx_array_[i]);
			centroid[0] += p[0];
			centroid[1] += p[1];
		}
		const double n = static_cast<double>(idx_end - idx_start);
		centroid[0] /= n;
		centroid[1] /= n;

		double radius = 0.0;
		for (size_t i = idx_start; i < idx_end; i++) {
			radius = std::max(radius, Haversine::RDist(centroid, Point(idx_array_[i])));
		}
		Node &node = nodes_[i_node];
		node.radius = Haversine::RDistToDist(radius);
		node.idx_start = idx_start;
		node.idx_end = idx_end;
	}

	// Index (0 = lat, 1 = lon) of the dimension with the largest spread.
	size_t SplitDim(size_t idx_start, size_t idx_end) const {
		size_t best = 0;
		double best_spread = 0.0;
		for (size_t j = 0; j < DIM; j++) {
			double lo = Point(idx_array_[idx_start])[j];
			double hi = lo;
			for (size_t i = idx_start + 1; i < idx_end; i++) {
				const double v = Point(idx_array_[i])[j];
				lo = std::min(lo, v);
				hi = std::max(hi, v);
			}
			if (hi - lo > best_spread) {
				best_spread = hi - lo;
				best = j;
			}
		}
		return best;
	}

	void Build(size_t i_node, size_t idx_start, size_t idx_end) {
		InitNode(i_node, idx_start, idx_end);
		Node &node = nodes_[i_node];
		const size_t n = idx_end - idx_start;

		if (2 * i_node + 1 >= n_nodes_ || n < 2) {
			node.is_leaf = true;
			return;
		}
		node.is_leaf = false;
		// Median split on the widest dimension: everything left of n_mid is <= everything right of it.
		const size_t dim = SplitDim(idx_start, idx_end);
		const size_t n_mid = n / 2;
		std::nth_element(idx_array_.begin() + idx_start, idx_array_.begin() + idx_start + n_mid,
		                 idx_array_.begin() + idx_end,
		                 [&](int64_t a, int64_t b) { return Point(a)[dim] < Point(b)[dim]; });
		Build(2 * i_node + 1, idx_start, idx_start + n_mid);
		Build(2 * i_node + 2, idx_start + n_mid, idx_end);
	}

	// Lower and upper bounds on the distance from `pt` to any point in the node.
	void MinMaxDist(size_t i_node, const double *pt, double &lo, double &hi) const {
		const double d = Haversine::Dist(pt, Centroid(i_node));
		lo = std::max(0.0, d - nodes_[i_node].radius);
		hi = d + nodes_[i_node].radius;
	}

	int64_t CountRadiusNode(size_t i_node, const double *pt, double r, double rr) const {
		const Node &node = nodes_[i_node];
		double lo, hi;
		MinMaxDist(i_node, pt, lo, hi);
		if (lo > r) {
			return 0; // every point in the ball is too far
		}
		if (hi <= r) {
			return static_cast<int64_t>(node.idx_end - node.idx_start); // every point in the ball is close enough
		}
		if (node.is_leaf) {
			int64_t count = 0;
			for (size_t i = node.idx_start; i < node.idx_end; i++) {
				count += Haversine::RDist(pt, Point(idx_array_[i])) <= rr;
			}
			return count;
		}
		return CountRadiusNode(2 * i_node + 1, pt, r, rr) + CountRadiusNode(2 * i_node + 2, pt, r, rr);
	}

	void QueryRadiusNode(size_t i_node, const double *pt, double r, double rr, std::vector<int64_t> &ids,
	                     std::vector<double> *dists) const {
		const Node &node = nodes_[i_node];
		double lo, hi;
		MinMaxDist(i_node, pt, lo, hi);
		if (lo > r) {
			return;
		}
		if (hi <= r) {
			for (size_t i = node.idx_start; i < node.idx_end; i++) {
				ids.push_back(idx_array_[i]);
				if (dists) {
					dists->push_back(Haversine::Dist(pt, Point(idx_array_[i])));
				}
			}
			return;
		}
		if (node.is_leaf) {
			for (size_t i = node.idx_start; i < node.idx_end; i++) {
				const double rd = Haversine::RDist(pt, Point(idx_array_[i]));
				if (rd <= rr) {
					ids.push_back(idx_array_[i]);
					if (dists) {
						dists->push_back(Haversine::RDistToDist(rd));
					}
				}
			}
			return;
		}
		QueryRadiusNode(2 * i_node + 1, pt, r, rr, ids, dists);
		QueryRadiusNode(2 * i_node + 2, pt, r, rr, ids, dists);
	}

	// Depth-first k-nearest search. `rdist_lb` is a lower bound (reduced distance) on the distance from
	// pt to anything in this node. The heap holds the best k so far, largest reduced distance on top.
	void KnnNode(size_t i_node, const double *pt, size_t k, double rdist_lb,
	             std::vector<std::pair<double, int64_t>> &heap) const {
		if (heap.size() == k && rdist_lb > heap.front().first) {
			return; // cannot beat the current k-th best
		}
		const Node &node = nodes_[i_node];
		if (node.is_leaf) {
			for (size_t i = node.idx_start; i < node.idx_end; i++) {
				const int64_t id = idx_array_[i];
				const double rd = Haversine::RDist(pt, Point(id));
				if (heap.size() < k) {
					heap.emplace_back(rd, id);
					std::push_heap(heap.begin(), heap.end());
				} else if (rd < heap.front().first) {
					std::pop_heap(heap.begin(), heap.end());
					heap.back() = {rd, id};
					std::push_heap(heap.begin(), heap.end());
				}
			}
			return;
		}
		// Visit the closer child first so the bound tightens before the farther one is considered.
		const size_t c1 = 2 * i_node + 1, c2 = c1 + 1;
		double lo1, hi1, lo2, hi2;
		MinMaxDist(c1, pt, lo1, hi1);
		MinMaxDist(c2, pt, lo2, hi2);
		const double lb1 = Haversine::DistToRDist(lo1), lb2 = Haversine::DistToRDist(lo2);
		if (lb1 <= lb2) {
			KnnNode(c1, pt, k, lb1, heap);
			KnnNode(c2, pt, k, lb2, heap);
		} else {
			KnnNode(c2, pt, k, lb2, heap);
			KnnNode(c1, pt, k, lb1, heap);
		}
	}

	std::vector<double> data_;
	size_t n_points_;
	size_t leaf_size_;
	size_t n_levels_ = 0;
	size_t n_nodes_ = 0;
	std::vector<int64_t> idx_array_;
	std::vector<Node> nodes_;
	std::vector<double> centroids_;
};

} // namespace balltree
