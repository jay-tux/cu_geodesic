#include <fstream>
#include "cuda_runtime.h"
#include "kernel.hpp"
#include "cuda_error.hpp"
#include <iostream>

using namespace cudijkstra;

 #define LIMITED_DEBUG

#ifdef LIMITED_DEBUG
template <typename ... Ts>
__device__ void debug(const Ts &... args) {
printf(args...);
}
#else
template <typename ... Ts>
__device__ void debug(Ts...) {}
#endif

__device__ const static double d_max = std::numeric_limits<double>::max();
__device__ const static double inf = std::numeric_limits<double>::infinity();
__device__ const static double reasonable_delta = 1e-3;

#define CUDA_SAFE_NO_RET(call) {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                   \
      cuda_log_error(err, #call, __FILE__, __LINE__);                          \
    } \
  }

#define CUDA_SAFE(call)                                                        \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                   \
      cuda_log_error(err, #call, __FILE__, __LINE__);                          \
      return std::nullopt; \
    } \
  }

#define CUDA_KERNEL(call) { call; cudaError_t err = cudaGetLastError(); if(err != cudaSuccess) {                                                  \
      cuda_log_error(err, #call, __FILE__, __LINE__);                                                                                             \
      return std::nullopt;\
    }                                                                          \
  }

__device__ inline bool same_point(const point &p1, const point &p2) {
  return p1.x == p2.x && p1.y == p2.y;
}

struct cu_res {
  double distance;
  point pt;
};

struct project_idx {
  point res;
  double dist;
};

struct intersection {
  bool is_intersection;
  point pt;
};

template <typename T>
struct cu_arr {
  size_t size;
  T *data;

  __host__ static cu_arr<T> alloc(size_t size) {
    cu_arr<T> res { .size = size, .data = nullptr };
    CUDA_SAFE_NO_RET(cudaMallocManaged(&res.data, sizeof(T) * size))
    return res;
  }

  __host__ static cu_arr<T> copy(size_t size, T *buf) {
    cu_arr<T> res = alloc(size);
    CUDA_SAFE_NO_RET(cudaMemcpy(res.data, buf, sizeof(T) * size, cudaMemcpyKind::cudaMemcpyDefault))
    return res;
  }

  __host__ void cleanup() {
    if(data != nullptr) {
      CUDA_SAFE_NO_RET(cudaFree(data))
      data = nullptr;
      size = 0;
    }
  }
};

struct cu_poly {
  enum struct status { OUTSIDE, POLYGON, HOLE, ON_EDGE };
  struct find_res {
    status status;
    size_t hole_idx;
  };

  [[nodiscard]] __device__ find_res status_for(const point &p) const;
  [[nodiscard]] __device__ bool intersects_any(const segment &s) const;

  cu_arr<segment> boundary;
  cu_arr<cu_arr<segment>> holes;
};

class cu_graph {
public:
  struct iterator {
    size_t idx;
    const size_t source;
    const cu_graph &graph;
    __device__ void operator++() {
      idx++; // increment first
      while(idx < graph.size()) {
        if(!isinf(graph.distance_between(source, idx)) && idx != source) {
          return;
        }
        idx++;
      }
    }
    __device__ operator bool() const {
      return idx < graph.size();
    }
  };

  __host__ cu_graph(const cu_poly *polygon) : polygon{polygon} {
    vertex_count = polygon->boundary.size;
    for(size_t idx = 0; idx < polygon->holes.size; idx++) {
      vertex_count += polygon->holes.data[idx].size;
    }
    // we are grossly over-allocating this, but let me be for now
    cudaMallocManaged(&distances, vertex_count * vertex_count * sizeof(double));
  }

  __host__ void clean() {
    if(distances != nullptr)
      cudaFree(distances);
    distances = nullptr;
  }

  __host__ const double *adjacencies() const { return distances; }

  __device__ const point &vertex(size_t idx) const {
    if(idx < polygon->boundary.size) {
      return polygon->boundary.data[idx].begin;
    }
    idx -= polygon->boundary.size;
    for(size_t hole = 0; hole < polygon->holes.size; hole++) {
      if(idx < polygon->holes.data[hole].size) {
        return polygon->holes.data[hole].data[idx].begin;
      }
      idx -= polygon->holes.data[hole].size;
    }
  }

  __device__ double &distance_between(size_t idx1, size_t idx2) {
    return distances[idx1 * vertex_count + idx2];
  }

  __device__ const double &distance_between(size_t idx1, size_t idx2) const {
    return distances[idx1 * vertex_count + idx2];
  }

  __device__ iterator neighbors_for(size_t idx) {
    size_t first = 0;
    while(first < size() && isinf(distance_between(idx, first))) first++;
    return {
        .idx = first,
        .source = idx,
        .graph = *this,
    };
  }

  __device__ const cu_poly &poly() const {
    return *polygon;
  }

  __host__ __device__ size_t size() const {
    return vertex_count;
  }

private:
  const cu_poly *polygon;
  size_t vertex_count;
  double *distances = nullptr;
};

class a_star_queue {
public:
  struct elem {
    [[nodiscard]] __device__ double estimate() const { return until_now + forward; }
    double until_now;
    double forward;
    size_t value;
  };

  __device__ bool empty() const {
    return _used == 0;
  }

  __device__ void enqueue(const elem &e) {
    if(_size + 1 >= _cap) {
      // assuming we'll be yeet'ing a bunch of elements
      node *upd = (node *)malloc((size_t)((float)_cap * 1.5) * sizeof(node));
      _cap = (size_t)((float)_cap * 1.5);
      size_t upd_size = 0;
      for(size_t i = 0; i < _size; i++) {
        if(!data[i].is_gravestone) upd[upd_size] = data[i];
        upd_size++;
      }

      free(data);
      data = upd;
      _size = upd_size;
    }

    data[_size] = { .val = e, .is_gravestone = false };
    _size++;
    _used++;
  }

  __device__ elem dequeue() {
    double lowest = d_max;
    size_t lowest_idx = 0;
    for(size_t i = 0; i < _size; i++) {
      if(!data[i].is_gravestone && data[i].val.estimate() < lowest) {
        lowest = data[i].val.estimate();
        lowest_idx = i;
      }
    }

    data[lowest_idx].is_gravestone = true;
    _used--;
    return data[lowest_idx].val;
  }

  __device__ void cleanup() {
    if(data != nullptr) {
      free(data);
      data = nullptr;
      _size = 0;
      _used = 0;
      _cap = 0;
    }
  }

  [[nodiscard]] __device__ size_t find(size_t v) const {
    for(size_t i = 0; i < _size; i++) {
      if(!data[i].is_gravestone && v == data[i].val.value) {
        return i;
      }
    }
    return _size;
  }

  __device__ elem &get(size_t idx) {
    return data[idx].val;
  }

  [[nodiscard]] __device__ size_t size() const { return _size; }

private:
  struct node {
    elem val;
    bool is_gravestone;
  };

  node *data = (node *)malloc(64 * sizeof(node));
  size_t _size = 0;
  size_t _used = 0;
  size_t _cap = 64;
};

__device__ inline bool between(double b1, double x, double b2) {
  return (b1 < b2) ? (b1 <= x && x <= b2) : (b1 >= x && x >= b2);
}

__device__ inline bool point_same(const point &p1, const point &p2) {
  return p1.x == p2.x && p1.y == p2.y;
}

__device__ double distance(point p1, point p2) {
  double tmp1 = p1.x - p2.x;
  double tmp2 = p1.y - p2.y;
  return sqrt(tmp1 * tmp1 + tmp2 * tmp2);
}

__device__ project_idx project_onto(point p, const segment &seg) {
  // adapted from
  // https://stackoverflow.com/questions/47481774/getting-point-on-line-segment-that-is-closest-to-another-point
  point diff = {.x = seg.end.x - seg.begin.x, .y = seg.end.y - seg.begin.y};
  point to_p = {.x = p.x - seg.begin.x, .y = p.y - seg.begin.y};
  double len = diff.x * diff.x + diff.y * diff.y;
  double interp = (to_p.x * diff.x + to_p.y * diff.y) / len;
  interp = (interp < 0) ? 0 : ((interp > 1) ? 1 : interp); // clamp
  point project = {.x = seg.begin.x + interp * diff.x,
                   .y = seg.begin.y + interp * diff.y};
  double dist = distance(p, project);
  return {.res = project, .dist = dist};
}

__device__ point project_onto_closest(point p, const cu_arr<segment> &segments) {
  double closest = d_max;
  point res{};
  for(size_t i = 0; i < segments.size; i++) {
    auto pr = project_onto(p, segments.data[i]);
    if(pr.dist < closest) {
      closest = pr.dist;
      res = pr.res;
    }
  }
  return res;
}

__device__ intersection intersects(const segment &s1, const segment &s2) {
  // check for common end points
  if(point_same(s1.begin, s2.begin) || point_same(s1.begin, s2.end)
      || point_same(s1.end, s2.begin) || point_same(s1.end, s2.end)) {
    return {.is_intersection = false, .pt = {}};
  }
  double cross_x = (s2.b - s1.b) / (s1.a - s2.a);
  bool on_s1 = between(s1.begin.x, cross_x, s1.end.x);
  bool on_s2 = between(s2.begin.x, cross_x, s2.end.x);
  if(on_s1 && on_s2) {
    point cross{
      .x = cross_x,
      .y = s1.a * cross_x + s1.b
    };
    return {.is_intersection = true,
            .pt = cross};
  }
  else {
    return {.is_intersection = false, .pt = {}};
  }
}

__device__ bool inside_single(point p, const cu_arr<segment> &polygon) {
  // adapted from https://en.wikipedia.org/wiki/Even%E2%80%93odd_rule
  bool res = false;
  point prev = polygon.data[polygon.size - 1].begin;
  for (size_t i = 0; i < polygon.size; i++) {
    point curr = polygon.data[i].begin;
    if (p.x == curr.x && p.y == curr.y) {
      return true;
    }
    if ((curr.y > p.y) != (prev.y > p.y)) {
      double slope = (p.x - curr.x) * (prev.y - curr.y) -
                     (prev.x - curr.x) * (p.y - curr.y);
      if ((slope < 0) != (prev.y < curr.y)) {
        res = !res;
      }
    }
    prev = curr;
  }
  return res;
}

__device__ bool point_on(const point &p, const segment &s) {
  if(!between(s.begin.x, p.x, s.end.x)) return false;
  double expected_y = s.a * p.x + s.b;
  return abs(expected_y - p.y) <= 1e-9;
}

__device__ cu_poly::find_res cu_poly::status_for(const point &p) const {
  for(size_t i = 0; i < boundary.size; i++) {
    if(point_on(p, boundary.data[i])) return { .status = status::ON_EDGE, .hole_idx = 0 };
  }

  for(size_t i = 0; i < holes.size; i++) {
    for(size_t j = 0; j < holes.data[i].size; j++) {
      if(point_on(p, holes.data[i].data[j])) return { .status = status::ON_EDGE, .hole_idx = 0 };
    }
  }

  if(!inside_single(p, boundary)) { return { .status = status::OUTSIDE, .hole_idx = 0 }; }

  for(size_t i = 0; i < holes.size; i++) {
    if(inside_single(p, holes.data[i])) {
      return { .status = status::HOLE, .hole_idx = i };
    }
  }

  return { .status = status::POLYGON, .hole_idx = 0 };
}

__device__ bool cu_poly::intersects_any(const cudijkstra::segment &s) const {
  for(size_t i = 0; i < boundary.size; i++) {
    if(intersects(s, boundary.data[i]).is_intersection) return true;
  }

  for(size_t i = 0; i < holes.size; i++) {
    for(size_t j = 0; j < holes.data[i].size; j++) {
      if(intersects(s, holes.data[i].data[j]).is_intersection) return true;
    }
  }

  return false;
}

__device__ segment compute_segment(const point &p1, const point &p2) {
  double tmp = (p2.y - p1.y) / (p2.x - p1.x);
  return {
    .begin = p1,
    .end = p2,
    .a = tmp,
    .b = p1.y - tmp * p1.x
  };
}

__device__ point interpolate_from_begin(const segment &s, double delta, bool print = false) {
  if(abs(s.begin.x - s.end.x) < 1e-6) {
    // almost-vertical case; "rotate" 90 degrees, interpolate
    double a = (s.end.x - s.begin.x) / (s.end.y - s.begin.y);
    double b = s.begin.x - a * s.begin.y;
    // correct interpolation direction
    double mod = s.begin.y < s.end.y ? delta : -delta;
    return {
      .x = a * (s.begin.y + mod) + b,
      .y = s.begin.y + mod
    };
  }
  else if(s.begin.x < s.end.x) {
    return {
      .x = s.begin.x + delta,
      .y = s.a * (s.begin.x + delta) + s.b
    };
  }
  else if(s.begin.x > s.end.x) {
    return {
      .x = s.begin.x - delta,
      .y = s.a * (s.begin.x - delta) + s.b
    };
  }
}

__device__ bool edge_viable(segment s, const cu_graph &graph, bool print = false) {
  point interpolate = interpolate_from_begin(s, reasonable_delta, print);
  auto inside = graph.poly().status_for(interpolate);
  // check if edge can be entirely within polygon
  if(inside.status == cu_poly::status::ON_EDGE) {
    // edge is an edge of the polygon
    return true;
  }
  else if(inside.status == cu_poly::status::POLYGON) {
    // edge starts inside polygon
    // check for any intersection
    if(!graph.poly().intersects_any(s)) {
      // no intersections, add distance
      return true;
    }
  }

  return false;
}

__global__ void compute_graph(cu_graph graph, size_t idx_offset) {
  uint idx = threadIdx.x + blockDim.x * blockIdx.x + idx_offset;
  if(idx >= graph.size()) return; // culling

  graph.distance_between(idx, idx) = 0.0; // should be obvious
  point start = graph.vertex(idx);
  for(size_t other = idx + 1; other < graph.size(); other++) {
    point partner = graph.vertex(other);
    segment seg = compute_segment(start, partner);

    if(edge_viable(seg, graph, true)) {
      double dist = distance(start, partner);
      graph.distance_between(idx, other) = dist;
      graph.distance_between(other, idx) = dist;
    }
    else {
      // set distance to infinity to make sure
      graph.distance_between(idx, other) = inf;
      graph.distance_between(other, idx) = inf;
    }
  }
}

__global__ void compute_geodesic_distance(uint max_idx, point start, cu_arr<double> res, cu_graph graph, size_t idx_offset) {
  uint idx = threadIdx.x + blockDim.x * blockIdx.x + idx_offset;
  if(idx >= max_idx) return; // culling

  // check viability of point, project if necessary
  // points are always viable as they are graph vertices
  point target = graph.vertex(idx);

  // check for direct route
  if(edge_viable(compute_segment(start, target), graph)) {
    res.data[idx] = distance(start, target);
    return; // done
  }

  a_star_queue queue;
  for(size_t i = 0; i < graph.size(); i++) {
    auto v = graph.vertex(i);
    auto e = compute_segment(v, start);
    if(edge_viable(e, graph)) {
      double d = distance(v, start);
      double f = distance(v, target);
      queue.enqueue({
          .until_now = d,
          .forward = f,
          .value = i
      });
    }
  }

  // A*
  while(!queue.empty()) {
    auto next = queue.dequeue();
    auto v = graph.vertex(next.value);
    if(next.forward == 0.0) {
      // found end: distance squared from any point to target = 0? point = target
      res.data[idx] = next.until_now;
      break; // done!
    }
    auto iter = graph.neighbors_for(next.value);
    while(iter) {
      auto v_ = graph.vertex(iter.idx);
      double tentative = next.until_now + graph.distance_between(next.value, iter.idx);
      size_t queue_idx = queue.find(iter.idx);
      if(queue_idx == queue.size()) {
        // not yet in queue
        queue.enqueue({
            .until_now = tentative,
            .forward = distance(graph.vertex(iter.idx), target),
            .value = iter.idx
        });
      }
      else {
        // in queue
        auto &elem = queue.get(queue_idx);
        auto _v_ = graph.vertex(elem.value);
        if(tentative < elem.until_now) {
          elem.until_now = tentative;
        }
      }

      ++iter;
    }
  }

  // clean up
  queue.cleanup();
}

__global__ void compute_to_point(uint max_idx, point start, cu_graph graph, point min, size_t per_row, double granularity, cu_arr<double> point_dist, cu_res *res, size_t idx_offset) {
  uint idx = threadIdx.x + blockDim.x * blockIdx.x + idx_offset;
  if(idx > max_idx) return; // out of range

  // compute target
  uint row_idx = idx / per_row;
  uint col_idx = idx % per_row;
  point target {
      .x = col_idx * granularity + min.x,
      .y = row_idx * granularity + min.y
  };
  res[idx].pt = target;
  auto in = graph.poly().status_for(target);
  if(in.status != cu_poly::status::POLYGON && in.status != cu_poly::status::ON_EDGE) {
    res[idx].distance = -1.0; // point is out of bounds
    return;
  }

  // check straight edge
  segment straight = compute_segment(start, target);
  if(edge_viable(straight, graph)) {
    res[idx].distance = distance(start, target);
    return;
  }

  // check each other vertex
  double min_dist = inf;
  for(size_t vert = 0; vert < graph.size(); vert++) {
    point vertex = graph.vertex(vert);
    segment between = compute_segment(vertex, target);
    if(edge_viable(between, graph)) {
      // Euclidean distance + backward distance from vertex
      double dist = distance(vertex, target) + point_dist.data[vert];
      if(dist < min_dist) {
        min_dist = dist;
      }
    }
  }

  // no path?
  res[idx].distance = (isinf(min_dist)) ? -1.0 : min_dist;
}

__host__ std::optional<result> kernel::operator()(double granularity, size_t core_cnt) {
  // alloc buffers & copy data
  std::cout << "[KERNEL]: allocating GPU memory and copying data..." << std::endl;
  cu_poly *polygon;
  cu_poly _polygon {
    .boundary = cu_arr<segment>::copy(data.boundary.size(), data.boundary.data()),
    .holes = cu_arr<cu_arr<segment>>::alloc(data.holes.size())
  };
  for(size_t i = 0; i < data.holes.size(); i++) {
    _polygon.holes.data[i] = cu_arr<segment>::copy(data.holes[i].size(), data.holes[i].data());
  }
  CUDA_SAFE(cudaMallocManaged(&polygon, sizeof(cu_poly)))
  CUDA_SAFE(cudaMemcpy(polygon, &_polygon, sizeof(cu_poly), cudaMemcpyKind::cudaMemcpyDefault))
  cu_graph graph(polygon);

  std::cout << "[KERNEL]: step one: starting GPU graph generation (attempting to use " << graph.size() << " CUDA threads)..." << std::endl;
  for(size_t i = 0; i < graph.size(); i += core_cnt) {
    CUDA_KERNEL((compute_graph<<<1, core_cnt>>>(graph, i)))
    CUDA_SAFE(cudaDeviceSynchronize()) // wait till cores are done
  }
  std::cout << "[KERNEL]: step one finished. GPU graph is generated." << std::endl;

  std::cout << "[KERNEL]: outputting GPU graph adjacencies..." << std::endl;
  {
    // sub scope for cleanup
    std::vector<double> adj(graph.size() * graph.size());
    CUDA_SAFE(cudaMemcpy(adj.data(), graph.adjacencies(),
                         graph.size() * graph.size() * sizeof(double),
                         cudaMemcpyKind::cudaMemcpyDefault))
    std::ofstream adj_dump("/tmp/adjacencies.txt");
    if (adj_dump.is_open()) {
      adj_dump << "from\\to";
      for (size_t i = 0; i < graph.size(); i++) {
        point p = data.vertex(i);
        adj_dump << "\t(" << p.x << "," << p.y << ")";
      }
      adj_dump << std::endl;
      for (size_t i = 0; i < graph.size(); i++) {
        point p2 = data.vertex(i);
        adj_dump << "(" << p2.x << "," << p2.y << ")";
        for (size_t j = 0; j < graph.size(); j++) {
          adj_dump << "\t" << adj[i * graph.size() + j];
        }
        adj_dump << std::endl;
      }
    }
  }
  std::cout << "[KERNEL]: GPU graph adjacencies outputted to /tmp/adjacencies.txt." << std::endl;

  // new second step: compute distance from start to each graph vertex
  std::cout << "[KERNEL]: step two: calculate geodesic distances from each vertex to the start..." << std::endl;
  cu_arr<double> to_vertices { .size = graph.size(), .data = nullptr };
  CUDA_SAFE(cudaMallocManaged(&to_vertices.data, graph.size() * sizeof(double)))
  for(size_t i = 0; i < graph.size(); i += core_cnt) {
    CUDA_KERNEL((compute_geodesic_distance<<<1, core_cnt>>>(graph.size(), start, to_vertices, graph, i)))
    CUDA_SAFE(cudaDeviceSynchronize()) // wait till cores are done
  }
  std::cout << "[KERNEL]: step two finished." << std::endl;
  std::cout << "[KERNEL]: cleaning up graph..." << std::endl;
  graph.clean();
  std::cout << "[KERNEL]: outputting geodesic distances to vertices..." << std::endl;
  {
    // sub scope for cleanup
    std::ofstream strm("/tmp/geodesic.csv");
    if(strm.is_open()) {
      std::vector<double> tmp(graph.size());
      strm << "vertex x,vertex y,distance from start" << std::endl;
      CUDA_SAFE(cudaMemcpy(tmp.data(), to_vertices.data, graph.size() * sizeof(double), cudaMemcpyKind::cudaMemcpyDefault))
      for(size_t i = 0; i < tmp.size(); i++) {
        const auto p = data.vertex(i);
        strm << p.x << "," << p.y << "," << tmp[i] << std::endl;
      }
    }
  }
  std::cout << "[KERNEL]: geodesic distances outputted to /tmp/geodesic.csv." << std::endl;

  bounding_box box = start_box;
  std::cout << "[KERNEL]: step three: calculate geodesic distances to points..." << std::endl;
  auto required_x_steps = (size_t)std::ceil((start_box.max.x - start_box.min.x) / granularity);
  auto required_y_steps = (size_t)std::ceil((start_box.max.y - start_box.min.y) / granularity);
  cu_res *res;
  CUDA_SAFE(cudaMallocManaged(&res, required_x_steps * required_y_steps * sizeof(cu_res)))
  std::cout << "[KERNEL]:   requiring " << required_x_steps << " steps in x-dimension, and "
            << required_y_steps << " in y-dimension; " << required_x_steps*required_y_steps
            << " steps in total" << std::endl;

  size_t total_steps = required_x_steps * required_y_steps;
  result res_cpu{
      .distances = std::vector<result::matrix_pt>(total_steps),
      .farthest = {},
      .contained = start_box
  };
  // run kernel
  for(size_t i = 0; i < total_steps; i += core_cnt) {
    CUDA_KERNEL((compute_to_point<<<1, core_cnt>>>(
        total_steps, start, graph, start_box.min, required_x_steps,
        granularity, to_vertices, res, i
    )))
    CUDA_SAFE(cudaDeviceSynchronize()) // wait for threads to finish
    std::cout << "\r[KERNEL]:   ran " << i << "/" << total_steps << " (" << (int)((double)i / total_steps * 100) << "%) steps" << std::flush;
  }
  std::cout << std::endl << "[KERNEL]: step three finished." << std::endl;

  std::cout << "[KERNEL]: step four: copy results to CPU..." << std::endl;
  std::vector<cu_res> distances_cpu(total_steps);
  CUDA_SAFE(cudaMemcpy(distances_cpu.data(), res, distances_cpu.size() * sizeof(cu_res), cudaMemcpyKind::cudaMemcpyDefault))
  double farthest = 0.0;
  for(size_t i = 0; i < distances_cpu.size(); i++) {
    const auto &p = distances_cpu[i];
    res_cpu.distances[i] = {
        .distance = distances_cpu[i].distance,
        .pt = distances_cpu[i].pt
    };
    if(p.distance > farthest) {
      farthest = p.distance;
      res_cpu.farthest = p.pt;
    }

    if(i % 1000 == 0) {
      std::cout << "\r[KERNEL]:   copied/checked " << i << "/"
                << distances_cpu.size() << " ("
                << 100 * i / distances_cpu.size() << "%) values." << std::flush;
    }
  }
  res_cpu.contained = box;
  std::cout << "\n[KERNEL]: step four finished. Farthest point is (" << res_cpu.farthest.x
            << ", " << res_cpu.farthest.y << "); distance " << farthest << "." << std::endl;

  std::cout << "[KERNEL]: computations done running. Cleaning up..." << std::endl;
  to_vertices.cleanup();
  _polygon.boundary.cleanup();
  for(size_t i = 0; i < _polygon.holes.size; i++) {
    _polygon.holes.data[i].cleanup();
  }
  _polygon.holes.cleanup();
  CUDA_SAFE(cudaFree(polygon))
  CUDA_SAFE(cudaFree(res))

  std::cout << "[KERNEL]: kernel finished running." << std::endl;
  return res_cpu;
}
