#ifndef CUDIJKSTRA_KERNEL_HPP
#define CUDIJKSTRA_KERNEL_HPP

#include "matrix.hpp"
#include "segment.hpp"
#include <optional>

namespace cudijkstra {
struct result {
  struct matrix_pt {
    double distance;
    point pt;
  };
  std::vector<matrix_pt> distances;
  point farthest{};
  bounding_box contained{};
};

struct kernel {
  std::optional<result> operator()(double granularity, size_t core_cnt);

  polygon data{};
  bounding_box start_box{};
  point start{};
};
}

#endif // CUDIJKSTRA_KERNEL_HPP
