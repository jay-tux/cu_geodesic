#ifndef CUDIJKSTRA_IO_HPP
#define CUDIJKSTRA_IO_HPP

#include <string>
#include <tuple>
#include "matrix.hpp"
#include "segment.hpp"

namespace cudijkstra {
struct loader {
  using res_t = std::tuple<polygon, point, bounding_box>;
  static res_t load(const std::string& filename);
};
}

#endif // CUDIJKSTRA_IO_HPP
