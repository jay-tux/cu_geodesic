#ifndef CUDIJKSTRA_SEGMENT_HPP
#define CUDIJKSTRA_SEGMENT_HPP

#include <vector>

namespace cudijkstra {
struct point {
  double x;
  double y;
};
struct bounding_box {
  point min;
  point max;
};
struct segment {
  point begin;
  point end;
  double a;
  double b;
};
struct polygon {
  using boundary_t = std::vector<segment>;
  using holes_t = std::vector<std::vector<segment>>;
  boundary_t boundary;
  holes_t holes;

  [[nodiscard]] inline point vertex(size_t idx) const {
    if(idx < boundary.size()) return boundary[idx].begin;
    idx -= boundary.size();
    for(const auto & hole : holes) {
      if(idx < hole.size()) return hole[idx].begin;
      idx -= hole.size();
    }

    return {};
  }
};
}

#endif // CUDIJKSTRA_SEGMENT_HPP
