#ifndef CU_GEODESIC_SEGMENT_HPP
#define CU_GEODESIC_SEGMENT_HPP

#include <vector>

/**
 * \brief Namespace containing all code for this project.
 */
namespace cu_geodesic {
/**
 * \brief Structure representing a 2D point.
 */
struct point {
  /**
   * \brief The x coordinate.
   */
  double x;
  /**
   * \brief The y coordinate.
   */
  double y;
};

/**
 * \brief Structure representing a bounding box.
 */
struct bounding_box {
  /**
   * \brief Point representing the bottom-left corner (minimal x and y values).
   */
  point min;
  /**
   * \brief Point representing the top-right corner (maximal x and y values).
   */
  point max;
};

/**
 * \brief Structure representing a line segment.
 *
 * Note: the `a`, `b` notation disallows for certain checks regarding vertical
 * line segments; however, the points allow for vertical line segments to exist.
 */
struct segment {
  /**
   * \brief First endpoint for the segment.
   */
  point begin;
  /**
   * \brief Second endpoint for the segment.
   */
  point end;
  /**
   * \brief Slope of the carrier line.
   *
   * If the carrier line is represented as `y = ax + b`, this is the `a` value.
   */
  double a;
  /**
   * \brief Vertical offset of the carrier line.
   *
   * If the carrier line is represented as `y = ax + b`, this is the `b` value.
   */
  double b;
};

/**
 * \brief Structure representing an arbitrary polygon (with or without holes).
 */
struct polygon {
  /**
   * \brief Type alias for the boundary type.
   *
   * The boundary type is an std::vector of cu_geodesic::segment.
   */
  using boundary_t = std::vector<segment>;
  /**
   * \brief Type alias for the hole set type.
   *
   * The hole set type is an std::vector of std::vector of cu_geodesic::segment.
   */
  using holes_t = std::vector<std::vector<segment>>;
  /**
   * \brief The polygon's boundary.
   */
  boundary_t boundary;
  /**
   * \brief The polygon's holes.
   */
  holes_t holes;

  /**
   * \brief Gets the nth vertex in the polygon.
   *
   * The ordering is semi-arbitrary but consistent - first we enumerate over the
   * starting points of all boundary points in the order they're given, then we
   * do the same with the holes (in their respective order). If the polygon is
   * not modified, we can guarantee that calling this function with the same
   * values at different times will result in the same points. The polygon is
   * supposed to be a write-once structure, but this is not enforced.
   * The return value is undefined when reading out-of-bounds.
   *
   * \param idx The vertex index.
   * \returns The vertex at the given index.
   */
  [[nodiscard]] inline point vertex(std::size_t idx) const {
    if (idx < boundary.size())
      return boundary[idx].begin;
    idx -= boundary.size();
    for (const auto &hole : holes) {
      if (idx < hole.size())
        return hole[idx].begin;
      idx -= hole.size();
    }

    return {};
  }
};
} // namespace cu_geodesic

#endif // CU_GEODESIC_SEGMENT_HPP
