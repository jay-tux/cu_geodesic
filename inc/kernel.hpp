#ifndef CU_GEODESIC_KERNEL_HPP
#define CU_GEODESIC_KERNEL_HPP

#include "segment.hpp"
#include <optional>

/**
 * \brief Namespace containing all code for this project.
 */
namespace cu_geodesic {
/**
 * \brief Structure representing the result of the kernel operation.
 */
struct result {
  /**
   * \brief Structure representing a point in the result space.
   */
  struct matrix_pt {
    /**
     * \brief The geodesic distance from the starting point to this point.
     */
    double distance;
    /**
     * \brief The point the distance is measured to.
     */
    point pt;
  };
  /**
   * \brief The set of result points.
   */
  std::vector<matrix_pt> distances;
  /**
   * \brief The point that has been computed to be the farthest away from the center.
   */
  point farthest{};
  /**
   * \brief The bounding box for the problem (polygon).
   */
  bounding_box contained{};
};

/**
 * \brief Wrapper struct around the GPU kernel.
 */
struct kernel {
  /**
   * \brief Runs the kernel.
   * \param granularity The granularity for the kernel (delta for both directions).
   * \param core_cnt The amount of CUDA cores we're allowed to use.
   * \returns The result (as a cu_geodesic::result), or std::nullopt (if something went wrong).
   */
  std::optional<result> operator()(double granularity, size_t core_cnt);

  /**
   * \brief The polygon we're working in.
   */
  polygon data{};
  /**
   * \brief The bounding box of the polygon.
   */
  bounding_box start_box{};
  /**
   * \brief The starting point.
   */
  point start{};
};
}

#endif // CU_GEODESIC_KERNEL_HPP
