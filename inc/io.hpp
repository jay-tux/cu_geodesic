#ifndef CU_GEODESIC_IO_HPP
#define CU_GEODESIC_IO_HPP

#include <string>
#include <tuple>
#include "segment.hpp"

/**
 * \brief Namespace containing all code for this project.
 */
namespace cu_geodesic {
/**
 * \brief Static structure to function as the file loader.
 */
struct loader {
  /**
   * \brief Alias for the return type of the load operation.
   *
   * The return type is a tuple of 1) a polygon, 2) the starting point and 3)
   * the bounding box for the polygon.
   */
  using res_t = std::tuple<polygon, point, bounding_box>;

  /**
   * \brief Static function to load the data from a given file as input data.
   *
   * Will cause the program to exit with a nonzero exit code if the file cannot
   * be parsed (with error messages and warnings on stderr).
   *
   * \param filename The file to load.
   * \returns The data in the file, ready for processing.
   */
  static res_t load(const std::string& filename);
};
}

#endif // CU_GEODESIC_IO_HPP
