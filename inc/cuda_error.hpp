#ifndef CU_GEODESIC_CUDA_ERROR_HPP
#define CU_GEODESIC_CUDA_ERROR_HPP

/**
 * \brief Namespace containing all code for this project.
 */
namespace cu_geodesic {
/**
 * \brief Helper function to log a CUDA error.
 *
 * Logs the error message from CUDA with line number and file name to stderr.
 *
 * \param err The CUDA error to log.
 * \param call The function that was called (used in conjunction with macros).
 * \param file The file in which the error occurred (used in conjunction with
 * macros).
 * \param line The line number at which the error occurred (used in
 * conjunction with macros).
 */
void cuda_log_error(cudaError_t err, const char *call, const char *file,
                    size_t line);
} // namespace cu_geodesic

#endif // CU_GEODESIC_CUDA_ERROR_HPP
