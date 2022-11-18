#ifndef CUDIJKSTRA_CUDA_ERROR_HPP
#define CUDIJKSTRA_CUDA_ERROR_HPP

namespace cudijkstra {
void cuda_log_error(cudaError_t err, const char *call, const char *file, size_t line);
}

#endif // CUDIJKSTRA_CUDA_ERROR_HPP
