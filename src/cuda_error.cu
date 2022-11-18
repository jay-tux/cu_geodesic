#include <iostream>
#include "cuda_runtime.h"
#include "cuda_error.hpp"

void cudijkstra::cuda_log_error(cudaError_t err, const char *call, const char *file, size_t line) {
  std::cerr << "[ERROR]: " << file << " on line " << line
            << "; while executing `" << call << "': " << cudaGetErrorString(err)
            << std::endl;
  cudaGetLastError(); // reset error to no error
}