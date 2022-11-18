#include "cuda_error.hpp"
#include "cuda_runtime.h"
#include <iostream>

void cu_geodesic::cuda_log_error(cudaError_t err, const char *call,
                                 const char *file, size_t line) {
  std::cerr << "[ERROR]: " << file << " on line " << line
            << "; while executing `" << call << "': " << cudaGetErrorString(err)
            << std::endl;
  cudaGetLastError(); // reset error to no error
}