#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

namespace deepnet {
namespace cuda {

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "    \
                << cudaGetErrorString(error) << std::endl;                     \
      throw std::runtime_error(cudaGetErrorString(error));                     \
    }                                                                          \
  } while (0)

// Check if CUDA is available
inline bool is_cuda_available() {
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  return error == cudaSuccess && device_count > 0;
}

// Get CUDA device properties
inline void print_device_info() {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));

  std::cout << "CUDA Devices Found: " << device_count << std::endl;

  for (int i = 0; i < device_count; ++i) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

    std::cout << "Device " << i << ": " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor
              << std::endl;
    std::cout << "  Total Global Memory: "
              << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock
              << std::endl;
  }
}

// Set CUDA device
inline void set_device(int device_id) { CUDA_CHECK(cudaSetDevice(device_id)); }

// Synchronize device
inline void synchronize() { CUDA_CHECK(cudaDeviceSynchronize()); }

} // namespace cuda
} // namespace deepnet

#else // !USE_CUDA

namespace deepnet {
namespace cuda {

inline bool is_cuda_available() { return false; }
inline void print_device_info() {}
inline void set_device(int) {}
inline void synchronize() {}

} // namespace cuda
} // namespace deepnet

#endif // USE_CUDA
