#pragma once
#include <cuda_runtime.h>
#include <new>

struct pinned {
  static void *operator new(size_t n) {
    void *ptr = 0;
    cudaError_t result = cudaMallocHost(&ptr, n);
    if (cudaSuccess != result || 0 == ptr) throw std::bad_alloc();
    return ptr;
  }

  static void operator delete(void *ptr) noexcept { cudaFreeHost(ptr); }
};
