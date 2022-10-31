#include "preprocess_struct.h"

#include <cassert>
#include <cuda_runtime.h>
#include "common/helper_cuda.h"

namespace hetuCTR{

void createPreprocessData(PreprocessData &pdata, size_t batch_size, size_t nrank) {
  assert(batch_size > 0);
  pdata.batch_size = 0;
  pdata.unique_size = 0;
  pdata.allocate_size = batch_size;
  checkCudaErrors(cudaMalloc(
    &pdata.d_idx, sizeof(index_t) * batch_size));
  checkCudaErrors(cudaMalloc(
    &pdata.d_unique_idx, sizeof(index_t) * batch_size));
  checkCudaErrors(cudaMalloc(
    &pdata.d_idx_map, sizeof(index_t) * batch_size));
  checkCudaErrors(cudaMalloc(
    &pdata.d_offset, sizeof(index_t) * batch_size));
  checkCudaErrors(cudaMalloc(
    &pdata.d_root, sizeof(worker_t) * batch_size));
  checkCudaErrors(cudaMalloc(
    &pdata.d_run_length, sizeof(index_t) * (batch_size + 1)));
  checkCudaErrors(cudaMalloc(
    &pdata.d_sorted_arg, sizeof(index_t) * batch_size));
  checkCudaErrors(cudaMalloc(
    &pdata.u_shape, sizeof(size_t) * (nrank + 1)));
  checkCudaErrors(cudaMalloc(
    &pdata.u_shape_exchanged, sizeof(size_t) * (nrank + 1)));
  checkCudaErrors(cudaMallocHost(
    &pdata.h_shape, sizeof(size_t) * (nrank + 1)));
  checkCudaErrors(cudaMallocHost(
    &pdata.h_shape_exchanged, sizeof(size_t) * (nrank + 1)));
}

void freePreprocessData(PreprocessData &pdata) {
  checkCudaErrors(cudaFree(pdata.d_idx));
  checkCudaErrors(cudaFree(pdata.d_unique_idx));
  checkCudaErrors(cudaFree(pdata.d_idx_map));
  checkCudaErrors(cudaFree(pdata.d_offset));
  checkCudaErrors(cudaFree(pdata.d_root));
  checkCudaErrors(cudaFree(pdata.d_run_length));
  checkCudaErrors(cudaFree(pdata.d_sorted_arg));
  checkCudaErrors(cudaFree(pdata.u_shape));
  checkCudaErrors(cudaFree(pdata.u_shape_exchanged));
  checkCudaErrors(cudaFreeHost(pdata.h_shape));
  checkCudaErrors(cudaFreeHost(pdata.h_shape_exchanged));
}

} // namespace hetuCTR
