#include "gpu_runtime.h"

/// Cuda block dim
const size_t DIM_BLOCK = 256;
#define DIM_GRID(x) ( ((size_t)x + DIM_BLOCK - 1) / DIM_BLOCK )

__global__ void embedding_lookup_kernel(const float *input, const float *ids, float *output, size_t size, size_t width) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  size_t wid = id % width;
  id = id / width;
  if (id >= size) return;
  int embedding_idx = ids[id];
  output[width * id + wid] = input[width * embedding_idx + wid];
}

int DLGpuEmbeddingLookUp(const DLArrayHandle input, const DLArrayHandle ids,
                         DLArrayHandle output, DLStreamHandle stream_handle = NULL){
  assert(input->ndim == 2);
  size_t size = 1;
  for(int i = 0; i < output->ndim; i++){
    if(i < output->ndim - 1){
      assert(ids->shape[i] == output->shape[i]);
    }else if(i == output->ndim - 1){
      assert(input->shape[1] == output->shape[i]);
    }
  }
  for(int i = 0; i < ids->ndim; i++){
    size = size * ids->shape[i];
  }
  size_t width = input->shape[1];
  // printf("size = %d, width = %d\n", size, width);
  float *output_data = (float *)output->data;
  const float *input_data = (const float *)input->data;
  const float *id_list = (const float*)ids->data;
  cudaStream_t s = stream_handle ? *(cudaStream_t*)stream_handle->handle : cudaStreamDefault;
  embedding_lookup_kernel<<<DIM_GRID(size * width), DIM_BLOCK, 0, s>>>(
    input_data, id_list, output_data, size, width);
  return 0;
}
