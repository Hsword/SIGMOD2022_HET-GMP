#include "hetu_gpu_table.h"
#include "common/helper_cuda.h"

#include <cub/cub.cuh>

namespace hetuCTR {

__global__ void writeback_update_kernel(HetuTable *tbl, size_t len) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < len) {
    index_t embedding_idx = tbl->d_query_idx_[1][id];
    auto iter = tbl->table_->find(embedding_idx);
    if (iter != tbl->table_->end()) {
      index_t mem_offset = iter->second;
      assert(mem_offset < tbl->kNonLocalStorageMax);
      tbl->d_version_[mem_offset] = tbl->d_return_version_[1][id];
      for (int i = 0; i < tbl->kEmbeddingWidth; i++) {
        tbl->d_embedding_[tbl->kEmbeddingWidth * mem_offset + i] =
          tbl->d_return_val_[1][tbl->kEmbeddingWidth * id + i] + tbl->d_gradient_[tbl->kEmbeddingWidth * mem_offset + i];
      }
    }
  }
}

__global__ void writeback_kernel(HetuTable *tbl, embed_t *dst) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  size_t width = tbl->kEmbeddingWidth;
  size_t wid = id % width;
  id = id / width;
  if (id >= tbl->cur_batch_.batch_size) return;
  index_t mapped_idx = tbl->cur_batch_.d_idx_map[id];
  embed_t *val;
  index_t mem_offset = tbl->cur_batch_.d_offset[mapped_idx];
  if (mem_offset == kInvalidIndex) {
    index_t ret_offset = tbl->d_return_outdated_[0][mapped_idx];
    val = &tbl->d_return_val_[1][width * ret_offset];
  } else {
    val = &tbl->d_embedding_[width * mem_offset];
  }
  dst[width * id + wid] = val[wid];
}

void HetuTable::writeBack(embed_t *dst) {
  // Compute the prefix sum for return_outdated
  checkCudaErrors(cub::DeviceScan::ExclusiveSum(d_temp_, temp_bytes_,
    d_return_outdated_[1], d_return_outdated_[0], cur_batch_.unique_size, stream_main_));

  // Select index that need to be updated into d_query_idx[1]
  checkCudaErrors(cub::DeviceSelect::Flagged(d_temp_, temp_bytes_,
    d_query_idx_[0], d_return_outdated_[1], d_query_idx_[1],
    d_shape_, cur_batch_.unique_size, stream_main_));

  // Update received value into local storage
  writeback_update_kernel<<<DIM_GRID(all2all_received_), DIM_BLOCK, 0, stream_main_>>>(d_this, all2all_received_);
  writeback_kernel<<<DIM_GRID(cur_batch_.batch_size * kEmbeddingWidth), DIM_BLOCK, 0, stream_main_>>>(d_this, dst);
}

} // namespace hetuCTR
