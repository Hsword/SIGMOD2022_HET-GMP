#include "hetu_gpu_table.h"

#include "common/helper_cuda.h"

using namespace hetuCTR;

unsigned long hetuCTR::__seed = 0;

void HetuTable::pushPull(embed_t *grad, embed_t *dst) {
  checkCudaErrors(cudaSetDevice(device_id_));

  // If no grad is provided, than this batch is considered as inference batch.
  // Set shapes in previous batch to 0, so that no kernel will be launched and no data will be sent and received
  if (grad == nullptr) {
    prev_batch_.batch_size = 0;
    prev_batch_.unique_size = 0;
    for (int i = 0; i <= nrank_; i++) {
      prev_batch_.h_shape[i] = 0;
      prev_batch_.h_shape_exchanged[i] = 0;
    }
  }

  generateGradient(grad);

  generateQuery();

  all2allExchangeQuery();

  handleGradient();

  handleQuery();

  checkCudaErrors(cudaStreamSynchronize(stream_main_));

  all2allReturnValue();

  writeBack(dst);

  checkCudaErrors(cudaStreamSynchronize(stream_main_));
  return;
}

void HetuTable::preprocess(index_t *data_ptr, size_t batch_size) {
  checkCudaErrors(cudaSetDevice(device_id_));
  std::swap(cur_batch_, prev_batch_);
  if (batch_size > batch_size_reserved_) {
    allocateAuxillaryMemory(batch_size);
  }
  if (batch_size > cur_batch_.allocate_size) {
    INFO("ReAllocate cuda memory for batch ", cur_batch_.batch_size, "->" , batch_size);
    freePreprocessData(cur_batch_);
    createPreprocessData(cur_batch_, batch_size, nrank_);
  }
  cur_batch_.batch_size = batch_size;

  // sync data with this pointer on device
  checkCudaErrors(cudaMemcpyAsync(
    d_this, this, sizeof(HetuTable), cudaMemcpyHostToDevice, stream_main_));

  preprocessIndex(data_ptr, batch_size);

  preprocessGradient();

  checkCudaErrors(cudaStreamSynchronize(stream_main_));
}
