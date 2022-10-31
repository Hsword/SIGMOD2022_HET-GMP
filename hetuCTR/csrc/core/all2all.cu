#include "hetu_gpu_table.h"
#include "common/helper_cuda.h"

namespace hetuCTR {

void HetuTable::all2allExchangeShape(const size_t *shape, size_t *shape_out) {
  assert(shape != shape_out);
  checkCudaErrors(ncclGroupStart());
  for (int i = 0; i < nrank_; i++) {
    checkCudaErrors(ncclSend(
      shape + i, 1, ncclUint64, i, communicator_, stream_main_));
    checkCudaErrors(ncclRecv(
      shape_out + i, 1, ncclUint64, i, communicator_, stream_main_));
  }
  checkCudaErrors(ncclGroupEnd());
}

void HetuTable::all2allExchangeQuery() {
  size_t snd_offset = 0, rcvd_offset = 0;
  checkCudaErrors(ncclGroupStart());
  for (int i = 0; i < nrank_; i++) {
    checkCudaErrors(ncclSend(
      d_query_idx_[0] + snd_offset, cur_batch_.h_shape[i], index_nccl_t, i, communicator_, stream_main_));
    checkCudaErrors(ncclRecv(
      d_query_idx_[1] + rcvd_offset, cur_batch_.h_shape_exchanged[i], index_nccl_t, i, communicator_, stream_main_));
    snd_offset += cur_batch_.h_shape[i];
    rcvd_offset += cur_batch_.h_shape_exchanged[i];
  }
  all2all_received_ = rcvd_offset;
  // currently, we have to make sure each worker have the same batchsize
  // under such assumption, the received number won't exceed this value
  assert(all2all_received_ <= batch_size_reserved_ * nrank_);
  checkCudaErrors(ncclGroupEnd());

  checkCudaErrors(ncclGroupStart());
  snd_offset = 0, rcvd_offset = 0;
  for (int i = 0; i < nrank_; i++) {
    checkCudaErrors(ncclSend(
      d_query_version_[0] + snd_offset, cur_batch_.h_shape[i], version_nccl_t, i, communicator_, stream_main_));
    checkCudaErrors(ncclRecv(
      d_query_version_[1] + rcvd_offset, cur_batch_.h_shape_exchanged[i], version_nccl_t, i, communicator_, stream_main_));
    snd_offset += cur_batch_.h_shape[i];
    rcvd_offset += cur_batch_.h_shape_exchanged[i];
  }
  checkCudaErrors(ncclGroupEnd());

  // ---- gradient part, using prev_batch ---
  checkCudaErrors(ncclGroupStart());
  snd_offset = 0, rcvd_offset = 0;
  for (int i = 0; i < nrank_; i++) {
    checkCudaErrors(ncclSend(
      d_query_gradient_idx_[0] + snd_offset, prev_batch_.h_shape[i], index_nccl_t, i, communicator_, stream_main_));
    checkCudaErrors(ncclRecv(
      d_query_gradient_idx_[1] + rcvd_offset, prev_batch_.h_shape_exchanged[i], index_nccl_t, i, communicator_, stream_main_));
    snd_offset += prev_batch_.h_shape[i];
    rcvd_offset += prev_batch_.h_shape_exchanged[i];
  }
  checkCudaErrors(ncclGroupEnd());

  checkCudaErrors(ncclGroupStart());
  snd_offset = 0, rcvd_offset = 0;
  for (int i = 0; i < nrank_; i++) {
    checkCudaErrors(ncclSend(
      d_query_updates_[0] + snd_offset, prev_batch_.h_shape[i], index_nccl_t, i, communicator_, stream_main_));
    checkCudaErrors(ncclRecv(
      d_query_updates_[1] + rcvd_offset, prev_batch_.h_shape_exchanged[i], index_nccl_t, i, communicator_, stream_main_));
    snd_offset += prev_batch_.h_shape[i];
    rcvd_offset += prev_batch_.h_shape_exchanged[i];
  }
  checkCudaErrors(ncclGroupEnd());

  checkCudaErrors(ncclGroupStart());
  snd_offset = 0, rcvd_offset = 0;
  for (int i = 0; i < nrank_; i++) {
    checkCudaErrors(ncclSend(
      d_query_val_[0] + snd_offset * kEmbeddingWidth, prev_batch_.h_shape[i] * kEmbeddingWidth,
      embed_nccl_t, i, communicator_, stream_main_));
    checkCudaErrors(ncclRecv(
      d_query_val_[1] + rcvd_offset * kEmbeddingWidth, prev_batch_.h_shape_exchanged[i] * kEmbeddingWidth,
      embed_nccl_t, i, communicator_, stream_main_));
    snd_offset += prev_batch_.h_shape[i];
    rcvd_offset += prev_batch_.h_shape_exchanged[i];
  }
  checkCudaErrors(ncclGroupEnd());
  all2all_gradient_received_ = rcvd_offset;
  INFO("Total gradient update receive/push = ", rcvd_offset, "/", snd_offset);
}

void HetuTable::all2allReturnOutdated() {
  checkCudaErrors(ncclGroupStart());
  size_t snd_offset = 0, rcvd_offset = 0;
  for (int i = 0; i < nrank_; i++) {
    checkCudaErrors(ncclSend(
      d_return_outdated_[0] + snd_offset, cur_batch_.h_shape_exchanged[i], index_nccl_t, i, communicator_, stream_main_));
    checkCudaErrors(ncclRecv(
      d_return_outdated_[1] + rcvd_offset, cur_batch_.h_shape[i], index_nccl_t, i, communicator_, stream_main_));
    snd_offset += cur_batch_.h_shape_exchanged[i];
    rcvd_offset += cur_batch_.h_shape[i];
  }
  checkCudaErrors(ncclGroupEnd());
}

void HetuTable::all2allReturnValue() {
  checkCudaErrors(ncclGroupStart());
  size_t snd_offset = 0, rcvd_offset = 0;
  for (int i = 0; i < nrank_; i++) {
    checkCudaErrors(ncclSend(
      d_return_version_[0] + snd_offset, cur_batch_.h_shape[i], version_nccl_t, i, communicator_, stream_main_));
    checkCudaErrors(ncclRecv(
      d_return_version_[1] + rcvd_offset, cur_batch_.h_shape_exchanged[i], version_nccl_t, i, communicator_, stream_main_));
    snd_offset += cur_batch_.h_shape[i];
    rcvd_offset += cur_batch_.h_shape_exchanged[i];
  }
  checkCudaErrors(ncclGroupEnd());
  checkCudaErrors(ncclGroupStart());
  snd_offset = 0, rcvd_offset = 0;
  for (int i = 0; i < nrank_; i++) {
    checkCudaErrors(ncclSend(
      d_return_val_[0] + snd_offset * kEmbeddingWidth, cur_batch_.h_shape[i] * kEmbeddingWidth,
      embed_nccl_t, i, communicator_, stream_main_));
    checkCudaErrors(ncclRecv(
      d_return_val_[1] + rcvd_offset * kEmbeddingWidth, cur_batch_.h_shape_exchanged[i] * kEmbeddingWidth,
      embed_nccl_t, i, communicator_, stream_main_));
    snd_offset += cur_batch_.h_shape[i];
    rcvd_offset += cur_batch_.h_shape_exchanged[i];
  }
  checkCudaErrors(ncclGroupEnd());
  all2all_received_ = rcvd_offset;
  INFO("Total embedding fetching serve/query = ", rcvd_offset, "/", snd_offset);
}

} // namespace hetuCTR

