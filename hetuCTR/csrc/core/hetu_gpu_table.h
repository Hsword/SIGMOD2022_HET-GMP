#pragma once

#include <cuda_runtime.h>
#include <nccl.h>

#include "types.h"
#include "preprocess_struct.h"
#include "common/sarray.h"
#include "common/logging.h"
#include "utils/initializer.h"

#include "utils/pinned.cuh"
#include "cudf/concurrent_unordered_map.cuh"

namespace hetuCTR {

/**
 * @brief Distributed GPU Table for embedding-based training
 *
 */
class HetuTable : public pinned {
public:
  const int rank_;
  const int nrank_;
  const int device_id_;

  const size_t kEmbeddingIDMax;
  const size_t kEmbeddingWidth;
  const size_t kStorageMax;
  size_t kNonLocalStorageMax;

  const embed_t learning_rate_;

  // maxinum size of a batch fed,
  // when the table received a larger batch, it will have to reallocate memory
  size_t batch_size_reserved_ = 1;
  size_t all2all_received_, all2all_gradient_received_;

  const version_t pull_bound_, push_bound_;

  cudaStream_t stream_main_, stream_sub_;
  ncclComm_t communicator_;

  embed_t * d_embedding_;
  embed_t * d_gradient_;
  version_t * d_updates_;
  version_t * d_version_;
  worker_t * d_root_;

  // temp memory used in some cuda-based algorithm
  void * d_temp_ = nullptr;
  size_t temp_bytes_ = 0;

  index_t * d_need_update_ = nullptr;
  index_t * d_update_prefix_ = nullptr;
  size_t * d_shape_ = nullptr;

  // a pointer points to self that can be used in device
  HetuTable *d_this;

  // query buffer, dual buffer for send and receive
  version_t * d_query_version_[2] = {};
  version_t * d_query_updates_[2] = {};
  index_t * d_query_idx_[2] = {};
  index_t * d_query_gradient_idx_[2] = {};
  embed_t * d_query_val_[2] = {};

  index_t * d_return_outdated_[2] = {};
  embed_t * d_return_val_[2] = {};
  version_t * d_return_version_[2] = {};

  PreprocessData cur_batch_, prev_batch_;
  concurrent_unordered_map<index_t, index_t, kInvalidIndex> *table_;

  int verbose_;
  /**
   * @brief Initialize cuda and nccl communicator
   *
   * @param ip IPv4 address to setup collective communication
   * @param port IPv4 port
   */
  void initializeNCCL(const std::string &ip, const int port);
  void initializeTable(SArray<worker_t> root_id_arr, SArray<index_t> storage_id_arr);
  void allocateAuxillaryMemory(size_t batch_size);
  void freeAuxillaryMemory();

  void generateQuery();
  void generateGradient(embed_t *grad);
  void preprocessGradient();
  void preprocessIndex(index_t *data, size_t batch_size);
  void handleGradient();
  void handleQuery();
  void writeBack(embed_t *dst);
  void all2allExchangeShape(const size_t *shape, size_t *shape_out);
  void all2allExchangeQuery();
  void all2allReturnOutdated();
  void all2allReturnValue();

  template <class T> int __printarg(T t) { std::cout << t; return 0; }
  template<class ...Args>
  inline void INFO(Args ...args) {
    if (verbose_ >= 1) {
      std::cout << "HetuTable rank " << rank_ << ": ";
      std::initializer_list<int>({__printarg(args)...});
      std::cout << std::endl;
    }
  }

  HetuTable(
    const int rank,
    const int nrank,
    const int device_id,
    const std::string &ip,
    const int port,
    const size_t embedding_length,
    const size_t embedding_width,
    const version_t pull_bound,
    const version_t push_bound,
    SArray<worker_t> root_id_arr,
    SArray<index_t> storage_id_arr,
    const Initializer &init,
    const embed_t learning_rate,
    const int verbose
  );
  HetuTable(const HetuTable &) = delete;
  HetuTable& operator=(const HetuTable&) = delete;
  /**
   * @brief preprocess next batch index
   *
   * @param data_ptr an address holding index
   * @param len the length of index array
   */
  void preprocess(index_t *data_ptr, size_t batch_size);

  /**
   * @brief Update embedding Table with the gradients and then fetch embedding value to dst
   *
   * @param grad points to gradients array
   * @param dst where embedding are written to
   */
  void pushPull(embed_t *grad, embed_t *dst);
  std::string debugString();
  std::string debugStringFull();
  ~HetuTable();
};

} // namespace hetuCTR
