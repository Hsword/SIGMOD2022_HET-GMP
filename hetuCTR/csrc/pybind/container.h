#pragma once

#include "pybind.h"
#include "core/hetu_gpu_table.h"

#include "utils/thread_pool.h"

namespace hetuCTR {

class TableContainer : public HetuTable {
public:
  using HetuTable::HetuTable;

  void pushPull(unsigned long grad, unsigned long dst) {
    py::gil_scoped_release release;
    HetuTable::pushPull(reinterpret_cast<embed_t *>(grad), reinterpret_cast<embed_t *>(dst));
  }

  void preprocess(unsigned long data_ptr, size_t batch_size) {
    py::gil_scoped_release release;
    HetuTable::preprocess(reinterpret_cast<index_t *>(data_ptr), batch_size);
  }

  std::future<void> pushPullAsync(unsigned long grad, unsigned long dst) {
    if (!pool_) pool_ = std::make_unique<ThreadPool>(1);
    return pool_->Enqueue(&HetuTable::pushPull, this,
      reinterpret_cast<embed_t *>(grad), reinterpret_cast<embed_t *>(dst));
  }

  std::future<void> preprocessAsync(unsigned long data_ptr, size_t batch_size) {
    if (!pool_) pool_ = std::make_unique<ThreadPool>(1);
    return pool_->Enqueue(&HetuTable::preprocess, this,
      reinterpret_cast<index_t *>(data_ptr), batch_size);
  }
private:
  std::unique_ptr<ThreadPool> pool_;
};

} // namespace hetuCTR
