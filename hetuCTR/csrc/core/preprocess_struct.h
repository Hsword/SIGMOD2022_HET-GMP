#pragma once

#include "types.h"

namespace hetuCTR {

struct PreprocessData {
  size_t batch_size = 0;
  size_t allocate_size = 0;
  size_t unique_size = 0;
  index_t *d_idx = nullptr;
  index_t *d_unique_idx = nullptr;
  index_t *d_sorted_arg = nullptr;
  index_t *d_run_length = nullptr;
  index_t *d_idx_map = nullptr;
  worker_t *d_root = nullptr;
  index_t *d_offset = nullptr;
  size_t *u_shape = nullptr;
  size_t *u_shape_exchanged = nullptr;
  // host copy of d_shape
  size_t *h_shape = nullptr;
  // host copy of d_shape_exchanged
  size_t *h_shape_exchanged = nullptr;
};

void createPreprocessData(PreprocessData &pdata, size_t batch_size, size_t nrank);

void freePreprocessData(PreprocessData &pdata);

} // namespace hetuCTR
