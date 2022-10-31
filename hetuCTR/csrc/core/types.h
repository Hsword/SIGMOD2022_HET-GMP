#pragma once

#include <limits>

extern "C" {
#include <sys/types.h>
}

namespace hetuCTR {

/// types of the embedding item
typedef float embed_t;

/// types of version number used in hetu
/// must be signed
typedef long long version_t;

/// types used in memory offset and feature input
typedef long long index_t;

/// Default invalid index usd in hash map
const index_t kInvalidIndex = -1;

/// This version is smaller than any other version, so it will be regarded as outdated
const version_t kInvalidVersion = std::numeric_limits<version_t>::min();

/// types of worker id
typedef unsigned char worker_t;

/// Cuda block dim
const size_t DIM_BLOCK = 128;

#define DIM_GRID(x) ( ((size_t)x + DIM_BLOCK - 1) / DIM_BLOCK )

#ifdef NCCL_H_
const ncclDataType_t embed_nccl_t = ncclFloat32;
const ncclDataType_t index_nccl_t = ncclInt64;
const ncclDataType_t version_nccl_t = ncclInt64;
#endif

extern unsigned long __seed;

} // namespace hetuCTR
