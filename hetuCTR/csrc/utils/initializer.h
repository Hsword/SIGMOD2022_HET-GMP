#pragma once

#include <core/types.h>
#include <utility>

namespace hetuCTR {

enum class InitType {
  kZero,
  kNormal,
  kUniform,
  kTruncatedNormal,
  kNone,
};

/**
 * @brief Initializer for variables
 * @details For normal initializer param would be mean, stddev
 *        For Uniform initializer param would be a, b
 */
struct Initializer {
  Initializer(InitType _type, float a, float b) : type(_type), param(a, b) {}
  InitType type;
  std::pair<float, float> param;
};

/**
 * @brief Initialize data array with initializer
 *
 * @param data point to memory , can be either host of device
 * @param len length of data to be initialized
 * @param init Initializer
 * @param host whether the data is on host memory
 * @param seed random seed
 */
void initialize(embed_t *data, size_t len, Initializer init, bool host=false, unsigned int seed=0);

} // namespace hetuCTR

