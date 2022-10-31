#include "initializer.h"
#include "common/logging.h"
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_ptr.h>

namespace hetuCTR {

struct _Uniform {
  const float a, b;
  const unsigned int seed;

  __host__ __device__
    _Uniform(float _a, float _b, unsigned int _seed)
    : a(_a), b(_b), seed(_seed) {};

  __host__ __device__
    embed_t operator()(const unsigned long long n) const
  {
    thrust::default_random_engine rng(seed);
    thrust::uniform_real_distribution<embed_t> dist(a, b);
    rng.discard(n);

    return dist(rng);
  }
};

struct _Normal {
  const float mean, stddev;
  const unsigned int seed;

  __host__ __device__
    _Normal(float _mean, float _stddev, unsigned int _seed)
      : mean(_mean), stddev(_stddev), seed(_seed) {};

  __host__ __device__
    embed_t operator()(const unsigned long long n) const
  {
    thrust::default_random_engine rng(seed);
    thrust::normal_distribution<embed_t> dist(mean, stddev);
    rng.discard(n);

    return dist(rng);
  }
};

struct _TruncatedNormal {
  const float mean, stddev;
  const unsigned int seed;

  __host__ __device__
    _TruncatedNormal(float _mean, float _stddev, unsigned int _seed)
      : mean(_mean), stddev(_stddev), seed(_seed) {};

  __host__ __device__
    embed_t operator()(const unsigned long long n) const
  {
    thrust::default_random_engine rng(seed);
    thrust::normal_distribution<embed_t> dist(mean, stddev);
    embed_t result;
    rng.discard(n);
    do {
      result = dist(rng);
      rng.seed(rng());
    } while (result > mean + 2 * stddev || result < mean - 2 * stddev);
    return result;
  }
};

template<class Iterator>
void _initialize(Iterator iter, size_t len, Initializer init, bool host, unsigned int seed) {
  thrust::counting_iterator<unsigned long long> sequence(0);
  switch (init.type)
  {
  case InitType::kZero:
    thrust::fill(iter, iter + len, 0);
    break;
  case InitType::kNormal:
    thrust::transform(sequence, sequence + len, iter, _Normal(init.param.first, init.param.second, seed));
    break;
  case InitType::kUniform:
    thrust::transform(sequence, sequence + len, iter, _Uniform(init.param.first, init.param.second, seed));
    break;
  case InitType::kTruncatedNormal:
    thrust::transform(sequence, sequence + len, iter, _TruncatedNormal(init.param.first, init.param.second, seed));
    break;
  default:
    LF << "Unknown initializer.";
  }
}

void initialize(embed_t* data, size_t len, Initializer init, bool host, unsigned int seed) {
  if (len == 0) return;
  thrust::counting_iterator<unsigned long long> sequence(0);
  if (host) {
    _initialize(data, len, init, host, seed);
  } else {
    thrust::device_ptr<embed_t> dev_ptr(data);
    _initialize(dev_ptr, len, init, host, seed);
  }
}

} // namespace hetuCTR
