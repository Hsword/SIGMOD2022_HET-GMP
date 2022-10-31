######################
### Set targets ######
######################

# hetu main version, choose from (mkl, gpu, all)
# if using mkl (for CPU) or all, OpenMP(*), mkl required
# if using gpu or all, OpenMP(*), CUDA(*), CUDNN(*) required
set(HETU_VERSION "gpu")

# whether to compile allreduce module
# nccl(*), openmpi required
set(HETU_ALLREDUCE ON)

# whether to compile ps module
# protobuf(*), zeromq required
set(HETU_PS OFF)

# whether to compile geometric module (for GNNs)
# pybind11(*), metis(*) required
set(HETU_GEOMETRIC OFF)

# whether to compile cache module (for PS)
# to enable this, you must turn HETU_PS on
# pybind11(*) required
set(HETU_CACHE OFF)

# whether to compile Hetu ML Module
set(HETU_ML OFF)
set(HETU_PARALLEL_ML ON)


######################
### Set paths ########
######################

# CUDA version >= 10.1
set(CUDAToolkit_ROOT /usr/local/cuda)
set(NCCL_ROOT $ENV{CONDA_PREFIX})
set(CUDNN_ROOT $ENV{CONDA_PREFIX})

set(ZMQ_ROOT $ENV{CONDA_PREFIX})
