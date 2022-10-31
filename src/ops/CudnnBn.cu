#include "gpu_runtime.h"


__global__ void init_mean_and_var_kernel(float *mean, float *var, size_t size){
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= size) return;
  mean[id] = 0;
  var[id] = 0;
}

__global__ void copy_mean_and_var_kernel(float *mean, float *var, float *saved_mean, float *saved_var, size_t size){
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= size) return;
  saved_mean[id] = mean[id];
  saved_var[id] = var[id];

}

// the shape of bn_scale/bias   1*C*1*1
int CuDNN_DLGpuBatch_Normalization(const DLArrayHandle input_X, const DLArrayHandle bn_scale, const DLArrayHandle bn_bias, DLArrayHandle output_Y,
                                   float momentum , float eps, DLArrayHandle save_mean_arr, DLArrayHandle save_var_arr, DLStreamHandle stream_handle = NULL){

  int dev_id = (input_X->ctx).device_id;
  cudaSetDevice(dev_id);
  cudnn_init(dev_id, stream_handle);

  // input
  size_t input_N = input_X -> shape[0];
  size_t input_C = input_X -> shape[1];
  size_t input_H = input_X -> shape[2];
  size_t input_W = input_X -> shape[3];
  const float * input_data = (const float*)(input_X -> data);

  //input descriptor
  cudnnTensorDescriptor_t input_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_N, input_C, input_H, input_W));

  //output
  float *output_data = (float *)(output_Y -> data);

  //output descriptor
  cudnnTensorDescriptor_t output_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&output_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_N, input_C, input_H, input_W));

  //bn parameter descriptor
  cudnnTensorDescriptor_t bnScaleBiasMeanVar_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&bnScaleBiasMeanVar_desc));
  CUDNN_CALL(cudnnDeriveBNTensorDescriptor(bnScaleBiasMeanVar_desc, input_desc, CUDNN_BATCHNORM_SPATIAL));// after conv

  //bn parameter
  const float *bn_scale_data = (const float *) (bn_scale -> data);
  const float *bn_bias_data = (const float *) (bn_bias -> data);

  if(is_chunk_init(dev_id) == false){
    chunk_init(dev_id);
  }
  size_t workspace_size = sizeof(float) * input_C;
  void* running_mean = find_chunk(workspace_size, dev_id);
  void* running_var = find_chunk(workspace_size, dev_id);
  // void *running_mean, *running_var;
  // CUDA_CALL(cudaMalloc(&running_mean, sizeof(float) * input_C));
  // CUDA_CALL(cudaMalloc(&running_var, sizeof(float) * input_C));
  // init running mean and running var
  // the tensors should be initialized to some reasonable values or to 0.
  size_t blocks = (input_C + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  if(stream_handle){
    cudaStream_t *s = (cudaStream_t*)(stream_handle->handle);
    init_mean_and_var_kernel<<<blocks, THREADS_PER_BLOCK, 0, *s>>>((float*)running_mean, (float*)running_var, input_C);
  }
  else{
    init_mean_and_var_kernel<<<blocks, THREADS_PER_BLOCK>>>((float*)running_mean, (float*)running_var, input_C);
  }
  /************************************************************/
  void *save_mean = save_mean_arr -> data;
  void *save_var = save_var_arr -> data;

  float alpha = 1.0f;
  float beta = 0.0f;

  CUDNN_CALL(cudnnBatchNormalizationForwardTraining(cudnn_map[dev_id], CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,
                                                    input_desc, input_data,
                                                    output_desc, output_data,
                                                    bnScaleBiasMeanVar_desc,
                                                    bn_scale_data, bn_bias_data,
                                                    momentum, (float *)save_mean, (float *)save_var,
                                                    eps,running_mean,running_var ));


  del_chunk(running_mean, dev_id);
  del_chunk(running_var, dev_id);

  //CUDA_CALL(cudaFree(running_mean));
  //CUDA_CALL(cudaFree(running_var));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(bnScaleBiasMeanVar_desc));

  return 0;
}

int CuDNN_DLGpuBatch_Normalization_gradient(const DLArrayHandle gradient_Y, const DLArrayHandle input_X, const DLArrayHandle bn_scale, DLArrayHandle gradient_X,
                                            DLArrayHandle gradient_bn_scale, DLArrayHandle gradient_bn_bias, float eps, DLArrayHandle save_mean_arr,
                                            DLArrayHandle save_var_arr, DLStreamHandle stream_handle = NULL){
  // CUDNN_CALL(cudnnCreate(&cudnn));
  int dev_id = (input_X->ctx).device_id;
  cudnn_init(dev_id, stream_handle);

  // input
  size_t input_N = input_X -> shape[0];
  size_t input_C = input_X -> shape[1];
  size_t input_H = input_X -> shape[2];
  size_t input_W = input_X -> shape[3];
  const float * input_data = (const float*)(input_X -> data);

  //input descriptor
  cudnnTensorDescriptor_t input_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_N, input_C, input_H, input_W));

  //output
  const float *gradient_y_data = (const float *)(gradient_Y -> data);

  //output descriptor
  cudnnTensorDescriptor_t output_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&output_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_N, input_C, input_H, input_W));

  //bn parameter descriptor
  cudnnTensorDescriptor_t bnScaleBiasMeanVar_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&bnScaleBiasMeanVar_desc));
  CUDNN_CALL(cudnnDeriveBNTensorDescriptor(bnScaleBiasMeanVar_desc, input_desc, CUDNN_BATCHNORM_SPATIAL));// after conv

  const float * bn_scale_data = (const float *)(bn_scale->data);

  // x gradient
  float *gradient_x_data = (float *) (gradient_X -> data);
  //bn gradient
  float *gradient_bn_bias_data = (float *)(gradient_bn_bias -> data);
  float *gradient_bn_scale_data = (float *)(gradient_bn_scale -> data);
  void *saved_mean = save_mean_arr -> data;
  void *saved_var = save_var_arr -> data;
  float one = 1.0f;
  float zero = 0.0f;

  CUDNN_CALL(cudnnBatchNormalizationBackward(cudnn_map[dev_id], CUDNN_BATCHNORM_SPATIAL_PERSISTENT, &one, &zero, &one, &zero,
                                             input_desc, input_data,
                                             output_desc, gradient_y_data,
                                             input_desc, gradient_x_data,
                                             bnScaleBiasMeanVar_desc,
                                             bn_scale_data,
                                             gradient_bn_scale_data,
                                             gradient_bn_bias_data,
                                             eps, saved_mean, saved_var));


  CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(bnScaleBiasMeanVar_desc));
  // CUDNN_CALL(cudnnDestroy(cudnn));
  return 0;
}

int CuDNN_DLGpuBatch_Normalization_inference(const DLArrayHandle input_X, const DLArrayHandle bn_scale, const DLArrayHandle bn_bias, DLArrayHandle output_Y, float eps, DLArrayHandle save_mean_arr, DLArrayHandle save_var_arr, DLStreamHandle stream_handle = NULL){

  int dev_id = (input_X->ctx).device_id;
  cudaSetDevice(dev_id);
  cudnn_init(dev_id, stream_handle);

  // input
  size_t input_N = input_X -> shape[0];
  size_t input_C = input_X -> shape[1];
  size_t input_H = input_X -> shape[2];
  size_t input_W = input_X -> shape[3];
  const float * input_data = (const float*)(input_X -> data);

  //input descriptor
  cudnnTensorDescriptor_t input_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_N, input_C, input_H, input_W));

  //output
  float *output_data = (float *)(output_Y -> data);

  //output descriptor
  cudnnTensorDescriptor_t output_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&output_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_N, input_C, input_H, input_W));

  //bn parameter descriptor
  cudnnTensorDescriptor_t bnScaleBiasMeanVar_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&bnScaleBiasMeanVar_desc));
  CUDNN_CALL(cudnnDeriveBNTensorDescriptor(bnScaleBiasMeanVar_desc, input_desc, CUDNN_BATCHNORM_SPATIAL));// after conv

  //bn parameter
  const float *bn_scale_data = (const float *) (bn_scale -> data);
  const float *bn_bias_data = (const float *) (bn_bias -> data);

  if(is_chunk_init(dev_id) == false){
    chunk_init(dev_id);
  }

  const float *save_mean = (const float *)save_mean_arr -> data;
  const float *save_var = (const float *)save_var_arr -> data;

  float alpha = 1.0f;
  float beta = 0.0f;

  CUDNN_CALL(cudnnBatchNormalizationForwardInference(cudnn_map[dev_id], CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,
                                                    input_desc, input_data,
                                                    output_desc, output_data,
                                                    bnScaleBiasMeanVar_desc,
                                                    bn_scale_data, bn_bias_data,
                                                    save_mean, save_var, eps));

  CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(bnScaleBiasMeanVar_desc));

  return 0;
}