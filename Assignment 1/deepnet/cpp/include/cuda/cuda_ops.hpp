#pragma once

#ifdef USE_CUDA

#include <vector>

namespace deepnet {
namespace cuda {

// Low-level CUDA kernels (operate on device pointers)
void add_cuda(const float *a, const float *b, float *out, int size);
void mul_cuda(const float *a, const float *b, float *out, int size);
void matmul_cuda(const float *a, const float *b, float *out, int M, int N,
                 int K);
void relu_cuda(const float *input, float *output, int size);
void sigmoid_cuda(const float *input, float *output, int size);
void tanh_cuda(const float *input, float *output, int size);
void sub_cuda(const float *a, const float *b, float *out, int size);
void div_cuda(const float *a, const float *b, float *out, int size);
void pow_cuda(const float *input, float exponent, float *output, int size);
void exp_cuda(const float *input, float *output, int size);
void log_cuda(const float *input, float *output, int size);
void sqrt_cuda(const float *input, float *output, int size);

// Scalars
void add_scalar_cuda(const float *input, float scalar, float *output, int size);
void mul_scalar_cuda(const float *input, float scalar, float *output, int size);

// Reductions
void sum_cuda(const float *input, float *output, int size);
void max_cuda(const float *input, float *output, int size);



// Memory operations
void *cuda_malloc(size_t size);
void cuda_free(void *ptr);
void cuda_memset(void *ptr, int value, size_t size);
void cuda_memcpy_host_to_device(void *dst, const void *src, size_t size);
void cuda_memcpy_device_to_host(void *dst, const void *src, size_t size);
void cuda_memcpy_device_to_device(void *dst, const void *src, size_t size);

// ============================================================
// Low-level CUDA kernels (device pointers)
// ============================================================
void relu_backward_cuda(const float *grad_output, const float *input,
                        float *grad_input, int size);
void sigmoid_backward_cuda(const float *grad_output, const float *output,
                           float *grad_input, int size);
void tanh_backward_cuda(const float *grad_output, const float *output,
                        float *grad_input, int size);

void max_pool2d_forward_cuda(const float *input, float *output, int *indices,
                             int N, int C, int H, int W, int out_h, int out_w,
                             int k, int stride);
void max_pool2d_backward_cuda(const float *grad_output, const int *indices,
                              float *grad_input, int N, int C, int H, int W,
                              int out_h, int out_w, int size);
void avg_pool2d_forward_cuda(const float *input, float *output, int N, int C,
                             int H, int W, int out_h, int out_w, int k,
                             int stride);
void avg_pool2d_backward_cuda(const float *grad_output, float *grad_input,
                              int N, int C, int H, int W, int out_h, int out_w,
                              int k, int stride, int size);

// BatchNorm
void batchnorm_forward_cuda(const float *input, const float *mean,
                            const float *var, const float *gamma,
                            const float *beta, float *output, int N, int C,
                            int H, int W, float eps);
                            
// New kernels for training forward and full backward
void batchnorm_stats_cuda(const float *input, float *mean, float *var,
                          int N, int C, int H, int W);
                          
void batchnorm_backward_reduce_cuda(const float *grad_output, const float *input,
                                    const float *mean, const float *var,
                                    float *grad_gamma, float *grad_beta,
                                    int N, int C, int H, int W, float eps);
                                    
void batchnorm_backward_apply_cuda(const float *grad_output, const float *input,
                                   const float *mean, const float *var,
                                   const float *gamma, const float *grad_gamma,
                                   const float *grad_beta, float *grad_input,
                                   int N, int C, int H, int W, float eps);


// ============================================================
// Device pointer wrappers: operate directly on device pointers (no copying)
// Use these when tensors already have device memory allocated
// ============================================================
void relu_cuda_device(const float *d_input, float *d_output, int size);
void sigmoid_cuda_device(const float *d_input, float *d_output, int size);
void tanh_cuda_device(const float *d_input, float *d_output, int size);
void add_cuda_device(const float *d_a, const float *d_b, float *d_out, int size);
void add_inplace_cuda_device(float *a, const float *b, int size);
void mul_cuda_device(const float *d_a, const float *d_b, float *d_out, int size);
void sub_cuda_device(const float *d_a, const float *d_b, float *d_out, int size);
void div_cuda_device(const float *d_a, const float *d_b, float *d_out, int size);
void matmul_cuda_device(const float *d_a, const float *d_b, float *d_out, int M, int N, int K);
void pow_cuda_device(const float *d_input, float exponent, float *d_output, int size);
void exp_cuda_device(const float *d_input, float *d_output, int size);
void log_cuda_device(const float *d_input, float *d_output, int size);
void sqrt_cuda_device(const float *d_input, float *d_output, int size);
void sum_cuda_device(const float *d_input, float *d_output, int size);
void max_cuda_device(const float *d_input, float *d_output, int size);
void add_scalar_cuda_device(const float *d_input, float scalar, float *d_output, int size);
void mul_scalar_cuda_device(const float *d_input, float scalar, float *d_output, int size);
void fill_cuda_device(float *d_data, float value, int size);


void relu_backward_cuda_device(const float *d_grad_output, const float *d_input,
                                float *d_grad_input, int size);
void sigmoid_backward_cuda_device(const float *d_grad_output, const float *d_output,
                                  float *d_grad_input, int size);
void tanh_backward_cuda_device(const float *d_grad_output, const float *d_output,
                               float *d_grad_input, int size);
void max_pool2d_forward_cuda_device(const float *d_input, float *d_output, int *d_indices,
                                     int N, int C, int H, int W, int out_h, int out_w,
                                     int k, int stride);
void max_pool2d_backward_cuda_device(const float *d_grad_output, const int *d_indices,
                                      float *d_grad_input, int size);
void avg_pool2d_forward_cuda_device(const float *d_input, float *d_output,
                                     int N, int C, int H, int W, int out_h, int out_w,
                                     int k, int stride);
void avg_pool2d_backward_cuda_device(const float *d_grad_output, float *d_grad_input,
                                      int N, int C, int H, int W, int out_h, int out_w,
                                      int k, int stride);
void batchnorm_forward_cuda_device(const float *d_input, const float *d_mean,
                                   const float *d_var, const float *d_gamma,
                                   const float *d_beta, float *d_output,
                                   int N, int C, int H, int W, float eps);

void batchnorm_training_forward_cuda_device(const float *d_input, float *d_mean,
                                            float *d_var, float *d_running_mean,
                                            float *d_running_var, const float *d_gamma,
                                            const float *d_beta, float *d_output,
                                            int N, int C, int H, int W, float eps,
                                            float momentum);

void batchnorm_backward_cuda_device(const float *d_grad_output, const float *d_input,
                                    const float *d_mean, const float *d_var,
                                    const float *d_gamma, float *d_grad_input,
                                    float *grad_gamma, float *grad_beta,
                                    int N, int C, int H, int W, float eps);

// Conv2D
void im2col_cuda_device(const float* d_im, const int N, const int C, const int H, const int W,
                        const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
                        const int stride_h, const int stride_w, const int out_h, const int out_w,
                        float* d_col);

void col2im_cuda_device(const float* d_col, const int N, const int C, const int H, const int W,
                        const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
                        const int stride_h, const int stride_w, const int out_h, const int out_w,
                        float* d_im);

void add_bias_shared_cuda_device(const float* d_bias, float* d_output, int N, int C, int H, int W);
void bias_backward_cuda_device(const float* d_grad_output, float* d_grad_bias, int N, int C, int H, int W);

void transpose2d_cuda_device(const float* d_in, float* d_out, int rows, int cols);
void permute4d_cuda_device(const float* d_in, float* d_out, int N, int C, int H, int W,
                           int d0, int d1, int d2, int d3);

// Optimizers
void sgd_update_cuda_device(float* d_params, float* d_velocity, const float* d_grads,
                            float lr, float momentum, float weight_decay, bool nesterov, int size);

void adam_update_cuda_device(float* d_params, float* d_m, float* d_v, const float* d_grads,
                             float lr, float beta1, float beta2, float eps, float weight_decay, int t, int size);

// Dropout
void dropout_cuda_device(const float* d_input, const float* d_mask, float* d_output, float scale, int size);
void dropout_backward_cuda_device(const float* d_grad_output, const float* d_mask, float* d_grad_input, float scale, int size);

// Loss
void log_softmax_cuda_device(const float* d_input, float* d_output, int N, int C);
void nll_loss_cuda_device(const float* d_log_probs, const int* d_targets, float* d_losses, int N, int C);
void nll_loss_backward_cuda_device(const int* d_targets, float* d_grad_input, int N, int C);




// ============================================================
// High-level host wrappers: host vector in -> compute on GPU -> host vector out
// These handle all device memory allocation and transfers internally.
// ============================================================
void add_cuda_host(const std::vector<float> &a, const std::vector<float> &b,
                   std::vector<float> &out, int size);
void mul_cuda_host(const std::vector<float> &a, const std::vector<float> &b,
                   std::vector<float> &out, int size);
void matmul_cuda_host(const std::vector<float> &a, const std::vector<float> &b,
                      std::vector<float> &out, int M, int N, int K);
void relu_cuda_host(const std::vector<float> &input,
                    std::vector<float> &output, int size);
void sigmoid_cuda_host(const std::vector<float> &input,
                       std::vector<float> &output, int size);
void tanh_cuda_host(const std::vector<float> &input,
                    std::vector<float> &output, int size);

void relu_backward_cuda_host(const std::vector<float> &grad_output,
                             const std::vector<float> &input,
                             std::vector<float> &grad_input);
void sigmoid_backward_cuda_host(const std::vector<float> &grad_output,
                                const std::vector<float> &output,
                                std::vector<float> &grad_input);
void tanh_backward_cuda_host(const std::vector<float> &grad_output,
                             const std::vector<float> &output,
                             std::vector<float> &grad_input);

void max_pool2d_forward_cuda_host(const std::vector<float> &input,
                                  std::vector<float> &output,
                                  std::vector<int> &indices, int N, int C,
                                  int H, int W, int k, int stride);
void max_pool2d_backward_cuda_host(const std::vector<float> &grad_output,
                                   const std::vector<int> &indices,
                                   std::vector<float> &grad_input, int N, int C,
                                   int H, int W, int k, int stride);

void avg_pool2d_forward_cuda_host(const std::vector<float> &input,
                                  std::vector<float> &output, int N, int C,
                                  int H, int W, int k, int stride);
void avg_pool2d_backward_cuda_host(const std::vector<float> &grad_output,
                                   std::vector<float> &grad_input, int N, int C,
                                   int H, int W, int k, int stride);

void batchnorm_forward_cuda_host(const std::vector<float> &input,
                                 const std::vector<float> &mean,
                                 const std::vector<float> &var,
                                 const std::vector<float> &gamma,
                                 const std::vector<float> &beta,
                                 std::vector<float> &output, int N, int C,
                                 int H, int W, float eps);
void batchnorm_backward_cuda_host(const std::vector<float> &grad_output,
                                  const std::vector<float> &input,
                                  const std::vector<float> &mean,
                                  const std::vector<float> &var,
                                  const std::vector<float> &gamma,
                                  std::vector<float> &grad_input,
                                  std::vector<float> &grad_gamma,
                                  std::vector<float> &grad_beta, int N, int C,
                                  int H, int W, float eps);

// Check if CUDA is available at runtime
bool is_cuda_available();

} // namespace cuda
} // namespace deepnet

#endif // USE_CUDA
