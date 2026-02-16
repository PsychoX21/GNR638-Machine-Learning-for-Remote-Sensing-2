#include "layers/pooling.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#ifdef USE_CUDA
#include "cuda/cuda_ops.hpp"
#include "cuda/cuda_utils.hpp"
#endif

namespace deepnet {

MaxPool2D::MaxPool2D(int kernel_size, int stride)
    : kernel_size(kernel_size), stride(stride == -1 ? kernel_size : stride) {}

TensorPtr MaxPool2D::forward(const TensorPtr &input) {
  if (input->shape.size() != 4) {
    throw std::runtime_error("MaxPool2D expects 4D input");
  }

  int batch = input->shape[0];
  int channels = input->shape[1];
  int in_h = input->shape[2];
  int in_w = input->shape[3];

  int out_h = (in_h - kernel_size) / stride + 1;
  int out_w = (in_w - kernel_size) / stride + 1;

  // Cache input shape for backward
  input_shape = input->shape;

  auto output = Tensor::zeros({batch, channels, out_h, out_w},
                              true, input->is_cuda);

  // Clear and resize max_indices
  max_indices.resize(batch * channels * out_h * out_w);

#ifdef USE_CUDA
  if (input->is_cuda) {
    int out_size = batch * channels * out_h * out_w;
    size_t idx_bytes = out_size * sizeof(int);
    int *d_indices = (int *)cuda::cuda_malloc(idx_bytes);
    
    cuda::max_pool2d_forward_cuda_device(input->data_ptr(), output->data_ptr(), d_indices,
                                         batch, channels, in_h, in_w, out_h, out_w, kernel_size, stride);
    cudaDeviceSynchronize();
    
    // Copy indices back to CPU
    cuda::cuda_memcpy_device_to_host(max_indices.data(), d_indices, idx_bytes);
    cuda::cuda_free(d_indices);
    return output;
  }
#endif

  input->sync_to_cpu();
  #pragma omp parallel for
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channels; ++c) {
      for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
          float max_val = -std::numeric_limits<float>::infinity();
          int max_idx = -1;
          bool found_valid = false;

          for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
              int ih = oh * stride + kh;
              int iw = ow * stride + kw;
              if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int in_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
                if (input->data[in_idx] > max_val) {
                  max_val = input->data[in_idx];
                  max_idx = in_idx;
                  found_valid = true;
                }
              }
            }
          }
          
          // Handle edge case: if no valid pixels found, use first pixel of channel
          if (!found_valid) {
            max_idx = ((b * channels + c) * in_h + 0) * in_w + 0;
            max_val = input->data[max_idx];
          }

          int out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
          output->data[out_idx] = max_val;
          max_indices[out_idx] = max_idx;
        }
      }
    }
  }

  return output;
}

TensorPtr MaxPool2D::backward(const TensorPtr &grad_output) {
  // Route gradients to the max elements
  auto grad_input = Tensor::zeros(input_shape, false, grad_output->is_cuda);

#ifdef USE_CUDA
  if (grad_output->is_cuda) {
    int size = grad_output->numel();
    size_t idx_bytes = max_indices.size() * sizeof(int);
    int *d_indices = (int *)cuda::cuda_malloc(idx_bytes);
    
    // Copy indices to device
    cuda::cuda_memcpy_host_to_device(d_indices, max_indices.data(), idx_bytes);
    
    cuda::max_pool2d_backward_cuda_device(grad_output->data_ptr(), d_indices,
                                         grad_input->data_ptr(), size);
    cudaDeviceSynchronize();
    
    cuda::cuda_free(d_indices);
    return grad_input;
  }
#endif

  grad_output->sync_to_cpu();
  for (size_t i = 0; i < max_indices.size(); ++i) {
    int max_idx = max_indices[i];
    if (max_idx >= 0 && max_idx < (int)grad_input->data.size()) {
      grad_input->data[max_idx] += grad_output->data[i];
    }
  }

  return grad_input;
}

// AvgPool2D Implementation
AvgPool2D::AvgPool2D(int kernel_size, int stride)
    : kernel_size(kernel_size), stride(stride == -1 ? kernel_size : stride) {}

TensorPtr AvgPool2D::forward(const TensorPtr &input) {
  if (input->shape.size() != 4) {
    throw std::runtime_error("AvgPool2D expects 4D input");
  }

  int batch = input->shape[0];
  int channels = input->shape[1];
  int in_h = input->shape[2];
  int in_w = input->shape[3];

  int out_h = (in_h - kernel_size) / stride + 1;
  int out_w = (in_w - kernel_size) / stride + 1;

  input_shape = input->shape;

  auto output = Tensor::zeros({batch, channels, out_h, out_w},
                              true, input->is_cuda);

#ifdef USE_CUDA
  if (input->is_cuda) {
    cuda::avg_pool2d_forward_cuda_device(input->data_ptr(), output->data_ptr(),
                                         batch, channels, in_h, in_w, out_h, out_w, kernel_size, stride);
    cudaDeviceSynchronize();
    return output;
  }
#endif

  float pool_size = static_cast<float>(kernel_size * kernel_size);

  #pragma omp parallel for
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channels; ++c) {
      for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
          float sum = 0.0f;

          int valid_count = 0;
          for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
              int ih = oh * stride + kh;
              int iw = ow * stride + kw;
              if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int in_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
                sum += input->data[in_idx];
                valid_count++;
              }
            }
          }
          
          float actual_pool_size = (valid_count > 0) ? static_cast<float>(valid_count) : pool_size;

          int out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
          output->data[out_idx] = (valid_count > 0) ? sum / actual_pool_size : 0.0f;
        }
      }
    }
  }

  return output;
}

TensorPtr AvgPool2D::backward(const TensorPtr &grad_output) {
  int batch = input_shape[0];
  int channels = input_shape[1];
  int in_h = input_shape[2];
  int in_w = input_shape[3];

  int out_h = grad_output->shape[2];
  int out_w = grad_output->shape[3];

  auto grad_input = Tensor::zeros(input_shape, false, grad_output->is_cuda);

#ifdef USE_CUDA
  if (grad_output->is_cuda) {
    cuda::avg_pool2d_backward_cuda_device(grad_output->data_ptr(), grad_input->data_ptr(),
                                          batch, channels, in_h, in_w, out_h, out_w, kernel_size, stride);
    cudaDeviceSynchronize();
    return grad_input;
  }
#endif

  grad_output->sync_to_cpu();
  float pool_size = static_cast<float>(kernel_size * kernel_size);

  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channels; ++c) {
      for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
          int out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
          float grad_val = grad_output->data[out_idx] / pool_size;

          for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
              int ih = oh * stride + kh;
              int iw = ow * stride + kw;
              if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int in_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
                grad_input->data[in_idx] += grad_val;
              }
            }
          }
        }
      }
    }
  }

  return grad_input;
}
} // namespace deepnet
