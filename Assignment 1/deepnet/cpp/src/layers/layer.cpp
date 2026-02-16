#include "layers/layer.hpp"
#include <cmath>
#include <random>
#include <iostream>
#include <stdexcept>
#include <ctime>
#ifdef USE_CUDA
#include "cuda/cuda_ops.hpp"
#endif

namespace deepnet {

// Conv2D Implementation
Conv2D::Conv2D(int in_channels, int out_channels, int kernel_size, int stride,
               int padding, bool bias)
    : in_channels(in_channels), out_channels(out_channels),
      kernel_size(kernel_size), stride(stride), padding(padding),
      use_bias(bias) {

  // Initialize weights with He initialization
  float std_val = std::sqrt(2.0f / (in_channels * kernel_size * kernel_size));

  weight = Tensor::randn({out_channels, in_channels, kernel_size, kernel_size},
                         0.0f, std_val, true, false);

  if (use_bias) {
    bias_ = Tensor::zeros({out_channels}, true, false);
  }
}

TensorPtr Conv2D::forward(const TensorPtr &input) {
  if (!input) throw std::runtime_error("Conv2D::forward: input is null");
  // Input shape: [batch, in_channels, height, width]
  if (input->shape.size() != 4) {
    throw std::runtime_error("Conv2D expects 4D input");
  }

  // Cache input for backward
  last_input = input;

  int batch = input->shape[0];
  int in_h = input->shape[2];
  int in_w = input->shape[3];

  int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
  int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

  // 1. Im2Col: (N * out_h * out_w, C * KH * KW)
  last_col = input->im2col(kernel_size, stride, padding);

  // 2. Reshape weights: (OutC, InC * KH * KW)
  int K = in_channels * kernel_size * kernel_size;
  auto weight_flat = weight->reshape({out_channels, K});

  // 3. MatMul: (N * OH * OW, K) @ (K, OutC) -> (N * OH * OW, OutC)
  // We compute col @ weight_flat.T
  auto output_flat = last_col->matmul(weight_flat->transpose(0, 1));

  // 4. Reshape to (N, OH, OW, OutC) and Permute to (N, OutC, OH, OW)
  auto reshaped = output_flat->reshape({batch, out_h, out_w, out_channels});
  auto output = reshaped->permute({0, 3, 1, 2});

  // 5. Add bias
  if (use_bias) {
#ifdef USE_CUDA
    if (output->is_cuda) {
        cuda::add_bias_shared_cuda_device(bias_->data_ptr(), output->data_ptr(),
                                        batch, out_channels, out_h, out_w);
    } else 
#endif
    {
    #pragma omp parallel for
    for (int b = 0; b < batch; ++b) {
      for (int c = 0; c < out_channels; ++c) {
        float b_val = bias_->data[c];
        for (int oh = 0; oh < out_h; ++oh) {
          for (int ow = 0; ow < out_w; ++ow) {
             int idx = ((b * out_channels + c) * out_h + oh) * out_w + ow;
             output->data[idx] += b_val;
          }
        }
      }
    }
    }
  }

  return output;
}

TensorPtr Conv2D::backward(const TensorPtr &grad_output) {
  if (!last_input || !last_col) {
    throw std::runtime_error("Conv2D::backward called without forward");
  }

  int batch = last_input->shape[0];
  int in_h = last_input->shape[2];
  int in_w = last_input->shape[3];
  int out_h = grad_output->shape[2];
  int out_w = grad_output->shape[3];
  int K = in_channels * kernel_size * kernel_size;

  // Ensure grad buffers are allocated
  if (weight->grad.size() != weight->data.size()) {
    weight->grad.resize(weight->data.size(), 0.0f);
  }
  if (use_bias && bias_->grad.size() != bias_->data.size()) {
    bias_->grad.resize(bias_->data.size(), 0.0f);
  }

  // 1. Prepare grad_output: (N, OutC, OH, OW) -> (N, OH, OW, OutC) -> (M, OutC)
  auto grad_permuted = grad_output->permute({0, 2, 3, 1});
  auto grad_flat = grad_permuted->reshape({batch * out_h * out_w, out_channels});

  // 2. Gradient w.r.t Bias: sum over batch and spatial dimensions
  if (use_bias) {
#ifdef USE_CUDA
    if (grad_output->is_cuda) {
       // Allocate device memory for grad if needed
       if (!bias_->d_grad) bias_->grad_ptr();
       cuda::bias_backward_cuda_device(grad_output->data_ptr(), bias_->d_grad,
                                     batch, out_channels, out_h, out_w);
    } else
#endif
    {
    int M = batch * out_h * out_w;
    #pragma omp parallel for
    for (int c = 0; c < out_channels; ++c) {
      float sum = 0.0f;
      for (int i = 0; i < M; ++i) {
        sum += grad_flat->data[i * out_channels + c];
      }
      bias_->grad[c] += sum;
    }
    }
  }

  // 3. Gradient w.r.t Weights: dL/dW = dL/dY.T @ X_col
  auto grad_weight_flat = grad_flat->transpose(0, 1)->matmul(last_col);
  weight->accumulate_grad(grad_weight_flat);

  // 4. Gradient w.r.t Input: dL/dX_col = dL/dY @ W
  auto weight_flat = weight->reshape({out_channels, K});
  auto grad_col = grad_flat->matmul(weight_flat);

  // 5. Col2Im
  return grad_col->col2im(last_input->shape, kernel_size, stride, padding);
}

std::vector<TensorPtr> Conv2D::parameters() {
  if (use_bias) {
    return {weight, bias_};
  }
  return {weight};
}

// Linear Implementation
Linear::Linear(int in_features, int out_features, bool bias)
    : in_features(in_features), out_features(out_features), use_bias(bias) {

  // Xavier initialization
  float limit = std::sqrt(6.0f / (in_features + out_features));
  weight = Tensor::randn({out_features, in_features}, 0.0f, limit, true, false);

  if (use_bias) {
    bias_ = Tensor::zeros({out_features}, true, false);
  }
}

TensorPtr Linear::forward(const TensorPtr &input) {
  if (!input) throw std::runtime_error("Linear::forward: input is null");
  // Input: [N, In]
  TensorPtr x = input;
  bool is_1d = (input->shape.size() == 1);
  if (is_1d) {
    x = input->reshape({1, input->shape[0]});
  }

  if (x->shape.size() != 2 || x->shape[1] != in_features) {
    throw std::runtime_error("Linear layer input size mismatch");
  }
  last_input = x;
  auto output = x->matmul(weight->transpose(0, 1));

  if (use_bias) {
#ifdef USE_CUDA
    if (output->is_cuda) {
        // Linear bias addition is same as Conv2D 1x1: (N, C, 1, 1) or (N, C)
        // Here output is (N, Out). We can reuse add_bias_shared if we treat it as (N, Out, 1, 1)
        cuda::add_bias_shared_cuda_device(bias_->data_ptr(), output->data_ptr(),
                                        x->shape[0], out_features, 1, 1);
    } else
#endif
    {
    int batch = x->shape[0];
    #pragma omp parallel for
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < out_features; ++j) {
        output->data[i * out_features + j] += bias_->data[j];
      }
    }
    }
  }

  if (is_1d) {
    output = output->reshape({out_features});
  }

  return output;
}

TensorPtr Linear::backward(const TensorPtr &grad_output) {
  if (!last_input) {
    throw std::runtime_error("Linear::backward called without forward");
  }
  
  TensorPtr grad = grad_output;
  if (grad->shape.size() == 1) {
    grad = grad->reshape({1, (int)grad->data.size()});
  }
  
  // Ensure buffers
  if (weight->grad.size() != weight->data.size()) {
    weight->grad.resize(weight->data.size(), 0.0f);
  }
  if (use_bias && bias_->grad.size() != bias_->data.size()) {
    bias_->grad.resize(bias_->data.size(), 0.0f);
  }

  auto grad_weight = grad->transpose(0, 1)->matmul(last_input);
  weight->accumulate_grad(grad_weight);

  if (use_bias) {
#ifdef USE_CUDA
    if (grad->is_cuda) {
        if (!bias_->d_grad) bias_->grad_ptr();
        // Reuse bias_backward_cuda_device with H=1, W=1
        cuda::bias_backward_cuda_device(grad->data_ptr(), bias_->d_grad,
                                      grad->shape[0], out_features, 1, 1);
    } else
#endif
    {
    int batch = grad->shape[0];
    #pragma omp parallel for
    for (int j = 0; j < out_features; ++j) {
      float sum = 0.0f;
      for (int i = 0; i < batch; ++i) {
        sum += grad->data[i * out_features + j];
      }
      bias_->grad[j] += sum;
    }
    }
  }

  auto grad_input = grad->matmul(weight);
  return grad_input;
}

std::vector<TensorPtr> Linear::parameters() {
  if (use_bias) {
    return {weight, bias_};
  }
  return {weight};
}

// ReLU Implementation
TensorPtr ReLU::forward(const TensorPtr &input) {
  if (!input) throw std::runtime_error("ReLU::forward: input is null");
  last_input = input;
  return input->relu();
}

TensorPtr ReLU::backward(const TensorPtr &grad_output) {
  auto grad_input = Tensor::zeros(last_input->shape, false, grad_output->is_cuda);

#ifdef USE_CUDA
  if (grad_output->is_cuda) {
    int size = grad_output->numel();
    cuda::relu_backward_cuda_device(grad_output->data_ptr(), last_input->data_ptr(),
                                    grad_input->data_ptr(), size);
    return grad_input;
  }
#endif

  grad_output->sync_to_cpu();
  last_input->sync_to_cpu();
  #pragma omp parallel for
  for (int i = 0; i < (int)grad_input->data.size(); ++i) {
    grad_input->data[i] = last_input->data[i] > 0 ? grad_output->data[i] : 0.0f;
  }
  return grad_input;
}

// LeakyReLU Implementation
TensorPtr LeakyReLU::forward(const TensorPtr &input) {
  if (!input) throw std::runtime_error("LeakyReLU::forward: input is null");
  last_input = input;
  return input->leaky_relu(negative_slope);
}

TensorPtr LeakyReLU::backward(const TensorPtr &grad_output) {
  auto grad_input = Tensor::zeros(last_input->shape, false, grad_output->is_cuda);
  #pragma omp parallel for
  for (int i = 0; i < (int)grad_input->data.size(); ++i) {
    grad_input->data[i] = last_input->data[i] > 0 ? grad_output->data[i]
                                                    : negative_slope * grad_output->data[i];
  }
  return grad_input;
}

// Tanh Implementation
TensorPtr Tanh::forward(const TensorPtr &input) {
  if (!input) throw std::runtime_error("Tanh::forward: input is null");
  last_output = input->tanh_();
  return last_output;
}

TensorPtr Tanh::backward(const TensorPtr &grad_output) {
  auto grad_input = Tensor::zeros(last_output->shape, false, grad_output->is_cuda);

#ifdef USE_CUDA
  if (grad_output->is_cuda) {
    int size = grad_output->numel();
    cuda::tanh_backward_cuda_device(grad_output->data_ptr(), last_output->data_ptr(),
                                   grad_input->data_ptr(), size);
    return grad_input;
  }
#endif

  grad_output->sync_to_cpu();
  last_output->sync_to_cpu();
  #pragma omp parallel for
  for (int i = 0; i < (int)grad_input->data.size(); ++i) {
    float t = last_output->data[i];
    grad_input->data[i] = grad_output->data[i] * (1.0f - t * t);
  }
  return grad_input;
}

// Sigmoid Implementation
TensorPtr Sigmoid::forward(const TensorPtr &input) {
  if (!input) throw std::runtime_error("Sigmoid::forward: input is null");
  last_output = input->sigmoid();
  return last_output;
}

TensorPtr Sigmoid::backward(const TensorPtr &grad_output) {
  auto grad_input = Tensor::zeros(last_output->shape, false, grad_output->is_cuda);

#ifdef USE_CUDA
  if (grad_output->is_cuda) {
    int size = grad_output->numel();
    cuda::sigmoid_backward_cuda_device(grad_output->data_ptr(), last_output->data_ptr(),
                                      grad_input->data_ptr(), size);
    return grad_input;
  }
#endif

  grad_output->sync_to_cpu();
  last_output->sync_to_cpu();
  #pragma omp parallel for
  for (int i = 0; i < (int)grad_input->data.size(); ++i) {
    float s = last_output->data[i];
    grad_input->data[i] = grad_output->data[i] * s * (1.0f - s);
  }
  return grad_input;
}

// Dropout Implementation
TensorPtr Dropout::forward(const TensorPtr &input) {
  if (!input) throw std::runtime_error("Dropout::forward: input is null");
  if (!training) return input;

  float scale = 1.0f / (1.0f - p);

#ifdef USE_CUDA
  if (input->is_cuda) {
      std::vector<float> mask_cpu(input->numel());
      std::bernoulli_distribution dist(1.0f - p);
      auto &gen = deepnet::get_generator();
      for (size_t i = 0; i < mask_cpu.size(); ++i) {
          mask_cpu[i] = dist(gen) ? 1.0f : 0.0f;
      }
      
      mask = Tensor::from_data(mask_cpu, input->shape, false, true);
      
      auto output = Tensor::zeros(input->shape, input->requires_grad, true);
      cuda::dropout_cuda_device(input->data_ptr(), mask->data_ptr(), output->data_ptr(), scale, input->numel());
      return output;
  }
#endif

  // CPU Implementation
  mask = Tensor::zeros(input->shape, false, false);
  std::bernoulli_distribution dist(1.0f - p);
  auto &gen = deepnet::get_generator();
  
  for (int i = 0; i < (int)mask->data.size(); ++i) {
      mask->data[i] = dist(gen) ? 1.0f : 0.0f;
  }
  
  auto output = Tensor::zeros(input->shape, input->requires_grad, false);
  #pragma omp parallel for
  for (int i = 0; i < (int)input->data.size(); ++i) {
      output->data[i] = input->data[i] * mask->data[i] * scale;
  }
  
  return output;
}

TensorPtr Dropout::backward(const TensorPtr &grad_output) {
  if (!training) return grad_output;
  
  auto grad_input = Tensor::zeros(grad_output->shape, false, grad_output->is_cuda);
  float scale = 1.0f / (1.0f - p);

#ifdef USE_CUDA
  if (grad_output->is_cuda) {
      cuda::dropout_backward_cuda_device(grad_output->data_ptr(), mask->data_ptr(), grad_input->data_ptr(), scale, grad_output->numel());
      return grad_input;
  }
#endif

  #pragma omp parallel for
  for (int i = 0; i < (int)grad_input->data.size(); ++i) {
      grad_input->data[i] = grad_output->data[i] * mask->data[i] * scale;
  }
  return grad_input;
}

// Flatten Implementation
TensorPtr Flatten::forward(const TensorPtr &input) {
  original_shape = input->shape;
  return input->flatten(start_dim, end_dim);
}

TensorPtr Flatten::backward(const TensorPtr &grad_output) {
  // Reshape gradient back to original shape
  return grad_output->reshape(original_shape);
}

} // namespace deepnet
