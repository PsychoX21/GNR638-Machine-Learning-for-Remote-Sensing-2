#include "tensor.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

#ifdef USE_CUDA
#include "cuda/cuda_ops.hpp"
#include "cuda/cuda_utils.hpp"
#include <cuda_runtime.h>
#endif

namespace deepnet {

// Global generator for CPU randomness
std::mt19937 &get_generator() {
  static std::mt19937 gen(std::random_device{}());
  return gen;
}

void manual_seed(unsigned int seed) {
  get_generator().seed(seed);
}

// Autograd function implementations
struct AddBackward : public AutogradFunction {
  std::vector<TensorPtr> backward(const TensorPtr &grad_output) override {
    return {grad_output, grad_output};
  }
};

struct MulBackward : public AutogradFunction {
  std::vector<TensorPtr> backward(const TensorPtr &grad_output) override {
    // d/dx (x * y) = y, d/dy (x * y) = x
    return {grad_output->mul(inputs[1]), grad_output->mul(inputs[0])};
  }
};

struct MatMulBackward : public AutogradFunction {
  std::vector<TensorPtr> backward(const TensorPtr &grad_output) override {
    // For C = A @ B:
    // dL/dA = dL/dC @ B^T
    // dL/dB = A^T @ dL/dC
    auto grad_a = grad_output->matmul(inputs[1]->transpose(0, 1));
    auto grad_b = inputs[0]->transpose(0, 1)->matmul(grad_output);
    return {grad_a, grad_b};
  }
};

struct ReLUBackward : public AutogradFunction {
  std::vector<TensorPtr> backward(const TensorPtr &grad_output) override {
    auto grad_input =
        Tensor::zeros(inputs[0]->shape, false, grad_output->is_cuda);
    
#ifdef USE_CUDA
    if (grad_output->is_cuda) {
        cuda::relu_backward_cuda_device(grad_output->data_ptr(), inputs[0]->data_ptr(), 
                                        grad_input->data_ptr(), grad_input->numel());
        return {grad_input};
    }
#endif

    grad_output->sync_to_cpu();
    inputs[0]->sync_to_cpu();
    for (size_t i = 0; i < grad_input->data.size(); ++i) {
      grad_input->data[i] =
          inputs[0]->data[i] > 0 ? grad_output->data[i] : 0.0f;
    }
    return {grad_input};
  }
};

struct ReshapeBackward : public AutogradFunction {
  std::vector<int> input_shape;
  
  ReshapeBackward(const std::vector<int>& shape) : input_shape(shape) {}

  std::vector<TensorPtr> backward(const TensorPtr &grad_output) override {
    // Reshape the gradient back to the input shape
    return {grad_output->reshape(input_shape)};
  }
};

struct SigmoidBackward : public AutogradFunction {
  TensorPtr output_cache;
  std::vector<TensorPtr> backward(const TensorPtr &grad_output) override {
    auto grad_input = Tensor::zeros(inputs[0]->shape, false, grad_output->is_cuda);
#ifdef USE_CUDA
    if (grad_output->is_cuda) {
        cuda::sigmoid_backward_cuda_device(grad_output->data_ptr(), output_cache->data_ptr(), 
                                           grad_input->data_ptr(), grad_input->numel());
        return {grad_input};
    }
#endif
    grad_output->sync_to_cpu();
    output_cache->sync_to_cpu();
    for (size_t i = 0; i < grad_input->data.size(); ++i) {
        float s = output_cache->data[i];
        grad_input->data[i] = grad_output->data[i] * s * (1.0f - s);
    }
    return {grad_input};
  }
};

struct TanhBackward : public AutogradFunction {
  TensorPtr output_cache;
  std::vector<TensorPtr> backward(const TensorPtr &grad_output) override {
    auto grad_input = Tensor::zeros(inputs[0]->shape, false, grad_output->is_cuda);
#ifdef USE_CUDA
    if (grad_output->is_cuda) {
        cuda::tanh_backward_cuda_device(grad_output->data_ptr(), output_cache->data_ptr(), 
                                        grad_input->data_ptr(), grad_input->numel());
        return {grad_input};
    }
#endif
    grad_output->sync_to_cpu();
    output_cache->sync_to_cpu();
    for (size_t i = 0; i < grad_input->data.size(); ++i) {
        float t = output_cache->data[i];
        grad_input->data[i] = grad_output->data[i] * (1.0f - t * t);
    }
    return {grad_input};
  }
};

struct SumBackward : public AutogradFunction {
  std::vector<int> input_shape;
  
  SumBackward(const std::vector<int>& shape) : input_shape(shape) {}

  std::vector<TensorPtr> backward(const TensorPtr &grad_output) override {
    // For now, let's assume global sum, so grad_output is 1 element.
    
    // Let's copy grad_val to host.
    TensorPtr grad = grad_output;
    if (grad->is_cuda) {
        grad = grad->clone(); // safety
        grad->sync_to_cpu();
    }
    float g = grad->data[0];
    
    auto out = Tensor::ones(input_shape, false, grad_output->is_cuda);
    // multiply by g
    if (out->is_cuda) {
        // use mul_scalar because it's efficient
        // But wait, mul_scalar modifies in place? No, returns new.
        return {out->mul_scalar(g)};
    } else {
        for(auto& v : out->data) v *= g;
        return {out};
    }
  }
};

struct PowBackward : public AutogradFunction {
  float exponent;
  
  PowBackward(float exp) : exponent(exp) {}
  
  std::vector<TensorPtr> backward(const TensorPtr &grad_output) override {
    // d/dx (x^n) = n * x^(n-1)
    // grad = grad_output * n * inputs[0]^(n-1)
    auto input = inputs[0];
    auto pow_res = input->pow(exponent - 1.0f);
    auto scaled = pow_res->mul_scalar(exponent);
    return {grad_output->mul(scaled)};
  }
};

struct MeanBackward : public AutogradFunction {
  std::vector<int> input_shape;
  int num_elements;
  
  MeanBackward(const std::vector<int>& shape, int n) : input_shape(shape), num_elements(n) {}
  
  std::vector<TensorPtr> backward(const TensorPtr &grad_output) override {
    // Mean = Sum / N
    // Grad = grad_output / N * Ones
    TensorPtr grad = grad_output;
    if (grad->is_cuda) {
        grad = grad->clone();
        grad->sync_to_cpu();
    }
    float g = grad->data[0];
    
    auto out = Tensor::ones(input_shape, false, grad_output->is_cuda);
    float scale = g / num_elements;
    
    if (out->is_cuda) {
        return {out->mul_scalar(scale)};
    } else {
        for(auto& v : out->data) v *= scale;
        return {out};
    }
  }
};


// Tensor destructor
Tensor::~Tensor() {
#ifdef USE_CUDA
  free_device_memory();
#endif
}

// Tensor constructors
Tensor::Tensor() : requires_grad(false), is_cuda(false)
#ifdef USE_CUDA
  , d_data(nullptr), d_grad(nullptr), cpu_dirty(false), cuda_dirty(false)
#endif
{}

Tensor::Tensor(const std::vector<int> &shape, bool requires_grad, bool cuda)
    : shape(shape), requires_grad(requires_grad), is_cuda(cuda)
#ifdef USE_CUDA
    , d_data(nullptr), d_grad(nullptr), cpu_dirty(false), cuda_dirty(false)
#endif
{
  int total_size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  
#ifdef USE_CUDA
  if (cuda) {
    allocate_device_memory();
    // Initialize device memory to zero
    if (d_data) {
      cuda::cuda_memset(d_data, 0, total_size * sizeof(float));
    }
    if (requires_grad && d_grad) {
      cuda::cuda_memset(d_grad, 0, total_size * sizeof(float));
    }
    cuda_dirty = true; // GPU has zeroes, CPU is empty/stale
    cpu_dirty = false;
  } else {
    data.resize(total_size, 0.0f);
    if (requires_grad) {
      grad.resize(total_size, 0.0f);
    }
    cpu_dirty = true; // CPU has zeroes, GPU is null/stale
    cuda_dirty = false;
  }
#else
  data.resize(total_size, 0.0f);
  if (requires_grad) {
    grad.resize(total_size, 0.0f);
  }
#endif
  compute_strides();
}

Tensor::Tensor(const std::vector<float> &data, const std::vector<int> &shape,
               bool requires_grad, bool cuda)
    : shape(shape), requires_grad(requires_grad), is_cuda(cuda)
#ifdef USE_CUDA
    , d_data(nullptr), d_grad(nullptr), cpu_dirty(false), cuda_dirty(false)
#endif
{
#ifdef USE_CUDA
  if (cuda) {
    allocate_device_memory();
    // Copy data to device
    if (d_data && !data.empty()) {
      cuda::cuda_memcpy_host_to_device(d_data, data.data(), data.size() * sizeof(float));
    }
    if (requires_grad) {
      if (d_grad) {
        cuda::cuda_memset(d_grad, 0, data.size() * sizeof(float));
      }
    }
    cuda_dirty = true; // GPU has fresh data
    cpu_dirty = false;
  } else {
    this->data = data;
    if (requires_grad) {
      grad.resize(data.size(), 0.0f);
    }
    cpu_dirty = true;
    cuda_dirty = false;
  }
#else
  this->data = data;
  if (requires_grad) {
    grad.resize(data.size(), 0.0f);
  }
#endif
  compute_strides();
}

// Factory methods
TensorPtr Tensor::zeros(const std::vector<int> &shape, bool requires_grad,
                        bool cuda) {
  return std::make_shared<Tensor>(shape, requires_grad, cuda);
}

TensorPtr Tensor::ones(const std::vector<int> &shape, bool requires_grad,
                       bool cuda) {
  auto tensor = std::make_shared<Tensor>(shape, requires_grad, false);
  std::fill(tensor->data.begin(), tensor->data.end(), 1.0f);
  if (cuda) tensor->cuda();
  return tensor;
}

TensorPtr Tensor::randn(const std::vector<int> &shape, float mean, float std,
                        bool requires_grad, bool cuda) {
  auto tensor = std::make_shared<Tensor>(shape, requires_grad, false);
  std::normal_distribution<float> dist(mean, std);
  auto &gen = get_generator();
  for (auto &val : tensor->data) {
    val = dist(gen);
  }
  if (cuda) tensor->cuda();
  return tensor;
}

TensorPtr Tensor::from_data(const std::vector<float> &data,
                            const std::vector<int> &shape, bool requires_grad,
                            bool cuda) {
  if (cuda) {
      auto tensor = std::make_shared<Tensor>(data, shape, requires_grad, false);
      tensor->cuda();
      return tensor;
  }
  return std::make_shared<Tensor>(data, shape, requires_grad, false);
}


// Shape operations
int Tensor::size() const { return shape.empty() ? 0 : shape[0]; }

int Tensor::size(int dim) const {
  if (dim < 0)
    dim += static_cast<int>(shape.size());
  return shape[dim];
}

int Tensor::numel() const {
  if (shape.empty()) return 0;
  int n = 1;
  for (int s : shape) n *= s;
  return n;
}

int Tensor::ndim() const { return static_cast<int>(shape.size()); }

void Tensor::compute_strides() {
  strides.resize(shape.size());
  int stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
}

TensorPtr Tensor::detach() {
  auto output = std::make_shared<Tensor>(shape, false, is_cuda);
#ifdef USE_CUDA
  if (is_cuda) {
    cuda::cuda_memcpy_device_to_device(output->data_ptr(), data_ptr(),
                                     numel() * sizeof(float));
    return output;
  }
#endif
  output->data = data;
  return output;
}

void Tensor::accumulate_grad(const TensorPtr &grad_in) {
    if (!grad_in) return;
    if (grad_in->numel() != numel()) {
        throw std::runtime_error("accumulate_grad: shape mismatch");
    }

#ifdef USE_CUDA
    if (is_cuda) {
        if (!d_grad) grad_ptr(); // Ensure grad is allocated
        cuda::add_inplace_cuda_device(d_grad, grad_in->data_ptr(), numel());
        cuda_dirty = true; // Mark as modified on GPU
        return;
    }
#endif

    // CPU path
    if (grad.size() != data.size()) {
        grad.resize(data.size(), 0.0f);
    }
    grad_in->sync_to_cpu();
    for (size_t i = 0; i < grad.size(); ++i) {
        grad[i] += grad_in->data[i];
    }
}

TensorPtr Tensor::reshape(const std::vector<int> &new_shape) {
  TensorPtr output;
#ifdef USE_CUDA
  if (is_cuda) {
    output = Tensor::zeros(new_shape, requires_grad, true);
    // Copy data
    cuda::cuda_memcpy_device_to_device(output->data_ptr(), data_ptr(), numel() * sizeof(float));
    // Copy grad if needed
    if (requires_grad && d_grad) {
        cuda::cuda_memcpy_device_to_device(output->grad_ptr(), grad_ptr(), numel() * sizeof(float));
    }
  } else 
#endif
  {
    output = Tensor::from_data(data, new_shape, requires_grad, is_cuda);
  }

  if (requires_grad) {
    auto grad_fn = std::make_shared<ReshapeBackward>(shape);
    grad_fn->inputs = {shared_from_this()};
    output->grad_fn = grad_fn;
  }
  return output;
}

TensorPtr Tensor::view(const std::vector<int> &new_shape) {
  return reshape(new_shape);
}

TensorPtr Tensor::flatten(int start_dim, int end_dim) {
  if (end_dim == -1)
    end_dim = static_cast<int>(shape.size()) - 1;

  std::vector<int> new_shape;
  int flat_size = 1;

  for (int i = 0; i < start_dim; ++i) {
    new_shape.push_back(shape[i]);
  }

  for (int i = start_dim; i <= end_dim; ++i) {
    flat_size *= shape[i];
  }
  new_shape.push_back(flat_size);

  for (size_t i = end_dim + 1; i < shape.size(); ++i) {
    new_shape.push_back(shape[i]);
  }

  return reshape(new_shape);
}

// Im2Col implementation
TensorPtr Tensor::im2col(int kernel_size, int stride, int padding) {
  if (ndim() != 4) {
    throw std::runtime_error("im2col expects 4D input (N, C, H, W)");
  }

  int N = shape[0];
  int C = shape[1];
  int H = shape[2];
  int W = shape[3];

  int out_h = (H + 2 * padding - kernel_size) / stride + 1;
  int out_w = (W + 2 * padding - kernel_size) / stride + 1;

  int M = N * out_h * out_w;
  int K = C * kernel_size * kernel_size;

  auto output = Tensor::zeros({M, K}, requires_grad, is_cuda);

#ifdef USE_CUDA
  if (is_cuda) {
    cuda::im2col_cuda_device(data_ptr(), N, C, H, W, kernel_size, kernel_size,
                             padding, padding, stride, stride, out_h, out_w,
                             output->data_ptr());
    return output;
  }
#endif

  #pragma omp parallel for
  for (int n = 0; n < N; ++n) {
    for (int oh = 0; oh < out_h; ++oh) {
      for (int ow = 0; ow < out_w; ++ow) {
        int patch_idx = (n * out_h + oh) * out_w + ow;
        
        for (int c = 0; c < C; ++c) {
          for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
              int ih = oh * stride - padding + kh;
              int iw = ow * stride - padding + kw;

              int k_idx = (c * kernel_size + kh) * kernel_size + kw;
              
              if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                int in_idx = ((n * C + c) * H + ih) * W + iw;
                output->data[patch_idx * K + k_idx] = data[in_idx];
              } else {
                output->data[patch_idx * K + k_idx] = 0.0f;
              }
            }
          }
        }
      }
    }
  }

  return output;
}

TensorPtr Tensor::col2im(const std::vector<int> &output_shape, int kernel_size,
                         int stride, int padding) {
  int N = output_shape[0];
  int C = output_shape[1];
  int H = output_shape[2];
  int W = output_shape[3];

  int out_h = (H + 2 * padding - kernel_size) / stride + 1;
  int out_w = (W + 2 * padding - kernel_size) / stride + 1;
  int K = C * kernel_size * kernel_size;

  auto output = Tensor::zeros(output_shape, false, is_cuda);

#ifdef USE_CUDA
  if (is_cuda) {
      cuda::col2im_cuda_device(data_ptr(), output_shape[0], output_shape[1],
                               output_shape[2], output_shape[3], kernel_size, kernel_size,
                               padding, padding, stride, stride, out_h, out_w,
                               output->data_ptr());
      return output;
  }
#endif

  #pragma omp parallel for
  for (int n = 0; n < N; ++n) {
    for (int oh = 0; oh < out_h; ++oh) {
      for (int ow = 0; ow < out_w; ++ow) {
        int patch_idx = (n * out_h + oh) * out_w + ow;

        for (int c = 0; c < C; ++c) {
          for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
              int ih = oh * stride - padding + kh;
              int iw = ow * stride - padding + kw;

              if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                int k_idx = (c * kernel_size + kh) * kernel_size + kw;
                int in_idx = ((n * C + c) * H + ih) * W + iw;
                output->data[in_idx] += data[patch_idx * K + k_idx];
              }
            }
          }
        }
      }
    }
  }
  
  return output;
}

// Element-wise operations
TensorPtr Tensor::add(const TensorPtr &other) {
  check_shape_compatible(other);
  if (is_cuda != other->is_cuda) {
      throw std::runtime_error("Tensor::add: Device mismatch (CUDA vs CPU)");
  }

  auto output =
      Tensor::zeros(shape, requires_grad || other->requires_grad, is_cuda);

  // CUDA path
#ifdef USE_CUDA
  if (is_cuda) {
    cuda::add_cuda_device(data_ptr(), other->data_ptr(), output->data_ptr(), (int)numel());
  } else
#endif
  {
    // CPU path
    #pragma omp parallel for
    for (int i = 0; i < (int)data.size(); ++i) {
      output->data[i] = data[i] + other->data[i];
    }
  }

  if (output->requires_grad) {
    auto grad_fn = std::make_shared<AddBackward>();
    grad_fn->inputs = {shared_from_this(), other};
    output->grad_fn = grad_fn;
  }

  return output;
}

TensorPtr Tensor::mul(const TensorPtr &other) {
  check_shape_compatible(other);
  if (is_cuda != other->is_cuda) {
      throw std::runtime_error("Tensor::mul: Device mismatch (CUDA vs CPU)");
  }

  auto output =
      Tensor::zeros(shape, requires_grad || other->requires_grad, is_cuda);

  // CUDA path
#ifdef USE_CUDA
  if (is_cuda) {
    cuda::mul_cuda_device(data_ptr(), other->data_ptr(), output->data_ptr(), (int)numel());
  } else
#endif
  {
    // CPU path
    #pragma omp parallel for
    for (int i = 0; i < (int)data.size(); ++i) {
      output->data[i] = data[i] * other->data[i];
    }
  }

  if (output->requires_grad) {
    auto grad_fn = std::make_shared<MulBackward>();
    grad_fn->inputs = {shared_from_this(), other};
    output->grad_fn = grad_fn;
  }

  return output;
}

TensorPtr Tensor::sub(const TensorPtr &other) {
  // Direct CUDA implementation for sub to avoid mul_scalar + add Overhead
  if (shape != other->shape) {
      throw std::runtime_error("sub shape mismatch");
  }
  if (is_cuda != other->is_cuda) {
      throw std::runtime_error("Tensor::sub: Device mismatch (CUDA vs CPU)");
  }

  auto output =
      Tensor::zeros(shape, requires_grad || other->requires_grad, is_cuda);

#ifdef USE_CUDA
  if (is_cuda) {
    cuda::sub_cuda_device(data_ptr(), other->data_ptr(), output->data_ptr(), (int)numel());
  } else
#endif
  {
      auto neg_other = other->mul_scalar(-1.0f);
      return add(neg_other);
  }
 
  if (output->requires_grad) {
      // SubBackward could be implemented similar to AddBackward but with -1 for second arg
      // For now, reusing existing structure via add(neg) logic for CPU might be safer for gradients,
      // but if we want full GPU speed we should implement SubBackward or just keep the CPU fallback logic for autograd construction?
      // Actually, if we do sub directly on GPU, we need to set up the grad_fn correctly.
      // Since 'sub' is just 'add' with a negation, the grad_fn logic in 'add(neg_other)' handles it.
      // But here we are doing direct sub. 
      // Let's implement SubBackward in tensor.cpp or just stick to add(neg) if the perf gain is minimal on sub?
      // Sub is basic, let's just use the add(neg_other) logic for now? 
      // Wait, I updated 'sub' to use device pointer, so I should handle grad_fn.
      // Easiest way: re-implement SubBackward or...
      // Actually, let's revert to `add(neg_other)` logic for `sub` but ensure `mul_scalar` is efficient?
      // `mul_scalar` uses `clone` and loop.
      // Let's stick to the high-performance kernel for forward pass.
      
      // We need a SubBackward.
      struct SubBackward : public AutogradFunction {
          std::vector<TensorPtr> backward(const TensorPtr &grad_output) override {
            return {grad_output, grad_output->mul_scalar(-1.0f)};
          }
      };
      auto grad_fn = std::make_shared<SubBackward>();
      grad_fn->inputs = {shared_from_this(), other};
      output->grad_fn = grad_fn;
  }
  return output;
}

TensorPtr Tensor::div(const TensorPtr &other) {
  if (shape != other->shape) {
     throw std::runtime_error("div shape mismatch");
  }
  if (is_cuda != other->is_cuda) {
      throw std::runtime_error("Tensor::div: Device mismatch (CUDA vs CPU)");
  }

  auto output =
      Tensor::zeros(shape, requires_grad || other->requires_grad, is_cuda);

#ifdef USE_CUDA
  if (is_cuda) {
    cuda::div_cuda_device(data_ptr(), other->data_ptr(), output->data_ptr(), (int)numel());
  } else
#endif
  {
    #pragma omp parallel for
    for (int i = 0; i < (int)data.size(); ++i) {
      output->data[i] = data[i] / (other->data[i] + 1e-8f);
    }
  }
  return output;
}

TensorPtr Tensor::add_scalar(float scalar) {
  auto output = clone();
  
#ifdef USE_CUDA
  if (is_cuda) {
    cuda::add_scalar_cuda_device(data_ptr(), scalar, output->data_ptr(), (int)numel());
    return output;
  }
#endif

  for (auto &val : output->data) {
    val += scalar;
  }
  return output;
}

TensorPtr Tensor::mul_scalar(float scalar) {
  auto output = clone();
  output->requires_grad = requires_grad;
  
#ifdef USE_CUDA
  if (is_cuda) {
    cuda::mul_scalar_cuda_device(data_ptr(), scalar, output->data_ptr(), (int)numel());
    
    if (requires_grad) {
        // Init grad buffer
        if (output->d_grad == nullptr) {
            output->grad_ptr(); // allocate
            cuda::cuda_memset(output->d_grad, 0, numel() * sizeof(float));
        }
    }
    return output;
  }
#endif

  for (auto &val : output->data) {
    val *= scalar;
  }
  if (requires_grad) {
    output->grad.resize(data.size(), 0.0f);
  }
  return output;
}

// Operators
TensorPtr Tensor::operator+(const TensorPtr &other) { return add(other); }
TensorPtr Tensor::operator-(const TensorPtr &other) { return sub(other); }
TensorPtr Tensor::operator*(const TensorPtr &other) { return mul(other); }
TensorPtr Tensor::operator/(const TensorPtr &other) { return div(other); }

// Matrix operations
TensorPtr Tensor::matmul(const TensorPtr &other) {
  if (shape.size() != 2 || other->shape.size() != 2) {
    throw std::runtime_error("matmul requires 2D tensors");
  }
  if (shape[1] != other->shape[0]) {
    throw std::runtime_error("matmul shape mismatch");
  }
  if (is_cuda != other->is_cuda) {
      throw std::runtime_error("Tensor::matmul: Device mismatch (CUDA vs CPU)");
  }

  int M = shape[0];
  int K = shape[1];
  int N = other->shape[1];

  auto output =
      Tensor::zeros({M, N}, requires_grad || other->requires_grad, is_cuda);

  // CUDA path
#ifdef USE_CUDA
  if (is_cuda) {
    cuda::matmul_cuda_device(data_ptr(), other->data_ptr(), output->data_ptr(), M, N, K);
  } else
#endif
  {
    // CPU path
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
          sum += data[i * K + k] * other->data[k * N + j];
        }
        output->data[i * N + j] = sum;
      }
    }
  }

  if (output->requires_grad) {
    auto grad_fn = std::make_shared<MatMulBackward>();
    grad_fn->inputs = {shared_from_this(), other};
    output->grad_fn = grad_fn;
  }

  return output;
}

TensorPtr Tensor::mm(const TensorPtr &other) { return matmul(other); }

TensorPtr Tensor::transpose(int dim0, int dim1) {
  std::vector<int> new_shape = shape;
  std::swap(new_shape[dim0], new_shape[dim1]);

  auto output = Tensor::zeros(new_shape, requires_grad, is_cuda);

#ifdef USE_CUDA
  if (is_cuda && shape.size() == 2 && dim0 == 0 && dim1 == 1) {
    cuda::transpose2d_cuda_device(data_ptr(), output->data_ptr(), shape[0], shape[1]);
    return output;
  }
#endif

  // Fallback to CPU if not CUDA 2D or other cases
  if (is_cuda) sync_to_cpu();

  // For 2D transpose
  if (shape.size() == 2 && dim0 == 0 && dim1 == 1) {
    for (int i = 0; i < shape[0]; ++i) {
      for (int j = 0; j < shape[1]; ++j) {
        output->data[j * shape[0] + i] = data[i * shape[1] + j];
      }
    }
  }

  return output;
}

// Reduction operations
TensorPtr Tensor::sum(int dim, bool keepdim) {
  if (dim == -1) {
    // Sum all elements
    TensorPtr output;
#ifdef USE_CUDA
    if (is_cuda) {
        output = Tensor::zeros({1}, requires_grad, true);
        cuda::sum_cuda_device(data_ptr(), output->data_ptr(), numel());
    } else
#endif
    {
        sync_to_cpu(); // Ensure CPU data valid for accumulation
        float total = std::accumulate(data.begin(), data.end(), 0.0f);
        output = Tensor::from_data({total}, {1}, requires_grad, is_cuda);
    }
    
    if (requires_grad) {
        auto grad_fn = std::make_shared<SumBackward>(shape);
        grad_fn->inputs = {shared_from_this()};
        output->grad_fn = grad_fn;
    }
    return output;
  }

// Dimension-specific sum (simplified)
  sync_to_cpu(); // Fallback to CPU for dim-sum for now
  std::vector<int> new_shape;
  int reduced_size = shape[dim];
  int outer = 1, inner = 1;
  for (int i = 0; i < dim; ++i) {
    outer *= shape[i];
    new_shape.push_back(shape[i]);
  }
  if (keepdim) new_shape.push_back(1);
  for (size_t i = dim + 1; i < shape.size(); ++i) {
    inner *= shape[i];
    new_shape.push_back(shape[i]);
  }

  auto output = Tensor::zeros(new_shape, requires_grad, is_cuda);
  for (int o = 0; o < outer; ++o) {
    for (int n = 0; n < inner; ++n) {
      float sum_val = 0.0f;
      for (int r = 0; r < reduced_size; ++r) {
        sum_val += data[(o * reduced_size + r) * inner + n];
      }
      output->data[o * inner + n] = sum_val;
    }
  }
  return output;
}

TensorPtr Tensor::mean(int dim, bool keepdim) {
  // Use direct mean implementation for global case to support autograd properly
  if (dim == -1) {
     auto sum_tensor = sum(dim, keepdim); // This sets SumBackward
     int n = numel();
     // If we use sum()->mul_scalar(), mul_scalar doesn't support autograd yet?
     // So we better implement Mean directly or rely on SumBackward + Div?
     // Or just manually set MeanBackward.
     
     // Let's use direct MeanBackward for efficiency and clarity.
     auto output = sum_tensor->mul_scalar(1.0f / n);
     
     if (requires_grad) {
         // Overwrite grad_fn? Or chain it?
         // If we use SumBackward, then mul_scalar needs to track grad?
         // mul_scalar currently does NOT.
         // So we should manually set MeanBackward on the output and inputs.
         auto grad_fn = std::make_shared<MeanBackward>(shape, n);
         grad_fn->inputs = {shared_from_this()};
         output->grad_fn = grad_fn;
         // Note: sum_tensor intermediate is bypassed for grad purposes if we do this.
     }
     return output;
  }

  auto sum_tensor = sum(dim, keepdim);
  int reduced_size = (dim == -1) ? numel() : shape[dim];
  return sum_tensor->mul_scalar(1.0f / reduced_size);
}

// Activations
TensorPtr Tensor::relu() {
  auto output = Tensor::zeros(shape, requires_grad, is_cuda);
#ifdef USE_CUDA
  if (is_cuda) {
    int size = numel();
    cuda::relu_cuda_device(data_ptr(), output->data_ptr(), size);
    cudaDeviceSynchronize();
  } else
#endif
  {
    sync_to_cpu(); // Ensure CPU data is available
    #pragma omp parallel for
    for (int i = 0; i < (int)data.size(); ++i) {
      output->data[i] = std::max(0.0f, data[i]);
    }
  }

  if (requires_grad) {
    auto grad_fn = std::make_shared<ReLUBackward>();
    grad_fn->inputs = {shared_from_this()};
    output->grad_fn = grad_fn;
  }

  return output;
}

TensorPtr Tensor::leaky_relu(float negative_slope) {
  auto output = Tensor::zeros(shape, requires_grad, is_cuda);
  #pragma omp parallel for
  for (int i = 0; i < (int)data.size(); ++i) {
    output->data[i] = data[i] > 0 ? data[i] : negative_slope * data[i];
  }
  return output;
}

TensorPtr Tensor::tanh_() {
  auto output = Tensor::zeros(shape, requires_grad, is_cuda);
#ifdef USE_CUDA
  if (is_cuda) {
    int size = numel();
    cuda::tanh_cuda_device(data_ptr(), output->data_ptr(), size);
    cudaDeviceSynchronize();
  } else
#endif
  {
    sync_to_cpu();
    #pragma omp parallel for
    for (int i = 0; i < (int)data.size(); ++i) {
      output->data[i] = std::tanh(data[i]);
    }
  }
  if (requires_grad) {
    auto grad_fn = std::make_shared<TanhBackward>();
    grad_fn->inputs = {shared_from_this()};
    grad_fn->output_cache = output;
    output->grad_fn = grad_fn;
  }
  return output;
}

TensorPtr Tensor::sigmoid() {
  auto output = Tensor::zeros(shape, requires_grad, is_cuda);
#ifdef USE_CUDA
  if (is_cuda) {
    int size = numel();
    cuda::sigmoid_cuda_device(data_ptr(), output->data_ptr(), size);
    cudaDeviceSynchronize();
  } else
#endif
  {
    sync_to_cpu();
    #pragma omp parallel for
    for (int i = 0; i < (int)data.size(); ++i) {
      output->data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }
  }

  if (requires_grad) {
    auto grad_fn = std::make_shared<SigmoidBackward>();
    grad_fn->inputs = {shared_from_this()};
    grad_fn->output_cache = output;
    output->grad_fn = grad_fn;
  }
  return output;
}

// Math operations
TensorPtr Tensor::exp() {
  auto output = Tensor::zeros(shape, requires_grad, is_cuda);
#ifdef USE_CUDA
  if (is_cuda) {
    cuda::exp_cuda_device(data_ptr(), output->data_ptr(), numel());
  } else
#endif
  {
    #pragma omp parallel for
    for (int i = 0; i < (int)data.size(); ++i) {
      output->data[i] = std::exp(data[i]);
    }
  }
  return output;
}

TensorPtr Tensor::log() {
  auto output = Tensor::zeros(shape, requires_grad, is_cuda);
#ifdef USE_CUDA
  if (is_cuda) {
    cuda::log_cuda_device(data_ptr(), output->data_ptr(), numel());
  } else
#endif
  {
    #pragma omp parallel for
    for (int i = 0; i < (int)data.size(); ++i) {
      output->data[i] = std::log(data[i] + 1e-8f);
    }
  }
  return output;
}

TensorPtr Tensor::pow(float exponent) {
  auto output = Tensor::zeros(shape, requires_grad, is_cuda);
#ifdef USE_CUDA
  if (is_cuda) {
    cuda::pow_cuda_device(data_ptr(), exponent, output->data_ptr(), numel());
  } else
#endif
  {
    #pragma omp parallel for
    for (int i = 0; i < (int)data.size(); ++i) {
      output->data[i] = std::pow(data[i], exponent);
    }
  }
  
  if (requires_grad) {
      auto grad_fn = std::make_shared<PowBackward>(exponent);
      grad_fn->inputs = {shared_from_this()};
      output->grad_fn = grad_fn;
  }
  
  return output;
}

TensorPtr Tensor::sqrt() {
  auto output = Tensor::zeros(shape, requires_grad, is_cuda);
#ifdef USE_CUDA
  if (is_cuda) {
    cuda::sqrt_cuda_device(data_ptr(), output->data_ptr(), numel());
  } else
#endif
  {
    #pragma omp parallel for
    for (int i = 0; i < (int)data.size(); ++i) {
      output->data[i] = std::sqrt(data[i]);
    }
  }
  return output;

}

TensorPtr Tensor::max(int dim, bool keepdim) {
  if (dim == -1) {
#ifdef USE_CUDA
    if (is_cuda) {
        auto output = Tensor::zeros({1}, requires_grad, true);
        cuda::max_cuda_device(data_ptr(), output->data_ptr(), numel());
        return output;
    }
#endif
    sync_to_cpu();
    if (data.empty()) return Tensor::from_data({0.0f}, {1}, false, false);
    float max_val = *std::max_element(data.begin(), data.end());
    return Tensor::from_data({max_val}, {1}, false, is_cuda);
  }
  throw std::runtime_error("Dimension-specific max not implemented yet");
}

TensorPtr Tensor::min(int dim, bool keepdim) {
  if (dim == -1) {
    sync_to_cpu();
    if (data.empty()) return Tensor::from_data({0.0f}, {1}, false, false);
    float min_val = *std::min_element(data.begin(), data.end());
    return Tensor::from_data({min_val}, {1}, false, is_cuda);
  }
  throw std::runtime_error("Dimension-specific min not implemented yet");
}

TensorPtr Tensor::permute(const std::vector<int> &dims) {
  // Only support 4D permute for now (most common: NCHW <-> NHWC)
  if (dims.size() != shape.size()) {
    throw std::runtime_error("permute: dims size must match tensor dimensions");
  }

  std::vector<int> new_shape(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    new_shape[i] = shape[dims[i]];
  }

  auto output = Tensor::zeros(new_shape, requires_grad, is_cuda);

#ifdef USE_CUDA
  if (is_cuda && shape.size() == 4) {
    cuda::permute4d_cuda_device(data_ptr(), output->data_ptr(), 
                                shape[0], shape[1], shape[2], shape[3],
                                dims[0], dims[1], dims[2], dims[3]);
    return output;
  }
#endif

  // Fallback to CPU
  if (is_cuda) sync_to_cpu();

  // Generic permute using stride computation
  std::vector<int> new_strides(dims.size());
  int stride = 1;
  for (int i = (int)dims.size() - 1; i >= 0; --i) {
    new_strides[i] = stride;
    stride *= new_shape[i];
  }

  int total = (int)data.size();
  for (int flat = 0; flat < total; ++flat) {
    // Convert flat index to multi-dimensional indices
    int remaining = flat;
    std::vector<int> old_idx(shape.size());
    for (int d = 0; d < (int)shape.size(); ++d) {
      old_idx[d] = remaining / strides[d];
      remaining %= strides[d];
    }

    // Compute new flat index
    int new_flat = 0;
    for (size_t d = 0; d < dims.size(); ++d) {
      new_flat += old_idx[dims[d]] * new_strides[d];
    }

    output->data[new_flat] = data[flat];
  }

  return output;
}

// Autograd
void Tensor::backward(const TensorPtr &gradient) {
  if (!requires_grad) return;

  // Initialize gradient
  if (!grad_fn) {
    // If we call backward on a leaf or the result of an op, 
    // it might not have grad_fn if it's the loss.
    // If no gradient provided, use 1.0 (mean to loss)
    int total_size = numel();
    if (gradient == nullptr) {
        if (is_cuda) {
#ifdef USE_CUDA
            if (!d_grad) {
                size_t bytes = total_size * sizeof(float);
                d_grad = (float *)cuda::cuda_malloc(bytes);
            }
            std::vector<float> ones_data(total_size, 1.0f);
            cuda::cuda_memcpy_host_to_device(d_grad, ones_data.data(), total_size * sizeof(float));
            cuda_dirty = true; // GPU has grad, CPU doesn't
#endif
        } else {
            grad.assign(total_size, 1.0f);
#ifdef USE_CUDA
            cpu_dirty = true;
#endif
        }
    } else {
        // Use provided gradient
        if (is_cuda) {
#ifdef USE_CUDA
            if (!d_grad) {
                size_t bytes = total_size * sizeof(float);
                d_grad = (float *)cuda::cuda_malloc(bytes);
            }
            cuda::cuda_memcpy_device_to_device(d_grad, gradient->data_ptr(), total_size * sizeof(float));
            cuda_dirty = true;
#endif
        } else {
            grad = gradient->data;
#ifdef USE_CUDA
            cpu_dirty = true;
#endif
        }
    }
  }

  // Accumulate gradient
  if (is_cuda) {
#ifdef USE_CUDA
    if (!d_grad) { // Should have been allocated by now if requires_grad
      size_t bytes = numel() * sizeof(float);
      d_grad = (float *)cuda::cuda_malloc(bytes);
      cuda::cuda_memset(d_grad, 0, bytes); // Initialize to zero
    }
    if (gradient) {
      cuda::add_cuda_device(d_grad, gradient->data_ptr(), d_grad, numel());
    } else {
      cuda::fill_cuda_device(d_grad, 1.0f, numel());
    }
    cuda_dirty = true;
#endif
  } else {
    if (grad.size() != data.size()) {
      grad.resize(data.size(), 0.0f);
    }
    if (gradient) {
      for (size_t i = 0; i < grad.size(); ++i) {
        grad[i] += gradient->data[i];
      }
    } else {
      std::fill(grad.begin(), grad.end(), 1.0f);
    }
#ifdef USE_CUDA
    cpu_dirty = true;
#endif
  }

  if (grad_fn) {
    // Create a gradient tensor from this tensor's accumulated grad
    auto grad_tensor = Tensor::from_data(grad, shape, false, is_cuda); // This will sync if needed
    auto input_grads = grad_fn->backward(grad_tensor);
    for (size_t i = 0; i < input_grads.size(); ++i) {
      if (i < grad_fn->inputs.size() && grad_fn->inputs[i]->requires_grad) {
        grad_fn->inputs[i]->backward(input_grads[i]);
      }
    }
  }
}

void Tensor::zero_grad() {
  if (!requires_grad) return;
  if (is_cuda) {
#ifdef USE_CUDA
    if (d_grad) {
      cuda::cuda_memset(d_grad, 0, numel() * sizeof(float));
      cuda_dirty = true; // GPU grad is now zero, CPU is stale
    }
#endif
  } else {
    std::fill(grad.begin(), grad.end(), 0.0f);
#ifdef USE_CUDA
    cpu_dirty = true; // CPU grad is now zero, GPU is stale
#endif
  }
}


// CUDA operations
void Tensor::cuda() {
#ifdef USE_CUDA
  if (!is_cuda) {
    is_cuda = true;
    allocate_device_memory();
    // If CPU data is present and not dirty, sync it to CUDA
    if (!data.empty() && cpu_dirty) {
      sync_to_cuda();
    } else {
      // If CPU data is stale or empty, CUDA memory is the source of truth
      cuda_dirty = true;
      cpu_dirty = false;
    }
  }
#else
  throw std::runtime_error("DeepNet compiled without CUDA support");
#endif
}

void Tensor::cpu() {
#ifdef USE_CUDA
  if (is_cuda) {
    sync_to_cpu(); // Copy device data to CPU if GPU is ahead
    free_device_memory();
  }
#endif
  is_cuda = false;
}

TensorPtr Tensor::to(bool cuda) {
  auto output = clone();
  if (output->is_cuda != cuda) {
    if (cuda) {
      output->cuda();
    } else {
      output->cpu();
    }
  }
  return output;
}

// Utility
void Tensor::copy_(const TensorPtr &other) {
  if (shape != other->shape) {
    throw std::runtime_error("copy_ expects same shape");
  }
  
#ifdef USE_CUDA
  if (is_cuda) {
      if (!d_data) allocate_device_memory();
      
      if (other->is_cuda) {
          cuda::cuda_memcpy_device_to_device(d_data, other->d_data, numel() * sizeof(float));
      } else {
          cuda::cuda_memcpy_host_to_device(d_data, other->data.data(), numel() * sizeof(float));
      }
      cuda_dirty = true;
      cpu_dirty = false;
  } else {
      if (other->is_cuda) {
          if (data.empty()) data.resize(numel());
          cuda::cuda_memcpy_device_to_host(data.data(), other->d_data, numel() * sizeof(float));
      } else {
          data = other->data;
      }
      cpu_dirty = true;
      cuda_dirty = false;
  }
#else
  if (other->is_cuda) {
     throw std::runtime_error("Compile without CUDA but source is CUDA?");
  }
  data = other->data;
#endif
}

void Tensor::fill_(float value) {
  if (is_cuda) {
#ifdef USE_CUDA
    if (!d_data) allocate_device_memory();
    cuda::fill_cuda_device(d_data, value, numel());
    if (requires_grad && d_grad) {
      cuda::fill_cuda_device(d_grad, 0.0f, numel());
    }
    cuda_dirty = true;
#endif
  } else {
    std::fill(data.begin(), data.end(), value);
    if (requires_grad) {
      std::fill(grad.begin(), grad.end(), 0.0f);
    }
#ifdef USE_CUDA
    cpu_dirty = true;
#endif
  }
}

void Tensor::uniform_(float min, float max) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(min, max);
  if (is_cuda) {
#ifdef USE_CUDA
    if (!d_data) allocate_device_memory();
    // For now, generate on CPU and sync
    data.resize(numel());
    for (auto &val : data) {
      val = dist(gen);
    }
    cuda::cuda_memcpy_host_to_device(d_data, data.data(), numel() * sizeof(float));
    cuda_dirty = true;
    cpu_dirty = true; // CPU data was just generated
#endif
  } else {
    for (auto &val : data) {
      val = dist(gen);
    }
#ifdef USE_CUDA
    cpu_dirty = true;
#endif
  }
}

void Tensor::normal_(float mean, float std) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::normal_distribution<float> dist(mean, std);
  if (is_cuda) {
#ifdef USE_CUDA
    if (!d_data) allocate_device_memory();
    // For now, generate on CPU and sync
    data.resize(numel());
    for (auto &val : data) {
      val = dist(gen);
    }
    cuda::cuda_memcpy_host_to_device(d_data, data.data(), numel() * sizeof(float));
    cuda_dirty = true;
    cpu_dirty = true; // CPU data was just generated
#endif
  } else {
    for (auto &val : data) {
      val = dist(gen);
    }
#ifdef USE_CUDA
    cpu_dirty = true;
#endif
  }
}

TensorPtr Tensor::clone() {
#ifdef USE_CUDA
  if (is_cuda) {
    // For CUDA tensors, create new tensor and copy device data
    auto output = Tensor::zeros(shape, requires_grad, true);
    if (d_data && output->d_data) {
      int total_size = numel();
      cuda::cuda_memcpy_device_to_device(output->d_data, d_data, total_size * sizeof(float));
      output->cuda_dirty = true; // Output's GPU data is fresh
    }
    // Do not copy gradients to match CPU behavior (fresh tensor gradient)
    return output;
  }
#endif
  // For CPU tensors, or if USE_CUDA is not defined
  auto output = Tensor::from_data(data, shape, requires_grad, is_cuda);
#ifdef USE_CUDA
  output->cpu_dirty = true; // Output's CPU data is fresh
#endif
  return output;
}

std::string Tensor::shape_str() const {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    ss << shape[i];
    if (i < shape.size() - 1)
      ss << ", ";
  }
  ss << "]";
  return ss.str();
}

void Tensor::print(const std::string &name) const {
  std::cout << name << " shape: " << shape_str() << std::endl;
  std::cout << "Data (first 10): ";
  for (size_t i = 0; i < std::min(size_t(10), data.size()); ++i) {
    std::cout << data[i] << " ";
  }
  std::cout << std::endl;
}

void Tensor::check_shape_compatible(const TensorPtr &other) const {
  if (shape != other->shape) {
    throw std::runtime_error("Shape mismatch: " + shape_str() + " vs " +
                             other->shape_str());
  }
}

float &Tensor::at(const std::vector<int> &indices) {
  return data[compute_offset(indices)];
}

const float &Tensor::at(const std::vector<int> &indices) const {
  return data[compute_offset(indices)];
}

int Tensor::compute_offset(const std::vector<int> &indices) const {
  int offset = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    offset += indices[i] * strides[i];
  }
  return offset;
}

#ifdef USE_CUDA
void Tensor::allocate_device_memory() {
  int total_size = numel();
  if (total_size == 0) return;
  
  size_t bytes = total_size * sizeof(float);
  if (!d_data) {
    d_data = (float *)cuda::cuda_malloc(bytes);
  }
  if (requires_grad && !d_grad) {
    d_grad = (float *)cuda::cuda_malloc(bytes);
  }
}

void Tensor::free_device_memory() {
  if (d_data) {
    cuda::cuda_free(d_data);
    d_data = nullptr;
  }
  if (d_grad) {
    cuda::cuda_free(d_grad);
    d_grad = nullptr;
  }
}

void Tensor::sync_to_cpu() {
  if (!is_cuda || !d_data) {
     if (data.empty() && numel() > 0) data.resize(numel(), 0.0f);
     return;
  }
  
  // Sync if GPU data is dirty (modified)
  if (cuda_dirty || data.empty()) {
    int total_size = numel();
    if (total_size == 0) return;
    
    if (data.size() != (size_t)total_size) {
      data.resize(total_size);
    }
    
    cuda::cuda_memcpy_device_to_host(data.data(), d_data, total_size * sizeof(float));
    
    if (requires_grad && d_grad && (cuda_dirty || grad.empty())) {
      if (grad.size() != (size_t)total_size) {
        grad.resize(total_size);
      }
      cuda::cuda_memcpy_device_to_host(grad.data(), d_grad, total_size * sizeof(float));
    }
    
    cuda_dirty = false;
  }
}

void Tensor::sync_to_cuda() {
  if (!is_cuda) return;
  if (!d_data) allocate_device_memory();
  
  // Sync if CPU data is dirty (modified)
  if (cpu_dirty) {
    int total_size = numel();
    if (total_size == 0) return;
    
    if (data.size() != (size_t)total_size) {
      data.resize(total_size, 0.0f);
    }
    
    cuda::cuda_memcpy_host_to_device(d_data, data.data(), total_size * sizeof(float));
    
    if (requires_grad && d_grad) {
      if (grad.size() != (size_t)total_size) {
        grad.resize(total_size, 0.0f);
      }
      cuda::cuda_memcpy_host_to_device(d_grad, grad.data(), total_size * sizeof(float));
    }
    
    cpu_dirty = false;
  }
}

float *Tensor::data_ptr() {
  if (is_cuda) {
    if (!d_data) allocate_device_memory(); // Ensure device memory is allocated
    if (cpu_dirty) sync_to_cuda(); // If CPU is ahead, sync to CUDA
    cuda_dirty = true; // Mark that GPU data could be modified
    cpu_dirty = false; // CPU is now definitely stale if GPU is modified
    return d_data;
  }
  if (cuda_dirty) sync_to_cpu(); // If GPU is ahead, sync to CPU
  cpu_dirty = true;  // Mark that CPU data could be modified
  cuda_dirty = false; // GPU is now definitely stale
  return data.data();
}

const float *Tensor::data_ptr() const {
  if (is_cuda) {
    if (cpu_dirty) const_cast<Tensor*>(this)->sync_to_cuda(); // If CPU is ahead, sync to CUDA
    return d_data;
  }
  if (cuda_dirty) const_cast<Tensor*>(this)->sync_to_cpu(); // If GPU is ahead, sync to CPU
  return data.data();
}

float *Tensor::grad_ptr() {
  if (!requires_grad) return nullptr; // No grad for non-requiring tensors

  if (is_cuda) {
    if (!d_grad) { // Ensure device grad memory is allocated
      allocate_device_memory();
      // If newly allocated, it's uninitialized, so mark as dirty
      cuda_dirty = true;
      cpu_dirty = false;
    }
    if (cpu_dirty) sync_to_cuda(); // If CPU is ahead, sync to CUDA
    cuda_dirty = true; // Mark that GPU grad could be modified
    cpu_dirty = false;
    return d_grad;
  }
  if (cuda_dirty) sync_to_cpu(); // If GPU is ahead, sync to CPU
  cpu_dirty = true; // Mark that CPU grad could be modified
  cuda_dirty = false;
  if (grad.size() != data.size()) { // Ensure CPU grad buffer is sized
    grad.resize(data.size(), 0.0f);
  }
  return grad.data();
}
#else
void Tensor::sync_to_cpu() {}
void Tensor::sync_to_cuda() {}
float *Tensor::data_ptr() { return data.data(); }
const float *Tensor::data_ptr() const { return data.data(); }
float *Tensor::grad_ptr() {
  if (requires_grad && grad.size() != data.size()) {
    grad.resize(data.size(), 0.0f);
  }
  return grad.data();
}
#endif

} // namespace deepnet
