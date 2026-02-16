#ifdef USE_CUDA

#include "cuda/cuda_ops.hpp"
#include "cuda/cuda_utils.hpp"
#include <cuda_runtime.h>
#include <vector>


namespace deepnet {
namespace cuda {

// Kernel implementations
constexpr int BLOCK_SIZE = 256;
constexpr int TILE_SIZE = 16;

// ============================================================
// CUDA Kernels
// ============================================================

// Conv2D Utils matching DeepNet layout [N, OH, OW, C*KH*KW]

__global__ void im2col_kernel(const int n, const float* data_im, 
                              const int N, const int C, const int H, const int W,
                              const int kernel_h, const int kernel_w,
                              const int pad_h, const int pad_w,
                              const int stride_h, const int stride_w,
                              const int out_h, const int out_w,
                              float* data_col) {
  // Grid stride loop or simple thread mapping
  // We parallelize over the output elements (data_col size)
  // data_col size = N * out_h * out_w * (C * kernel_h * kernel_w)
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int K = C * kernel_h * kernel_w;
  int num_elements = n; 

  if (index < num_elements) {
    // index = ((n * out_h + oh) * out_w + ow) * K + k_idx
    int k_idx = index % K;
    int temp = index / K;
    int ow = temp % out_w;
    int temp2 = temp / out_w;
    int oh = temp2 % out_h;
    int n_idx = temp2 / out_h;

    // Decode k_idx into c, kh, kw
    // k_idx = (c * kernel_h + kh) * kernel_w + kw
    int kw = k_idx % kernel_w;
    int temp3 = k_idx / kernel_w;
    int kh = temp3 % kernel_h;
    int c = temp3 / kernel_h;

    int ih = oh * stride_h - pad_h + kh;
    int iw = ow * stride_w - pad_w + kw;

    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
      int in_idx = ((n_idx * C + c) * H + ih) * W + iw;
      data_col[index] = data_im[in_idx];
    } else {
      data_col[index] = 0.0f;
    }
  }
}

__global__ void col2im_kernel(const int n, const float* data_col,
                              const int N, const int C, const int H, const int W,
                              const int kernel_h, const int kernel_w,
                              const int pad_h, const int pad_w,
                              const int stride_h, const int stride_w,
                              const int out_h, const int out_w,
                              float* data_im) {
  // Parallelize over data_col (gradient) elements and atomicAdd to data_im
  // data_col size = N * out_h * out_w * (C * kernel_h * kernel_w)
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int K = C * kernel_h * kernel_w;
  
  if (index < n) {
      // Decode index (same as im2col)
      int k_idx = index % K;
      int temp = index / K;
      int ow = temp % out_w;
      int temp2 = temp / out_w;
      int oh = temp2 % out_h;
      int n_idx = temp2 / out_h;

      int kw = k_idx % kernel_w;
      int temp3 = k_idx / kernel_w;
      int kh = temp3 % kernel_h;
      int c = temp3 / kernel_h;

      int ih = oh * stride_h - pad_h + kh;
      int iw = ow * stride_w - pad_w + kw;

      if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
        int in_idx = ((n_idx * C + c) * H + ih) * W + iw;
        atomicAdd(&data_im[in_idx], data_col[index]);
      }
  }
}

// Bias Addition: Add bias[c] to input[n, c, h, w]
__global__ void add_bias_shared_kernel(const float* bias, float* output,
                                      int N, int C, int H, int W) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_elements = N * C * H * W;
  
  if (index < num_elements) {
    // index = ((n * C + c) * H + h) * W + w
    int c = (index / (H * W)) % C;
    output[index] += bias[c];
  }
}

// Bias Gradient: Sum output[n, c, h, w] over n, h, w into grad_bias[c]
__global__ void bias_backward_kernel(const float* grad_output, float* grad_bias,
                                    int N, int C, int H, int W) {
  int c = blockIdx.x; // Block per channel
  if (c >= C) return;
  
  int tid = threadIdx.x;
  float sum = 0.0f;
  int spatial_size = H * W;
  int num_elements = N * spatial_size; // loop stride
  
  // Grid-stride loop within channel
  for (int i = tid; i < num_elements; i += blockDim.x) {
    int n = i / spatial_size;
    int hw = i % spatial_size;
    // int h = hw / W; 
    // int w = hw % W; 
    
    // idx = ((n * C + c) * H + h) * W + w
    //     = n * C * spatial + c * spatial + hw
    int idx = n * (C * spatial_size) + c * spatial_size + hw;
    sum += grad_output[idx];
  }
  
  // Block reduction
  __shared__ float s_sum[256];
  s_sum[tid] = sum;
  __syncthreads();
  
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
    }
    __syncthreads();
  }
  
  if (tid == 0) {
    atomicAdd(&grad_bias[c], s_sum[0]);
  }
}

// Optimizer Kernels
__global__ void sgd_update_kernel(float* params, float* velocity, const float* grads,
                                  float lr, float momentum, float weight_decay,
                                  bool nesterov, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float p = params[idx];
    float g = grads[idx];
    
    if (weight_decay > 0.0f) {
      g += weight_decay * p;
    }
    
    if (momentum > 0.0f) {
       float v = velocity[idx];
       v = momentum * v + g;
       velocity[idx] = v;
       
       if (nesterov) {
         p -= lr * (momentum * v + g);
       } else {
         p -= lr * v;
       }
    } else {
       p -= lr * g;
    }
    
    params[idx] = p;
  }
}

__global__ void adam_update_kernel(float* params, float* m, float* v, const float* grads,
                                   float lr, float beta1, float beta2, float eps,
                                   float weight_decay, int t, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float p = params[idx];
    float g = grads[idx];
    
    if (weight_decay > 0.0f) {
      g += weight_decay * p;
    }
    
    float m_val = m[idx];
    float v_val = v[idx];
    
    m_val = beta1 * m_val + (1.0f - beta1) * g;
    v_val = beta2 * v_val + (1.0f - beta2) * g * g;
    
    m[idx] = m_val;
    v[idx] = v_val;
    
    float m_hat = m_val / (1.0f - powf(beta1, t));
    float v_hat = v_val / (1.0f - powf(beta2, t));
    
    p -= lr * m_hat / (sqrtf(v_hat) + eps);
    
    params[idx] = p;
  }
}


__global__ void add_kernel(const float *a, const float *b, float *out,
                           int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = a[idx] + b[idx];
  }
}

__global__ void mul_kernel(const float *a, const float *b, float *out,
                           int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = a[idx] * b[idx];
  }
}

__global__ void add_inplace_kernel(float *a, const float *b, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    a[idx] += b[idx];
  }
}

__global__ void matmul_kernel(const float *A, const float *B, float *C, int M,
                              int N, int K) {
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  float sum = 0.0f;

  for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
    if (row < M && t * TILE_SIZE + threadIdx.x < K)
      As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
    else
      As[threadIdx.y][threadIdx.x] = 0.0f;

    if (t * TILE_SIZE + threadIdx.y < K && col < N)
      Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
    else
      Bs[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    for (int k = 0; k < TILE_SIZE; ++k) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

__global__ void relu_kernel(const float *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = fmaxf(0.0f, input[idx]);
  }
}

__global__ void sigmoid_kernel(const float *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = 1.0f / (1.0f + expf(-input[idx]));
  }
}

__global__ void tanh_kernel(const float *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = tanhf(input[idx]);
  }
}

__global__ void sub_kernel(const float *a, const float *b, float *out, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = a[idx] - b[idx];
  }
}

__global__ void div_kernel(const float *a, const float *b, float *out, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = a[idx] / (b[idx] + 1e-8f);
  }
}

__global__ void pow_kernel(const float *input, float exponent, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = powf(input[idx], exponent);
  }
}

__global__ void exp_kernel(const float *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = expf(input[idx]);
  }
}

__global__ void log_kernel(const float *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = logf(input[idx] + 1e-8f);
  }
}

__global__ void sqrt_kernel(const float *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = sqrtf(input[idx]);
  }
}

// Simple atomic reduction for global sum/max
__global__ void sum_kernel(const float *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  float local_sum = 0.0f;
  for (int i = idx; i < size; i += stride) {
    local_sum += input[i];
  }
  atomicAdd(output, local_sum);
}

__global__ void max_kernel(const float *input, float *output, int size) {
    // Note: this is a simplified max using atomicMax which only works for int/uint smoothly. 
    // For float, we need a CAS loop or a specific trick. 
    // Since standard atomicMax for float isn't available in all archs, 
    // we will implement a basic version or assume atomicMax supported or use a lock.
    // For simplicity/compatibility, let's use a critical section approach or multi-pass.
    // Actually, let's use a CAS loop for float atomicMax.
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float local_max = -1e37f; // -infinity approximation
    bool has_data = false;

    for (int i = idx; i < size; i += stride) {
        local_max = fmaxf(local_max, input[i]);
        has_data = true;
    }

    if (has_data) {
        // Atomic max for float using CAS
        int *address_as_int = (int *)output;
        int old = *address_as_int, assumed;
        do {
            assumed = old;
            float assumed_float = __int_as_float(assumed);
            float new_val = fmaxf(assumed_float, local_max);
            old = atomicCAS(address_as_int, assumed, __float_as_int(new_val));
        } while (assumed != old);
    }
}


__global__ void add_scalar_kernel(const float *input, float scalar, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input[idx] + scalar;
  }
}

__global__ void mul_scalar_kernel(const float *input, float scalar, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input[idx] * scalar;
  }
}

__global__ void fill_kernel(float *data, float value, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] = value;
  }
}

// ============================================================
// Low-level kernel launchers (device pointers)
// ============================================================

void add_cuda(const float *a, const float *b, float *out, int size) {
  if (size <= 0) return;
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  add_kernel<<<blocks, BLOCK_SIZE>>>(a, b, out, size);
  CUDA_CHECK(cudaGetLastError());
}

void mul_cuda(const float *a, const float *b, float *out, int size) {
  if (size <= 0) return;
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  mul_kernel<<<blocks, BLOCK_SIZE>>>(a, b, out, size);
  CUDA_CHECK(cudaGetLastError());
}

void add_inplace_cuda_device(float *a, const float *b, int size) {
  if (size <= 0) return;
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  add_inplace_kernel<<<blocks, BLOCK_SIZE>>>(a, b, size);
  CUDA_CHECK(cudaGetLastError());
}

void matmul_cuda(const float *a, const float *b, float *out, int M, int N,
                 int K) {
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
  matmul_kernel<<<grid, block>>>(a, b, out, M, N, K);
  CUDA_CHECK(cudaGetLastError());
}

void relu_cuda(const float *input, float *output, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  relu_kernel<<<blocks, BLOCK_SIZE>>>(input, output, size);
  CUDA_CHECK(cudaGetLastError());
}

void sigmoid_cuda(const float *input, float *output, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  sigmoid_kernel<<<blocks, BLOCK_SIZE>>>(input, output, size);
  CUDA_CHECK(cudaGetLastError());
}

void tanh_cuda(const float *input, float *output, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tanh_kernel<<<blocks, BLOCK_SIZE>>>(input, output, size);
  CUDA_CHECK(cudaGetLastError());
}

// Low-level wrappers implementing the new kernels
void sub_cuda(const float *a, const float *b, float *out, int size) {
  if (size <= 0) return;
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  sub_kernel<<<blocks, BLOCK_SIZE>>>(a, b, out, size);
  CUDA_CHECK(cudaGetLastError());
}

void div_cuda(const float *a, const float *b, float *out, int size) {
  if (size <= 0) return;
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  div_kernel<<<blocks, BLOCK_SIZE>>>(a, b, out, size);
  CUDA_CHECK(cudaGetLastError());
}

void pow_cuda(const float *input, float exponent, float *output, int size) {
  if (size <= 0) return;
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  pow_kernel<<<blocks, BLOCK_SIZE>>>(input, exponent, output, size);
  CUDA_CHECK(cudaGetLastError());
}

void exp_cuda(const float *input, float *output, int size) {
  if (size <= 0) return;
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  exp_kernel<<<blocks, BLOCK_SIZE>>>(input, output, size);
  CUDA_CHECK(cudaGetLastError());
}

void log_cuda(const float *input, float *output, int size) {
  if (size <= 0) return;
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  log_kernel<<<blocks, BLOCK_SIZE>>>(input, output, size);
  CUDA_CHECK(cudaGetLastError());
}

void sqrt_cuda(const float *input, float *output, int size) {
  if (size <= 0) return;
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  sqrt_kernel<<<blocks, BLOCK_SIZE>>>(input, output, size);
  CUDA_CHECK(cudaGetLastError());
}

void sum_cuda(const float *input, float *output, int size) {
  // Initialize output to 0
  cuda_memset(output, 0, sizeof(float));
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  // Use a fixed number of blocks for reduction to avoid overhead
  blocks = std::min(blocks, 256); 
  sum_kernel<<<blocks, BLOCK_SIZE>>>(input, output, size);
  CUDA_CHECK(cudaGetLastError());
}

void max_cuda(const float *input, float *output, int size) {
  // Initialize to very small number
  float min_val = -1e37f;
  cuda_memcpy_host_to_device(output, &min_val, sizeof(float));
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  blocks = std::min(blocks, 256);
  max_kernel<<<blocks, BLOCK_SIZE>>>(input, output, size);
  CUDA_CHECK(cudaGetLastError());
}


void add_scalar_cuda(const float *input, float scalar, float *output, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  add_scalar_kernel<<<blocks, BLOCK_SIZE>>>(input, scalar, output, size);
  CUDA_CHECK(cudaGetLastError());
}
void mul_scalar_cuda(const float *input, float scalar, float *output, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  mul_scalar_kernel<<<blocks, BLOCK_SIZE>>>(input, scalar, output, size);
  CUDA_CHECK(cudaGetLastError());
}

void fill_cuda_device(float *d_data, float value, int size) {
  if (size <= 0) return;
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  fill_kernel<<<blocks, BLOCK_SIZE>>>(d_data, value, size);
  CUDA_CHECK(cudaGetLastError());
}

void add_cuda_device(const float *a, const float *b, float *out, int size) {
  add_cuda(a, b, out, size);
}

void mul_cuda_device(const float *a, const float *b, float *out, int size) {
  mul_cuda(a, b, out, size);
}

void matmul_cuda_device(const float *a, const float *b, float *out, int M, int N, int K) {
  matmul_cuda(a, b, out, M, N, K);
}

void relu_cuda_device(const float *input, float *output, int size) {
  relu_cuda(input, output, size);
}

void sigmoid_cuda_device(const float *input, float *output, int size) {
  sigmoid_cuda(input, output, size);
}

void tanh_cuda_device(const float *input, float *output, int size) {
  tanh_cuda(input, output, size);
}

void sub_cuda_device(const float *a, const float *b, float *out, int size) {
  sub_cuda(a, b, out, size);
}

void div_cuda_device(const float *a, const float *b, float *out, int size) {
  div_cuda(a, b, out, size);
}

void pow_cuda_device(const float *input, float exponent, float *output, int size) {
  pow_cuda(input, exponent, output, size);
}

void exp_cuda_device(const float *input, float *output, int size) {
  exp_cuda(input, output, size);
}

void log_cuda_device(const float *input, float *output, int size) {
  log_cuda(input, output, size);
}

void sqrt_cuda_device(const float *input, float *output, int size) {
  sqrt_cuda(input, output, size);
}

void sum_cuda_device(const float *input, float *output, int size) {
  sum_cuda(input, output, size);
}

void max_cuda_device(const float *input, float *output, int size) {
  max_cuda(input, output, size);
}

void add_scalar_cuda_device(const float *input, float scalar, float *output, int size) {
  add_scalar_cuda(input, scalar, output, size);
}

void mul_scalar_cuda_device(const float *input, float scalar, float *output, int size) {
  mul_scalar_cuda(input, scalar, output, size);
}



// ============================================================
// Memory operations
// ============================================================

void *cuda_malloc(size_t size) {
  void *ptr;
  CUDA_CHECK(cudaMalloc(&ptr, size));
  return ptr;
}

void cuda_free(void *ptr) {
  if (ptr) {
    CUDA_CHECK(cudaFree(ptr));
  }
}

void cuda_memset(void *ptr, int value, size_t size) {
  CUDA_CHECK(cudaMemset(ptr, value, size));
}

void cuda_memcpy_host_to_device(void *dst, const void *src, size_t size) {
  CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

void cuda_memcpy_device_to_host(void *dst, const void *src, size_t size) {
  CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

void cuda_memcpy_device_to_device(void *dst, const void *src, size_t size) {
  CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
}

// ============================================================
// High-level host wrappers: host vector -> GPU compute -> host vector
// Pattern: allocate device mem, copy in, run kernel, copy out, free
// ============================================================

void add_cuda_host(const std::vector<float> &a, const std::vector<float> &b,
                   std::vector<float> &out, int size) {
  size_t bytes = size * sizeof(float);
  float *d_a = (float *)cuda_malloc(bytes);
  float *d_b = (float *)cuda_malloc(bytes);
  float *d_out = (float *)cuda_malloc(bytes);

  cuda_memcpy_host_to_device(d_a, a.data(), bytes);
  cuda_memcpy_host_to_device(d_b, b.data(), bytes);

  add_cuda(d_a, d_b, d_out, size);
  cuda_memcpy_device_to_host(out.data(), d_out, bytes);

  cuda_free(d_a);
  cuda_free(d_b);
  cuda_free(d_out);
}

void mul_cuda_host(const std::vector<float> &a, const std::vector<float> &b,
                   std::vector<float> &out, int size) {
  size_t bytes = size * sizeof(float);
  float *d_a = (float *)cuda_malloc(bytes);
  float *d_b = (float *)cuda_malloc(bytes);
  float *d_out = (float *)cuda_malloc(bytes);

  cuda_memcpy_host_to_device(d_a, a.data(), bytes);
  cuda_memcpy_host_to_device(d_b, b.data(), bytes);

  mul_cuda(d_a, d_b, d_out, size);
  cuda_memcpy_device_to_host(out.data(), d_out, bytes);

  cuda_free(d_a);
  cuda_free(d_b);
  cuda_free(d_out);
}

void matmul_cuda_host(const std::vector<float> &a, const std::vector<float> &b,
                      std::vector<float> &out, int M, int N, int K) {
  size_t bytes_a = M * K * sizeof(float);
  size_t bytes_b = K * N * sizeof(float);
  size_t bytes_out = M * N * sizeof(float);

  float *d_a = (float *)cuda_malloc(bytes_a);
  float *d_b = (float *)cuda_malloc(bytes_b);
  float *d_out = (float *)cuda_malloc(bytes_out);

  cuda_memcpy_host_to_device(d_a, a.data(), bytes_a);
  cuda_memcpy_host_to_device(d_b, b.data(), bytes_b);

  matmul_cuda(d_a, d_b, d_out, M, N, K);
  cuda_memcpy_device_to_host(out.data(), d_out, bytes_out);

  cuda_free(d_a);
  cuda_free(d_b);
  cuda_free(d_out);
}

void relu_cuda_host(const std::vector<float> &input,
                    std::vector<float> &output, int size) {
  size_t bytes = size * sizeof(float);
  float *d_in = (float *)cuda_malloc(bytes);
  float *d_out = (float *)cuda_malloc(bytes);

  cuda_memcpy_host_to_device(d_in, input.data(), bytes);

  relu_cuda(d_in, d_out, size);
  cuda_memcpy_device_to_host(output.data(), d_out, bytes);

  cuda_free(d_in);
  cuda_free(d_out);
}

void sigmoid_cuda_host(const std::vector<float> &input,
                       std::vector<float> &output, int size) {
  size_t bytes = size * sizeof(float);
  float *d_in = (float *)cuda_malloc(bytes);
  float *d_out = (float *)cuda_malloc(bytes);

  cuda_memcpy_host_to_device(d_in, input.data(), bytes);

  sigmoid_cuda(d_in, d_out, size);

  cuda_memcpy_device_to_host(output.data(), d_out, bytes);

  cuda_free(d_in);
  cuda_free(d_out);
}

void tanh_cuda_host(const std::vector<float> &input,
                    std::vector<float> &output, int size) {
  size_t bytes = size * sizeof(float);
  float *d_in = (float *)cuda_malloc(bytes);
  float *d_out = (float *)cuda_malloc(bytes);

  cuda_memcpy_host_to_device(d_in, input.data(), bytes);

  tanh_cuda(d_in, d_out, size);

  cuda_memcpy_device_to_host(output.data(), d_out, bytes);

  cuda_free(d_in);
  cuda_free(d_out);
}

// ============================================================
// Activation Backward Kernels
// ============================================================

__global__ void relu_backward_kernel(const float *grad_output, const float *input,
                                     float *grad_input, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    grad_input[idx] = input[idx] > 0 ? grad_output[idx] : 0.0f;
  }
}

__global__ void sigmoid_backward_kernel(const float *grad_output,
                                        const float *output, float *grad_input,
                                        int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float s = output[idx];
    grad_input[idx] = grad_output[idx] * s * (1.0f - s);
  }
}

__global__ void tanh_backward_kernel(const float *grad_output,
                                     const float *output, float *grad_input,
                                     int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float t = output[idx];
    grad_input[idx] = grad_output[idx] * (1.0f - t * t);
  }
}

// ============================================================
// Pooling Kernels
// ============================================================

__global__ void max_pool2d_forward_kernel(const float *input, float *output,
                                          int *indices, int N, int C, int H,
                                          int W, int out_h, int out_w, int k,
                                          int stride) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_outputs = N * C * out_h * out_w;

  if (idx < num_outputs) {
    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int c = (idx / (out_w * out_h)) % C;
    int n = idx / (out_w * out_h * C);

    float max_val = -1e37f;
    int max_idx = -1;

    for (int kh = 0; kh < k; ++kh) {
      for (int kw = 0; kw < k; ++kw) {
        int ih = oh * stride + kh;
        int iw = ow * stride + kw;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
          int in_idx = ((n * C + c) * H + ih) * W + iw;
          float val = input[in_idx];
          if (val > max_val) {
            max_val = val;
            max_idx = in_idx;
          }
        }
      }
    }
    
    // Handle edge case: if no valid pixels found, use first pixel of channel
    if (max_idx == -1) {
      max_idx = ((n * C + c) * H + 0) * W + 0;
      max_val = input[max_idx];
    }
    
    output[idx] = max_val;
    indices[idx] = max_idx;
  }
}

__global__ void max_pool2d_backward_kernel(const float *grad_output,
                                           const int *indices,
                                           float *grad_input, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    int max_idx = indices[idx];
    if (max_idx >= 0) {
      atomicAdd(&grad_input[max_idx], grad_output[idx]);
    }
  }
}

__global__ void avg_pool2d_forward_kernel(const float *input, float *output,
                                          int N, int C, int H, int W, int out_h,
                                          int out_w, int k, int stride) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_outputs = N * C * out_h * out_w;
  float pool_size = (float)(k * k);

  if (idx < num_outputs) {
    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int c = (idx / (out_w * out_h)) % C;
    int n = idx / (out_w * out_h * C);

    float sum = 0.0f;
    for (int kh = 0; kh < k; ++kh) {
      for (int kw = 0; kw < k; ++kw) {
        int ih = oh * stride + kh;
        int iw = ow * stride + kw;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
          int in_idx = ((n * C + c) * H + ih) * W + iw;
          sum += input[in_idx];
        }
      }
    }
    output[idx] = sum / pool_size;
  }
}

__global__ void avg_pool2d_backward_kernel(const float *grad_output,
                                           float *grad_input, int N, int C,
                                           int H, int W, int out_h, int out_w,
                                           int k, int stride) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_outputs = N * C * out_h * out_w;
  float pool_size = (float)(k * k);

  if (idx < num_outputs) {
    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int c = (idx / (out_w * out_h)) % C;
    int n = idx / (out_w * out_h * C);

    float grad = grad_output[idx] / pool_size;

    for (int kh = 0; kh < k; ++kh) {
      for (int kw = 0; kw < k; ++kw) {
        int ih = oh * stride + kh;
        int iw = ow * stride + kw;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
          int in_idx = ((n * C + c) * H + ih) * W + iw;
          atomicAdd(&grad_input[in_idx], grad);
        }
      }
    }
  }
}

// ============================================================
// BatchNorm Kernels
// ============================================================

__global__ void batchnorm_forward_kernel(const float *input, const float *mean,
                                         const float *var, const float *gamma,
                                         const float *beta, float *output,
                                         int N, int C, int H, int W,
                                         float eps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_elements = N * C * H * W;

  if (idx < num_elements) {
    int c = (idx / (H * W)) % C;
    float m = mean[c];
    float v = var[c];
    float g = gamma[c];
    float b = beta[c];
    float x = input[idx];

    output[idx] = g * (x - m) / sqrtf(v + eps) + b;
  }
}

// Compute mean and variance per channel
__global__ void batchnorm_stats_kernel(const float *input, float *mean, float *var,
                                       int N, int C, int H, int W) {
  int c = blockIdx.x; // One block per channel
  if (c >= C) return;

  int tid = threadIdx.x;
  int num_elements = N * H * W;
  float sum = 0.0f;
  float sum_sq = 0.0f;

  for (int i = tid; i < num_elements; i += blockDim.x) {
    // Index mapping: (n, c, h, w) -> ((n * C + c) * H + h) * W + w
    // We need to iterate over n, h, w for fixed c.
    // Flat index i corresponds to (n, h, w) tuple.
    // i = n * (H * W) + h * W + w
    int n = i / (H * W);
    int hw = i % (H * W);
    int h = hw / W;
    int w = hw % W;
    
    int idx = ((n * C + c) * H + h) * W + w;
    float val = input[idx];
    sum += val;
    sum_sq += val * val;
  }

  // Block reduction
  // Using shared memory
  __shared__ float s_sum[256];
  __shared__ float s_sum_sq[256];

  s_sum[tid] = sum;
  s_sum_sq[tid] = sum_sq;
  __syncthreads();

  // Simple clean reduction
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
      s_sum_sq[tid] += s_sum_sq[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    float m = s_sum[0] / num_elements;
    float v = s_sum_sq[0] / num_elements - m * m;
    mean[c] = m;
    var[c] = v;
  }
}

__global__ void batchnorm_update_stats_kernel(float *running_mean, float *running_var,
                                              const float *mean, const float *var,
                                              float momentum, int C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < C) {
    running_mean[idx] = (1.0f - momentum) * running_mean[idx] + momentum * mean[idx];
    running_var[idx] = (1.0f - momentum) * running_var[idx] + momentum * var[idx];
  }
}

__global__ void batchnorm_backward_reduce_kernel(const float *grad_output, const float *input,
                                                 const float *mean, const float *var,
                                                 float *grad_gamma, float *grad_beta,
                                                 int N, int C, int H, int W, float eps) {
  int c = blockIdx.x;
  if (c >= C) return;

  int tid = threadIdx.x;
  int num_elements = N * H * W;
  float sum_dy = 0.0f;
  float sum_dy_xhat = 0.0f;

  float m = mean[c];
  float v = var[c];
  float std_inv = 1.0f / sqrtf(v + eps);

  for (int i = tid; i < num_elements; i += blockDim.x) {
    int n = i / (H * W);
    int hw = i % (H * W);
    int h = hw / W;
    int w = hw % W;
    
    int idx = ((n * C + c) * H + h) * W + w;
    float val = input[idx];
    float dy = grad_output[idx];
    float x_hat = (val - m) * std_inv;
    
    sum_dy += dy;
    sum_dy_xhat += dy * x_hat;
  }

  __shared__ float s_dy[256];
  __shared__ float s_dy_xhat[256];

  s_dy[tid] = sum_dy;
  s_dy_xhat[tid] = sum_dy_xhat;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_dy[tid] += s_dy[tid + stride];
      s_dy_xhat[tid] += s_dy_xhat[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(&grad_gamma[c], s_dy_xhat[0]);
    atomicAdd(&grad_beta[c], s_dy[0]);
  }
}

__global__ void batchnorm_backward_apply_kernel(const float *grad_output, const float *input,
                                                const float *mean, const float *var,
                                                const float *gamma, const float *grad_gamma,
                                                const float *grad_beta, float *grad_input,
                                                int N, int C, int H, int W, float eps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_elements = N * C * H * W;

  if (idx < num_elements) {
    int c = (idx / (H * W)) % C;
    
    float m = mean[c];
    float v = var[c];
    float g = gamma[c];
    float gg = grad_gamma[c];
    float gb = grad_beta[c];
    
    float val = input[idx];
    float dy = grad_output[idx];
    float std_inv = 1.0f / sqrtf(v + eps);
    float x_hat = (val - m) * std_inv;
    
    // Formula: dx = (1/N) * gamma * std_inv * (N * dy - sum_dy - x_hat * sum_dy_xhat)
    // sum_dy = gb, sum_dy_xhat = gg
    float M = (float)(N * H * W); // Effective batch size for stats
    
    grad_input[idx] = (g * std_inv / M) * (M * dy - gb - x_hat * gg);
  }
}


// ============================================================
// Host Wrappers
// ============================================================

void relu_backward_cuda(const float *grad_output, const float *input,
                        float *grad_input, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  relu_backward_kernel<<<blocks, BLOCK_SIZE>>>(grad_output, input, grad_input,
                                               size);
  CUDA_CHECK(cudaGetLastError());
}

void sigmoid_backward_cuda(const float *grad_output, const float *output,
                           float *grad_input, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  sigmoid_backward_kernel<<<blocks, BLOCK_SIZE>>>(grad_output, output,
                                                  grad_input, size);
  CUDA_CHECK(cudaGetLastError());
}

void tanh_backward_cuda(const float *grad_output, const float *output,
                        float *grad_input, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tanh_backward_kernel<<<blocks, BLOCK_SIZE>>>(grad_output, output, grad_input,
                                               size);
  CUDA_CHECK(cudaGetLastError());
}

void max_pool2d_forward_cuda(const float *input, float *output, int *indices,
                             int N, int C, int H, int W, int out_h, int out_w,
                             int k, int stride) {
  int num_outputs = N * C * out_h * out_w;
  int blocks = (num_outputs + BLOCK_SIZE - 1) / BLOCK_SIZE;
  max_pool2d_forward_kernel<<<blocks, BLOCK_SIZE>>>(input, output, indices, N,
                                                    C, H, W, out_h, out_w, k,
                                                    stride);
  CUDA_CHECK(cudaGetLastError());
}

void max_pool2d_backward_cuda(const float *grad_output, const int *indices,
                              float *grad_input, int N, int C, int H, int W,
                              int out_h, int out_w, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  max_pool2d_backward_kernel<<<blocks, BLOCK_SIZE>>>(grad_output, indices,
                                                     grad_input, size);
  CUDA_CHECK(cudaGetLastError());
}

void avg_pool2d_forward_cuda(const float *input, float *output, int N, int C,
                             int H, int W, int out_h, int out_w, int k,
                             int stride) {
  int num_outputs = N * C * out_h * out_w;
  int blocks = (num_outputs + BLOCK_SIZE - 1) / BLOCK_SIZE;
  avg_pool2d_forward_kernel<<<blocks, BLOCK_SIZE>>>(input, output, N, C, H, W,
                                                    out_h, out_w, k, stride);
  CUDA_CHECK(cudaGetLastError());
}

void avg_pool2d_backward_cuda(const float *grad_output, float *grad_input,
                              int N, int C, int H, int W, int out_h, int out_w,
                              int k, int stride, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  avg_pool2d_backward_kernel<<<blocks, BLOCK_SIZE>>>(
      grad_output, grad_input, N, C, H, W, out_h, out_w, k, stride);
  CUDA_CHECK(cudaGetLastError());
}

void batchnorm_forward_cuda(const float *input, const float *mean,
                            const float *var, const float *gamma,
                            const float *beta, float *output, int N, int C,
                            int H, int W, float eps) {
  int num_elements = N * C * H * W;
  int blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
  batchnorm_forward_kernel<<<blocks, BLOCK_SIZE>>>(input, mean, var, gamma,
                                                   beta, output, N, C, H, W,
                                                   eps);
  CUDA_CHECK(cudaGetLastError());
}

void batchnorm_backward_cuda(const float *grad_output, const float *input,
                             const float *mean, const float *var,
                             const float *gamma, float *grad_input,
                             float *grad_gamma, float *grad_beta, int N, int C,
                             int H, int W, float eps) {
  batchnorm_backward_cuda_device(grad_output, input, mean, var, gamma, grad_input, grad_gamma, grad_beta, N, C, H, W, eps);
}

// ============================================================
// High-level Host Wrappers Implementations
// ============================================================

void relu_backward_cuda_host(const std::vector<float> &grad_output,
                             const std::vector<float> &input,
                             std::vector<float> &grad_input) {
  int size = (int)input.size();
  size_t bytes = size * sizeof(float);
  float *d_go = (float *)cuda_malloc(bytes);
  float *d_in = (float *)cuda_malloc(bytes);
  float *d_gi = (float *)cuda_malloc(bytes);

  cuda_memcpy_host_to_device(d_go, grad_output.data(), bytes);
  cuda_memcpy_host_to_device(d_in, input.data(), bytes);

  relu_backward_cuda(d_go, d_in, d_gi, size);
  cuda_memcpy_device_to_host(grad_input.data(), d_gi, bytes);

  cuda_free(d_go);
  cuda_free(d_in);
  cuda_free(d_gi);
}

void sigmoid_backward_cuda_host(const std::vector<float> &grad_output,
                                const std::vector<float> &output,
                                std::vector<float> &grad_input) {
  int size = (int)output.size();
  size_t bytes = size * sizeof(float);
  float *d_go = (float *)cuda_malloc(bytes);
  float *d_out = (float *)cuda_malloc(bytes);
  float *d_gi = (float *)cuda_malloc(bytes);

  cuda_memcpy_host_to_device(d_go, grad_output.data(), bytes);
  cuda_memcpy_host_to_device(d_out, output.data(), bytes);

  sigmoid_backward_cuda(d_go, d_out, d_gi, size);
  cuda_memcpy_device_to_host(grad_input.data(), d_gi, bytes);

  cuda_free(d_go);
  cuda_free(d_out);
  cuda_free(d_gi);
}

void tanh_backward_cuda_host(const std::vector<float> &grad_output,
                             const std::vector<float> &output,
                             std::vector<float> &grad_input) {
  int size = (int)output.size();
  size_t bytes = size * sizeof(float);
  float *d_go = (float *)cuda_malloc(bytes);
  float *d_out = (float *)cuda_malloc(bytes);
  float *d_gi = (float *)cuda_malloc(bytes);

  cuda_memcpy_host_to_device(d_go, grad_output.data(), bytes);
  cuda_memcpy_host_to_device(d_out, output.data(), bytes);

  tanh_backward_cuda(d_go, d_out, d_gi, size);
  cuda_memcpy_device_to_host(grad_input.data(), d_gi, bytes);

  cuda_free(d_go);
  cuda_free(d_out);
  cuda_free(d_gi);
}

// Backward declarations (already existed somewhat, ensuring wrappers)

// Backward declarations (already existed somewhat, ensuring wrappers)
void relu_backward_cuda_device(const float *d_grad_output, const float *d_input,
                                float *d_grad_input, int size) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_backward_kernel<<<blocks, BLOCK_SIZE>>>(d_grad_output, d_input, d_grad_input, size);
    CUDA_CHECK(cudaGetLastError());
}
void sigmoid_backward_cuda_device(const float *d_grad_output, const float *d_output,
                                  float *d_grad_input, int size) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sigmoid_backward_kernel<<<blocks, BLOCK_SIZE>>>(d_grad_output, d_output, d_grad_input, size);
    CUDA_CHECK(cudaGetLastError());
}
void tanh_backward_cuda_device(const float *d_grad_output, const float *d_output,
                               float *d_grad_input, int size) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    tanh_backward_kernel<<<blocks, BLOCK_SIZE>>>(d_grad_output, d_output, d_grad_input, size);
    CUDA_CHECK(cudaGetLastError());
}

void max_pool2d_forward_cuda_device(const float *d_input, float *d_output, int *d_indices,
                                     int N, int C, int H, int W, int out_h, int out_w,
                                     int k, int stride) {
    int num_outputs = N * C * out_h * out_w;
    int blocks = (num_outputs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    max_pool2d_forward_kernel<<<blocks, BLOCK_SIZE>>>(d_input, d_output, d_indices, N, C, H, W, out_h, out_w, k, stride);
    CUDA_CHECK(cudaGetLastError());
}
void max_pool2d_backward_cuda_device(const float *d_grad_output, const int *d_indices,
                                      float *d_grad_input, int size) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    max_pool2d_backward_kernel<<<blocks, BLOCK_SIZE>>>(d_grad_output, d_indices, d_grad_input, size);
    CUDA_CHECK(cudaGetLastError());
}
void avg_pool2d_forward_cuda_device(const float *d_input, float *d_output,
                                     int N, int C, int H, int W, int out_h, int out_w,
                                     int k, int stride) {
    int num_outputs = N * C * out_h * out_w;
    int blocks = (num_outputs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    avg_pool2d_forward_kernel<<<blocks, BLOCK_SIZE>>>(d_input, d_output, N, C, H, W, out_h, out_w, k, stride);
    CUDA_CHECK(cudaGetLastError());
}
void avg_pool2d_backward_cuda_device(const float *d_grad_output, float *d_grad_input,
                                      int N, int C, int H, int W, int out_h, int out_w,
                                      int k, int stride) {
    int num_outputs = N * C * out_h * out_w;
    int blocks = (num_outputs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Note: avg_pool2d_backward_kernel iterates internally effectively but uses atomicAdd.
    // The kernel is launched with out_size threads.
    avg_pool2d_backward_kernel<<<blocks, BLOCK_SIZE>>>(d_grad_output, d_grad_input, N, C, H, W, out_h, out_w, k, stride);
    CUDA_CHECK(cudaGetLastError());
}
void batchnorm_forward_cuda_device(const float *d_input, const float *d_mean,
                                   const float *d_var, const float *d_gamma,
                                   const float *d_beta, float *d_output,
                                   int N, int C, int H, int W, float eps) {
    int num_elements = N * C * H * W;
    int blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    batchnorm_forward_kernel<<<blocks, BLOCK_SIZE>>>(d_input, d_mean, d_var, d_gamma, d_beta, d_output, N, C, H, W, eps);
    CUDA_CHECK(cudaGetLastError());
}

void batchnorm_training_forward_cuda_device(const float *d_input, float *d_mean,
                                            float *d_var, float *d_running_mean,
                                            float *d_running_var, const float *d_gamma,
                                            const float *d_beta, float *d_output,
                                            int N, int C, int H, int W, float eps,
                                            float momentum) {
  // 1. Compute stats
  batchnorm_stats_kernel<<<C, 256>>>(d_input, d_mean, d_var, N, C, H, W);
  CUDA_CHECK(cudaGetLastError());

  // 2. Update running stats
  int blocks_c = (C + BLOCK_SIZE - 1) / BLOCK_SIZE;
  batchnorm_update_stats_kernel<<<blocks_c, BLOCK_SIZE>>>(d_running_mean, d_running_var, d_mean, d_var, momentum, C);
  CUDA_CHECK(cudaGetLastError());

  // 3. Normalize and scale
  batchnorm_forward_cuda(d_input, d_mean, d_var, d_gamma, d_beta, d_output, N, C, H, W, eps);
}

void batchnorm_backward_cuda_device(const float *d_grad_output, const float *d_input,
                                    const float *d_mean, const float *d_var,
                                    const float *d_gamma, float *d_grad_input,
                                    float *d_grad_gamma, float *d_grad_beta,
                                    int N, int C, int H, int W, float eps) {
  // 1. Reduce gradients
  batchnorm_backward_reduce_kernel<<<C, 256>>>(d_grad_output, d_input, d_mean, d_var, d_grad_gamma, d_grad_beta, N, C, H, W, eps);
  CUDA_CHECK(cudaGetLastError());
  
  // 2. Apply gradients to input
  int num_elements = N * C * H * W;
  int blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
  batchnorm_backward_apply_kernel<<<blocks, BLOCK_SIZE>>>(d_grad_output, d_input, d_mean, d_var, d_gamma, d_grad_gamma, d_grad_beta, d_grad_input, N, C, H, W, eps);
  CUDA_CHECK(cudaGetLastError());
}


void max_pool2d_forward_cuda_host(const std::vector<float> &input,
                                  std::vector<float> &output,
                                  std::vector<int> &indices, int N, int C,
                                  int H, int W, int k, int stride) {
  int in_size = (int)input.size();
  int out_h = (H - k) / stride + 1;
  int out_w = (W - k) / stride + 1;
  int out_size = N * C * out_h * out_w;

  size_t in_bytes = in_size * sizeof(float);
  size_t out_bytes = out_size * sizeof(float);
  size_t idx_bytes = out_size * sizeof(int);

  float *d_in = (float *)cuda_malloc(in_bytes);
  float *d_out = (float *)cuda_malloc(out_bytes);
  int *d_idx = (int *)cuda_malloc(idx_bytes);

  cuda_memcpy_host_to_device(d_in, input.data(), in_bytes);

  max_pool2d_forward_cuda(d_in, d_out, d_idx, N, C, H, W, out_h, out_w, k,
                          stride);
  cuda_memcpy_device_to_host(output.data(), d_out, out_bytes);
  cuda_memcpy_device_to_host(indices.data(), d_idx, idx_bytes);

  cuda_free(d_in);
  cuda_free(d_out);
  cuda_free(d_idx);
}

void max_pool2d_backward_cuda_host(const std::vector<float> &grad_output,
                                   const std::vector<int> &indices,
                                   std::vector<float> &grad_input, int N, int C,
                                   int H, int W, int k, int stride) {
  int out_h = (H - k) / stride + 1;
  int out_w = (W - k) / stride + 1;
  int out_size = N * C * out_h * out_w;
  int in_size = N * C * H * W;

  size_t go_bytes = out_size * sizeof(float);
  size_t idx_bytes = out_size * sizeof(int);
  size_t gi_bytes = in_size * sizeof(float);

  float *d_go = (float *)cuda_malloc(go_bytes);
  int *d_idx = (int *)cuda_malloc(idx_bytes);
  float *d_gi = (float *)cuda_malloc(gi_bytes);

  cuda_memcpy_host_to_device(d_go, grad_output.data(), go_bytes);
  cuda_memcpy_host_to_device(d_idx, indices.data(), idx_bytes);
  
  // Zero out grad_input on device
  CUDA_CHECK(cudaMemset(d_gi, 0, gi_bytes));

  max_pool2d_backward_cuda(d_go, d_idx, d_gi, N, C, H, W, out_h, out_w, out_size);
  cuda_memcpy_device_to_host(grad_input.data(), d_gi, gi_bytes);

  cuda_free(d_go);
  cuda_free(d_idx);
  cuda_free(d_gi);
}

void avg_pool2d_forward_cuda_host(const std::vector<float> &input,
                                  std::vector<float> &output, int N, int C,
                                  int H, int W, int k, int stride) {
  int in_size = (int)input.size();
  int out_h = (H - k) / stride + 1;
  int out_w = (W - k) / stride + 1;
  int out_size = N * C * out_h * out_w;

  size_t in_bytes = in_size * sizeof(float);
  size_t out_bytes = out_size * sizeof(float);

  float *d_in = (float *)cuda_malloc(in_bytes);
  float *d_out = (float *)cuda_malloc(out_bytes);

  cuda_memcpy_host_to_device(d_in, input.data(), in_bytes);

  avg_pool2d_forward_cuda(d_in, d_out, N, C, H, W, out_h, out_w, k, stride);
  cuda_memcpy_device_to_host(output.data(), d_out, out_bytes);

  cuda_free(d_in);
  cuda_free(d_out);
}

void avg_pool2d_backward_cuda_host(const std::vector<float> &grad_output,
                                   std::vector<float> &grad_input, int N, int C,
                                   int H, int W, int k, int stride) {
  int out_h = (H - k) / stride + 1;
  int out_w = (W - k) / stride + 1;
  int out_size = N * C * out_h * out_w;
  int in_size = N * C * H * W;

  size_t go_bytes = out_size * sizeof(float);
  size_t gi_bytes = in_size * sizeof(float);

  float *d_go = (float *)cuda_malloc(go_bytes);
  float *d_gi = (float *)cuda_malloc(gi_bytes);

  cuda_memcpy_host_to_device(d_go, grad_output.data(), go_bytes);
  // Zero out grad_input on device
  CUDA_CHECK(cudaMemset(d_gi, 0, gi_bytes));

  avg_pool2d_backward_cuda(d_go, d_gi, N, C, H, W, out_h, out_w, k, stride, out_size);
  cuda_memcpy_device_to_host(grad_input.data(), d_gi, gi_bytes);

  cuda_free(d_go);
  cuda_free(d_gi);
}

void batchnorm_forward_cuda_host(const std::vector<float> &input,
                                 const std::vector<float> &mean,
                                 const std::vector<float> &var,
                                 const std::vector<float> &gamma,
                                 const std::vector<float> &beta,
                                 std::vector<float> &output, int N, int C,
                                 int H, int W, float eps) {
  int num_elements = N * C * H * W;
  size_t data_bytes = num_elements * sizeof(float);
  size_t param_bytes = C * sizeof(float);

  float *d_in = (float *)cuda_malloc(data_bytes);
  float *d_out = (float *)cuda_malloc(data_bytes);
  float *d_m = (float *)cuda_malloc(param_bytes);
  float *d_v = (float *)cuda_malloc(param_bytes);
  float *d_g = (float *)cuda_malloc(param_bytes);
  float *d_b = (float *)cuda_malloc(param_bytes);

  cuda_memcpy_host_to_device(d_in, input.data(), data_bytes);
  cuda_memcpy_host_to_device(d_m, mean.data(), param_bytes);
  cuda_memcpy_host_to_device(d_v, var.data(), param_bytes);
  cuda_memcpy_host_to_device(d_g, gamma.data(), param_bytes);
  cuda_memcpy_host_to_device(d_b, beta.data(), param_bytes);

  batchnorm_forward_cuda(d_in, d_m, d_v, d_g, d_b, d_out, N, C, H, W, eps);
  cuda_memcpy_device_to_host(output.data(), d_out, data_bytes);

  cuda_free(d_in);
  cuda_free(d_out);
  cuda_free(d_m);
  cuda_free(d_v);
  cuda_free(d_g);
  cuda_free(d_b);
}

void batchnorm_backward_cuda_host(const std::vector<float> &grad_output,
                                  const std::vector<float> &input,
                                  const std::vector<float> &mean,
                                  const std::vector<float> &var,
                                  const std::vector<float> &gamma,
                                  std::vector<float> &grad_input,
                                  std::vector<float> &grad_gamma,
                                  std::vector<float> &grad_beta, int N, int C,
                                  int H, int W, float eps) {
  int num_elements = N * C * H * W;
  size_t data_bytes = num_elements * sizeof(float);
  size_t param_bytes = C * sizeof(float);

  float *d_go = (float *)cuda_malloc(data_bytes);
  float *d_in = (float *)cuda_malloc(data_bytes);
  float *d_gi = (float *)cuda_malloc(data_bytes);
  float *d_m = (float *)cuda_malloc(param_bytes);
  float *d_v = (float *)cuda_malloc(param_bytes);
  float *d_g = (float *)cuda_malloc(param_bytes);
  float *d_gg = (float *)cuda_malloc(param_bytes);
  float *d_gb = (float *)cuda_malloc(param_bytes);

  cuda_memcpy_host_to_device(d_go, grad_output.data(), data_bytes);
  cuda_memcpy_host_to_device(d_in, input.data(), data_bytes);
  cuda_memcpy_host_to_device(d_m, mean.data(), param_bytes);
  cuda_memcpy_host_to_device(d_v, var.data(), param_bytes);
  cuda_memcpy_host_to_device(d_g, gamma.data(), param_bytes);
  
  CUDA_CHECK(cudaMemset(d_gg, 0, param_bytes));
  CUDA_CHECK(cudaMemset(d_gb, 0, param_bytes));

  batchnorm_backward_cuda(d_go, d_in, d_m, d_v, d_g, d_gi, d_gg, d_gb, N, C, H,
                          W, eps);
  cuda_memcpy_device_to_host(grad_input.data(), d_gi, data_bytes);
  cuda_memcpy_device_to_host(grad_gamma.data(), d_gg, param_bytes);
  cuda_memcpy_device_to_host(grad_beta.data(), d_gb, param_bytes);

  cuda_free(d_go);
  cuda_free(d_in);
  cuda_free(d_gi);
  cuda_free(d_m);
  cuda_free(d_v);
  cuda_free(d_g);
  cuda_free(d_gg);
  cuda_free(d_gb);
}







void im2col_cuda_device(const float* d_im, const int N, const int C, const int H, const int W,
                        const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
                        const int stride_h, const int stride_w, const int out_h, const int out_w,
                        float* d_col) {
  int num_kernels = N * out_h * out_w * C * kernel_h * kernel_w;
  int blocks = (num_kernels + BLOCK_SIZE - 1) / BLOCK_SIZE;
  im2col_kernel<<<blocks, BLOCK_SIZE>>>(num_kernels, d_im, N, C, H, W, kernel_h, kernel_w,
                                        pad_h, pad_w, stride_h, stride_w, out_h, out_w, d_col);
  CUDA_CHECK(cudaGetLastError());
}

void col2im_cuda_device(const float* d_col, const int N, const int C, const int H, const int W,
                        const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
                        const int stride_h, const int stride_w, const int out_h, const int out_w,
                        float* d_im) {
  int num_kernels = N * out_h * out_w * C * kernel_h * kernel_w;
  int blocks = (num_kernels + BLOCK_SIZE - 1) / BLOCK_SIZE;
  col2im_kernel<<<blocks, BLOCK_SIZE>>>(num_kernels, d_col, N, C, H, W, kernel_h, kernel_w,
                                        pad_h, pad_w, stride_h, stride_w, out_h, out_w, d_im);
  CUDA_CHECK(cudaGetLastError());
}

void add_bias_shared_cuda_device(const float* d_bias, float* d_output, int N, int C, int H, int W) {
  int num_elements = N * C * H * W;
  int blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
  add_bias_shared_kernel<<<blocks, BLOCK_SIZE>>>(d_bias, d_output, N, C, H, W);
  CUDA_CHECK(cudaGetLastError());
}

void bias_backward_cuda_device(const float* d_grad_output, float* d_grad_bias, int N, int C, int H, int W) {
  bias_backward_kernel<<<C, 256>>>(d_grad_output, d_grad_bias, N, C, H, W);
  CUDA_CHECK(cudaGetLastError());
}

void sgd_update_cuda_device(float* d_params, float* d_velocity, const float* d_grads,
                            float lr, float momentum, float weight_decay, bool nesterov, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  sgd_update_kernel<<<blocks, BLOCK_SIZE>>>(d_params, d_velocity, d_grads, lr, momentum, weight_decay, nesterov, size);
  CUDA_CHECK(cudaGetLastError());
}

void adam_update_cuda_device(float* d_params, float* d_m, float* d_v, const float* d_grads,
                             float lr, float beta1, float beta2, float eps, float weight_decay, int t, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  adam_update_kernel<<<blocks, BLOCK_SIZE>>>(d_params, d_m, d_v, d_grads, lr, beta1, beta2, eps, weight_decay, t, size);
  CUDA_CHECK(cudaGetLastError());
}

// Dropout Kernel (applies mask)
__global__ void dropout_kernel(const float* input, const float* mask, float* output, float scale, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input[idx] * mask[idx] * scale;
  }
}

__global__ void dropout_backward_kernel(const float* grad_output, const float* mask, float* grad_input, float scale, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    grad_input[idx] = grad_output[idx] * mask[idx] * scale;
  }
}

// Transpose 2D Kernel
__global__ void transpose2d_kernel(const float* in, float* out, int rows, int cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (r < rows && c < cols) {
        out[c * rows + r] = in[r * cols + c];
    }
}

// Permute 4D Kernel (Generic but optimized for small dimensions)
__global__ void permute4d_kernel(const float* in, float* out, 
                                int N, int C, int H, int W,
                                int s0, int s1, int s2, int s3, // original strides
                                int d0, int d1, int d2, int d3, // permutation
                                int n0, int n1, int n2, int n3) // new shape
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    
    if (idx < total) {
        // Compute logical indices in the new shape
        int i3 = idx % n3;
        int tmp2 = idx / n3;
        int i2 = tmp2 % n2;
        int tmp1 = tmp2 / n2;
        int i1 = tmp1 % n1;
        int i0 = tmp1 / n1;
        
        // Map new indices back to original indices using the permutation
        int old_idx_arr[4];
        old_idx_arr[d0] = i0;
        old_idx_arr[d1] = i1;
        old_idx_arr[d2] = i2;
        old_idx_arr[d3] = i3;
        
        // Compute original flat offset
        int original_offset = old_idx_arr[0] * s0 + 
                             old_idx_arr[1] * s1 + 
                             old_idx_arr[2] * s2 + 
                             old_idx_arr[3] * s3;
        
        out[idx] = in[original_offset];
    }
}

// LogSoftmax Kernel (Naive implementation: 1 thread per row for simplicity, or 1 block per row)
// optimized for small C
__global__ void log_softmax_kernel(const float* input, float* output, int N, int C) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    // Find max
    float max_val = -1e30f;
    for (int c = 0; c < C; ++c) {
      float val = input[n * C + c];
      if (val > max_val) max_val = val;
    }

    // Sum exp
    float sum_exp = 0.0f;
    for (int c = 0; c < C; ++c) {
      sum_exp += expf(input[n * C + c] - max_val);
    }
    float log_sum = logf(sum_exp);

    // Compute log_prob
    for (int c = 0; c < C; ++c) {
      output[n * C + c] = input[n * C + c] - max_val - log_sum;
    }
  }
}

// NLL Loss Kernel (Forward) -> Returns element-wise losses to be reduced later, or reduce here?
// Let's return element-wise losses (size N) to avoid complex reduction here, then use sum_cuda_device.
__global__ void nll_loss_kernel(const float* log_probs, const int* targets, float* losses, int N, int C) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    int t = targets[n];
    if (t >= 0 && t < C) {
      losses[n] = -log_probs[n * C + t];
    } else {
      losses[n] = 0.0f; // Ignore invalid targets
    }
  }
}

// NLL Loss Backward
__global__ void nll_loss_backward_kernel(const int* targets, float* grad_input, int N, int C, float scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int size = N * C;
  if (idx < size) {
    int n = idx / C;
    int c = idx % C;
    if (targets[n] == c) {
      grad_input[idx] = -scale; 
    } else {
      grad_input[idx] = 0.0f;
    }
  }
}

void dropout_cuda_device(const float* d_input, const float* d_mask, float* d_output, float scale, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dropout_kernel<<<blocks, BLOCK_SIZE>>>(d_input, d_mask, d_output, scale, size);
  CUDA_CHECK(cudaGetLastError());
}

void dropout_backward_cuda_device(const float* d_grad_output, const float* d_mask, float* d_grad_input, float scale, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dropout_backward_kernel<<<blocks, BLOCK_SIZE>>>(d_grad_output, d_mask, d_grad_input, scale, size);
  CUDA_CHECK(cudaGetLastError());
}

void log_softmax_cuda_device(const float* d_input, float* d_output, int N, int C) {
  int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  log_softmax_kernel<<<blocks, BLOCK_SIZE>>>(d_input, d_output, N, C);
  CUDA_CHECK(cudaGetLastError());
}

void nll_loss_cuda_device(const float* d_log_probs, const int* d_targets, float* d_losses, int N, int C) {
  int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  nll_loss_kernel<<<blocks, BLOCK_SIZE>>>(d_log_probs, d_targets, d_losses, N, C);
  CUDA_CHECK(cudaGetLastError());
}

void nll_loss_backward_cuda_device(const int* d_targets, float* d_grad_input, int N, int C) {
  int size = N * C;
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  nll_loss_backward_kernel<<<blocks, BLOCK_SIZE>>>(d_targets, d_grad_input, N, C, 1.0f);
  CUDA_CHECK(cudaGetLastError());
}

void transpose2d_cuda_device(const float* d_in, float* d_out, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    transpose2d_kernel<<<grid, block>>>(d_in, d_out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void permute4d_cuda_device(const float* d_in, float* d_out, int N, int C, int H, int W,
                           int d0, int d1, int d2, int d3) {
    int total = N * C * H * W;
    if (total <= 0) return;
    
    // Original strides
    int s3 = 1;
    int s2 = W;
    int s1 = H * W;
    int s0 = C * H * W;
    
    // New shape
    int shape_arr[4] = {N, C, H, W};
    int n0 = shape_arr[d0];
    int n1 = shape_arr[d1];
    int n2 = shape_arr[d2];
    int n3 = shape_arr[d3];
    
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    permute4d_kernel<<<blocks, BLOCK_SIZE>>>(d_in, d_out, N, C, H, W,
                                             s0, s1, s2, s3,
                                             d0, d1, d2, d3,
                                             n0, n1, n2, n3);
    CUDA_CHECK(cudaGetLastError());
}


} // namespace cuda
} // namespace deepnet

#endif // USE_CUDA
