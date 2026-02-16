#pragma once

#include <memory>
#include <random>
#include <string>
#include <vector>

#ifdef USE_CUDA
#include "cuda/cuda_utils.hpp"
#endif

namespace deepnet {
void manual_seed(unsigned int seed);
std::mt19937 &get_generator();

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

// Autograd function for backward pass
struct AutogradFunction {
  virtual ~AutogradFunction() = default;
  virtual std::vector<TensorPtr> backward(const TensorPtr &grad_output) = 0;
  std::vector<TensorPtr> inputs;
  std::vector<bool> requires_grad;
};

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
  // Data storage
  std::vector<float> data;      // CPU data (only allocated if !is_cuda)
  std::vector<float> grad;       // CPU grad (only allocated if !is_cuda)
#ifdef USE_CUDA
  float *d_data;                 // Device data pointer (only allocated if is_cuda)
  float *d_grad;                 // Device grad pointer (only allocated if is_cuda)
  bool cpu_dirty;                // True if CPU data is out of sync
  bool cuda_dirty;                // True if CUDA data is out of sync
#endif
  std::vector<int> shape;
  std::vector<int> strides;
  bool requires_grad;
  bool is_cuda;

  // Autograd
  std::shared_ptr<AutogradFunction> grad_fn;
  
  // Destructor to free device memory
  ~Tensor();

  // Constructors
  Tensor();
  Tensor(const std::vector<int> &shape, bool requires_grad = false,
         bool cuda = false);
  Tensor(const std::vector<float> &data, const std::vector<int> &shape,
         bool requires_grad = false, bool cuda = false);

  // Factory methods
  static TensorPtr zeros(const std::vector<int> &shape,
                         bool requires_grad = false, bool cuda = false);
  static TensorPtr ones(const std::vector<int> &shape,
                        bool requires_grad = false, bool cuda = false);
  static TensorPtr randn(const std::vector<int> &shape, float mean = 0.0f,
                         float std = 1.0f, bool requires_grad = false,
                         bool cuda = false);
  static TensorPtr from_data(const std::vector<float> &data,
                             const std::vector<int> &shape,
                             bool requires_grad = false, bool cuda = false);

  // Shape operations
  int size() const;
  int size(int dim) const;
  int numel() const;
  int ndim() const;
  TensorPtr reshape(const std::vector<int> &new_shape);
  TensorPtr view(const std::vector<int> &new_shape);
  TensorPtr transpose(int dim0, int dim1);
  TensorPtr permute(const std::vector<int> &dims);
  TensorPtr flatten(int start_dim = 0, int end_dim = -1);

  // Im2Col / Col2Im
  TensorPtr im2col(int kernel_size, int stride, int padding);
  TensorPtr col2im(const std::vector<int> &output_shape, int kernel_size,
                   int stride, int padding);

  // Element-wise operations
  TensorPtr operator+(const TensorPtr &other);
  TensorPtr operator-(const TensorPtr &other);
  TensorPtr operator*(const TensorPtr &other);
  TensorPtr operator/(const TensorPtr &other);
  TensorPtr add(const TensorPtr &other);
  TensorPtr sub(const TensorPtr &other);
  TensorPtr mul(const TensorPtr &other);
  TensorPtr div(const TensorPtr &other);
  TensorPtr add_scalar(float scalar);
  TensorPtr mul_scalar(float scalar);

  // Matrix operations
  TensorPtr matmul(const TensorPtr &other);
  TensorPtr mm(const TensorPtr &other);

  // Reduction operations
  TensorPtr sum(int dim = -1, bool keepdim = false);
  TensorPtr mean(int dim = -1, bool keepdim = false);
  TensorPtr max(int dim = -1, bool keepdim = false);
  TensorPtr min(int dim = -1, bool keepdim = false);

  // Activation functions (forward pass)
  TensorPtr relu();
  TensorPtr leaky_relu(float negative_slope = 0.01f);
  TensorPtr tanh_();
  TensorPtr sigmoid();

  // Math operations
  TensorPtr exp();
  TensorPtr log();
  TensorPtr pow(float exponent);
  TensorPtr sqrt();

  // Autograd operations
  void backward(const TensorPtr &gradient = nullptr);
  void zero_grad();
  TensorPtr detach();
  void accumulate_grad(const TensorPtr &grad_in);

  // CUDA operations
  void cuda();
  void cpu();
  TensorPtr to(bool cuda);

  // Utility
  void fill_(float value);
  void copy_(const std::shared_ptr<Tensor> &other);
  void uniform_(float min, float max);
  void normal_(float mean, float std);
  TensorPtr clone();
  std::string shape_str() const;
  void print(const std::string &name = "") const;

  // Data accessors
  float &at(const std::vector<int> &indices);
  const float &at(const std::vector<int> &indices) const;
  float *data_ptr();
  const float *data_ptr() const;
  float *grad_ptr();
  
  // Memory sync methods
  void sync_to_cpu();  // Copy device data to CPU
  void sync_to_cuda(); // Copy CPU data to device

private:
  void compute_strides();
  int compute_offset(const std::vector<int> &indices) const;
  void check_shape_compatible(const TensorPtr &other) const;
  void allocate_device_memory();
  void free_device_memory();
};

} // namespace deepnet
