#pragma once

#include "../tensor.hpp"
#include <memory>

namespace deepnet {

// Base Layer class
class Layer {
public:
  virtual ~Layer() = default;
  virtual TensorPtr forward(const TensorPtr &input) = 0;
  // Backward pass: receives gradient w.r.t. output, returns gradient w.r.t. input
  // Also accumulates gradients into parameter .grad buffers
  virtual TensorPtr backward(const TensorPtr &grad_output) { return grad_output; }
  virtual std::vector<TensorPtr> parameters() { return {}; }
  virtual void train() { training = true; }
  virtual void eval() { training = false; }

protected:
  bool training = true;
};

// Conv2D Layer
class Conv2D : public Layer {
public:
  Conv2D(int in_channels, int out_channels, int kernel_size, int stride = 1,
         int padding = 0, bool bias = true);

  TensorPtr forward(const TensorPtr &input) override;
  TensorPtr backward(const TensorPtr &grad_output) override;
  std::vector<TensorPtr> parameters() override;

private:
  int in_channels, out_channels, kernel_size, stride, padding;
  TensorPtr
      weight; // Shape: [out_channels, in_channels, kernel_size, kernel_size]
  TensorPtr bias_; // Shape: [out_channels]
  bool use_bias;
  TensorPtr last_input; // Cached for backward
  TensorPtr last_col;   // Cached for im2col output
};

// Linear (Fully Connected) Layer
class Linear : public Layer {
public:
  Linear(int in_features, int out_features, bool bias = true);

  TensorPtr forward(const TensorPtr &input) override;
  TensorPtr backward(const TensorPtr &grad_output) override;
  std::vector<TensorPtr> parameters() override;

private:
  int in_features, out_features;
  TensorPtr weight; // Shape: [out_features, in_features]
  TensorPtr bias_;  // Shape: [out_features]
  bool use_bias;
  TensorPtr last_input; // Cached for backward [batch, in_features]

  void xavier_init();
  void he_init();
};

// ReLU Activation
class ReLU : public Layer {
public:
  ReLU() = default;
  TensorPtr forward(const TensorPtr &input) override;
  TensorPtr backward(const TensorPtr &grad_output) override;

private:
  TensorPtr last_input; // Cached for backward
};

// LeakyReLU Activation
class LeakyReLU : public Layer {
public:
  LeakyReLU(float negative_slope = 0.01f) : negative_slope(negative_slope) {}
  TensorPtr forward(const TensorPtr &input) override;
  TensorPtr backward(const TensorPtr &grad_output) override;

private:
  float negative_slope;
  TensorPtr last_input;
};

// Tanh Activation
class Tanh : public Layer {
public:
  Tanh() = default;
  TensorPtr forward(const TensorPtr &input) override;
  TensorPtr backward(const TensorPtr &grad_output) override;

private:
  TensorPtr last_output; // Cached for backward
};

// Sigmoid Activation
class Sigmoid : public Layer {
public:
  Sigmoid() = default;
  TensorPtr forward(const TensorPtr &input) override;
  TensorPtr backward(const TensorPtr &grad_output) override;

private:
  TensorPtr last_output; // Cached for backward
};

// Dropout Layer
class Dropout : public Layer {
public:
  Dropout(float p = 0.5f) : p(p) {}
  TensorPtr forward(const TensorPtr &input) override;
  TensorPtr backward(const TensorPtr &grad_output) override;

private:
  float p;
  TensorPtr mask; // Cached for backward
};

// Flatten Layer
class Flatten : public Layer {
public:
  Flatten(int start_dim = 1, int end_dim = -1)
      : start_dim(start_dim), end_dim(end_dim) {}

  TensorPtr forward(const TensorPtr &input) override;
  TensorPtr backward(const TensorPtr &grad_output) override;

private:
  int start_dim, end_dim;
  std::vector<int> original_shape; // Cached for backward
};

} // namespace deepnet
