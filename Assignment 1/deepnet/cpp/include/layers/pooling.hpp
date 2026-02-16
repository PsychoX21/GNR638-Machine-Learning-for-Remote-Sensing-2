#pragma once

#include "../tensor.hpp"
#include "layer.hpp"

namespace deepnet {

// MaxPool2D Layer
class MaxPool2D : public Layer {
public:
  MaxPool2D(int kernel_size, int stride = -1);

  TensorPtr forward(const TensorPtr &input) override;
  TensorPtr backward(const TensorPtr &grad_output) override;

private:
  int kernel_size, stride;
  std::vector<int> max_indices; // For backward pass
  std::vector<int> input_shape; // Cached input shape
};

// AvgPool2D Layer
class AvgPool2D : public Layer {
public:
  AvgPool2D(int kernel_size, int stride = -1);

  TensorPtr forward(const TensorPtr &input) override;
  TensorPtr backward(const TensorPtr &grad_output) override;

private:
  int kernel_size, stride;
  std::vector<int> input_shape; // Cached input shape
};

// AdaptiveAvgPool2D Layer
class AdaptiveAvgPool2D : public Layer {
public:
  AdaptiveAvgPool2D(int output_size) : output_size(output_size) {}

  TensorPtr forward(const TensorPtr &input) override;

private:
  int output_size;
};

} // namespace deepnet
