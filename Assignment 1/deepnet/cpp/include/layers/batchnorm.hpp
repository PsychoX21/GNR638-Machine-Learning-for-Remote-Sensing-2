#pragma once

#include "../tensor.hpp"
#include "layer.hpp"

namespace deepnet {

// Dropout definition removed (moved to layer.hpp)

// BatchNorm2D Layer
class BatchNorm2D : public Layer {
public:
  BatchNorm2D(int num_features, float eps = 1e-5f, float momentum = 0.1f);

  TensorPtr forward(const TensorPtr &input) override;
  TensorPtr backward(const TensorPtr &grad_output) override;
  std::vector<TensorPtr> parameters() override;

private:
  int num_features;
  float eps, momentum;

  TensorPtr gamma; // Scale parameter
  TensorPtr beta;  // Shift parameter
  TensorPtr running_mean;
  TensorPtr running_var;
  TensorPtr last_input;   // Cached for backward
  TensorPtr normalized;   // Cached x_hat for backward

  std::vector<float> batch_std_inv; // Cached 1/sqrt(var+eps) (CPU)
  TensorPtr saved_mean;   // MEAN of batch (CUDA)
  TensorPtr saved_var;    // VAR of batch (CUDA)
};

// BatchNorm1D Layer (for Linear layers)
class BatchNorm1D : public Layer {
public:
  BatchNorm1D(int num_features, float eps = 1e-5f, float momentum = 0.1f);

  TensorPtr forward(const TensorPtr &input) override;
  TensorPtr backward(const TensorPtr &grad_output) override;
  std::vector<TensorPtr> parameters() override;

private:
  int num_features;
  float eps, momentum;

  TensorPtr gamma;
  TensorPtr beta;
  TensorPtr running_mean;
  TensorPtr running_var;
  TensorPtr last_input;
  TensorPtr normalized;

  std::vector<float> batch_std_inv;
  TensorPtr saved_mean;
  TensorPtr saved_var;
};

} // namespace deepnet
