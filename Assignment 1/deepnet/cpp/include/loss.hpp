#pragma once

#include "tensor.hpp"

namespace deepnet {

// Cross Entropy Loss
class CrossEntropyLoss {
public:
  CrossEntropyLoss() = default;

  // Compute loss: input shape [batch, num_classes], target shape [batch]
  TensorPtr forward(const TensorPtr &input, const std::vector<int> &targets);

  // Get gradient of loss w.r.t. input logits (computed during forward)
  TensorPtr get_input_grad() const { return input_grad; }

private:
  TensorPtr softmax(const TensorPtr &input);
  TensorPtr log_softmax(const TensorPtr &input);
  TensorPtr input_grad; // dL/d(logits) stored during forward
};

// Mean Squared Error Loss
class MSELoss {
public:
  MSELoss() = default;

  TensorPtr forward(const TensorPtr &input, const TensorPtr &target);
};

} // namespace deepnet
