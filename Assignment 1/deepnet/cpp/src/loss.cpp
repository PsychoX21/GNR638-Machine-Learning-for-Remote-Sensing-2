#include "loss.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#ifdef USE_CUDA
#include "cuda/cuda_ops.hpp"
#endif


namespace deepnet {

// Softmax helper
TensorPtr CrossEntropyLoss::softmax(const TensorPtr &input) {
  if (!input) throw std::runtime_error("CrossEntropyLoss::softmax: input is null");
  // sync_to_cpu is now handled by input->data[index] access if it hits data_ptr()
  // Wait, input->data is a public vector, direct access DOES NOT call data_ptr().
  // So for safety when using input->data directly, we STILL need manual sync or use data_ptr().
  // However, input->data access in CrossEntropyLoss::softmax is on the CPU loop.
  // We should call sync_to_cpu() here to be safe since we use input->data.
  input->sync_to_cpu();
  // Input shape: [batch, num_classes]
  int batch = input->shape[0];
  int num_classes = input->shape[1];

  auto output = Tensor::zeros(input->shape, false, input->is_cuda);

  for (int b = 0; b < batch; ++b) {
    // Find max for numerical stability
    float max_val = -std::numeric_limits<float>::infinity();
    for (int c = 0; c < num_classes; ++c) {
      max_val = std::max(max_val, input->data[b * num_classes + c]);
    }

    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int c = 0; c < num_classes; ++c) {
      float exp_val = std::exp(input->data[b * num_classes + c] - max_val);
      output->data[b * num_classes + c] = exp_val;
      sum_exp += exp_val;
    }

    // Normalize
    for (int c = 0; c < num_classes; ++c) {
      output->data[b * num_classes + c] /= sum_exp;
    }
  }

  return output;
}

// Log-Softmax helper (more numerically stable)
TensorPtr CrossEntropyLoss::log_softmax(const TensorPtr &input) {
  if (!input) throw std::runtime_error("CrossEntropyLoss::log_softmax: input is null");
  input->sync_to_cpu();
  // Input shape: [batch, num_classes]
  int batch = input->shape[0];
  int num_classes = input->shape[1];

  auto output = Tensor::zeros(input->shape, false, input->is_cuda);

  for (int b = 0; b < batch; ++b) {
    // Find max for numerical stability
    float max_val = -std::numeric_limits<float>::infinity();
    for (int c = 0; c < num_classes; ++c) {
      max_val = std::max(max_val, input->data[b * num_classes + c]);
    }

    // Compute log-sum-exp
    float sum_exp = 0.0f;
    for (int c = 0; c < num_classes; ++c) {
      sum_exp += std::exp(input->data[b * num_classes + c] - max_val);
    }
    float log_sum_exp = max_val + std::log(sum_exp);

    // Compute log probabilities
    for (int c = 0; c < num_classes; ++c) {
      output->data[b * num_classes + c] =
          input->data[b * num_classes + c] - log_sum_exp;
    }
  }

  return output;
}

// Cross Entropy Loss
TensorPtr CrossEntropyLoss::forward(const TensorPtr &input,
                                    const std::vector<int> &targets) {
  if (!input) throw std::runtime_error("CrossEntropyLoss::forward: input is null");
  if (input->shape.size() != 2) {
    throw std::runtime_error(
        "CrossEntropyLoss expects 2D input [batch, num_classes]");
  }

  int batch = input->shape[0];
  int num_classes = input->shape[1];

  if (static_cast<int>(targets.size()) != batch) {
    throw std::runtime_error("Target size mismatch");
  }

#ifdef USE_CUDA
  if (input->is_cuda) {
      // 1. LogSoftmax
      auto log_probs = Tensor::zeros(input->shape, false, true);
      cuda::log_softmax_cuda_device(input->data_ptr(), log_probs->data_ptr(), batch, num_classes);
      
      // 2. Targets to device
      int* d_targets;
      // cuda_ops.hpp exposes cuda_malloc but returns void*
      d_targets = (int*)cuda::cuda_malloc(batch * sizeof(int));
      cuda::cuda_memcpy_host_to_device(d_targets, targets.data(), batch * sizeof(int));

      // 3. NLL Loss
      auto losses = Tensor::zeros({batch}, false, true);
      cuda::nll_loss_cuda_device(log_probs->data_ptr(), d_targets, losses->data_ptr(), batch, num_classes);
      
      auto sum_loss = losses->sum();
      auto mean_loss_tensor = sum_loss->mul_scalar(1.0f / batch);
      
      // 5. Gradient: softmax(logits) - one_hot(targets)
      // probs = exp(log_probs)
      auto probs = log_probs->exp();
      
      // Create offset tensor: -1 at target, 0 elsewhere
      auto offset = Tensor::zeros(input->shape, false, true);
      cuda::nll_loss_backward_cuda_device(d_targets, offset->data_ptr(), batch, num_classes);
      
      // Grad = (probs + offset) / batch
      input_grad = probs->add(offset)->mul_scalar(1.0f / batch);
      
      
      
      cuda::cuda_free(d_targets);
      
      return mean_loss_tensor;
  }
#endif

  // Compute softmax probabilities (needed for both loss and gradient)
  auto probs = softmax(input);

  // Compute log-softmax for numerically stable loss
  auto log_probs = log_softmax(input);

  // Compute negative log likelihood
  float total_loss = 0.0f;
  for (int b = 0; b < batch; ++b) {
    int target_class = targets[b];
    if (target_class < 0 || target_class >= num_classes) {
      throw std::runtime_error("Target class out of range");
    }
    total_loss -= log_probs->data[b * num_classes + target_class];
  }

  // Compute gradient: dL/d(logits) = softmax(logits) - one_hot(targets)
  // divided by batch_size for mean reduction
  input_grad = Tensor::zeros(input->shape, false, input->is_cuda);
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < num_classes; ++c) {
      float grad_val = probs->data[b * num_classes + c];
      if (c == targets[b]) {
        grad_val -= 1.0f;
      }
      input_grad->data[b * num_classes + c] = grad_val / batch;
    }
  }

  // Return mean loss
  float mean_loss = total_loss / batch;
  return Tensor::from_data({mean_loss}, {1}, false, input->is_cuda);
}

// MSE Loss
TensorPtr MSELoss::forward(const TensorPtr &input, const TensorPtr &target) {
  if (!input || !target) throw std::runtime_error("MSELoss::forward: input or target is null");
  if (input->shape != target->shape) {
    throw std::runtime_error("MSELoss: input and target shapes must match");
  }

#ifdef USE_CUDA
  if (input->is_cuda) {
      // (input - target)^2
      auto diff = input->sub(target);
      auto sq_diff = diff->pow(2.0f);
      auto output = sq_diff->mean();
      return output;
  }
#endif

  float sum_sq_error = 0.0f;
  for (size_t i = 0; i < input->data.size(); ++i) {
    float diff = input->data[i] - target->data[i];
    sum_sq_error += diff * diff;
  }

  float mean_loss = sum_sq_error / input->data.size();
  return Tensor::from_data({mean_loss}, {1}, false, input->is_cuda);
}


} // namespace deepnet
