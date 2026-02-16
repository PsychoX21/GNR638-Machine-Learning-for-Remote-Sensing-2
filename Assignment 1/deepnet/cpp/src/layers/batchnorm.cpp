#include "layers/batchnorm.hpp"
#include <cmath>
#include <random>
#include <stdexcept>
#ifdef USE_CUDA
#include "cuda/cuda_ops.hpp"
#endif


namespace deepnet {

// Dropout implementation removed (moved to layer.cpp)

// BatchNorm2D Implementation
BatchNorm2D::BatchNorm2D(int num_features, float eps, float momentum)
    : num_features(num_features), eps(eps), momentum(momentum) {

  gamma = Tensor::ones({num_features}, true, false);
  beta = Tensor::zeros({num_features}, true, false);
  running_mean = Tensor::zeros({num_features}, false, false);
  running_var = Tensor::ones({num_features}, false, false);
}

TensorPtr BatchNorm2D::forward(const TensorPtr &input) {
  if (!input) throw std::runtime_error("BatchNorm2D::forward: input is null");
  // Input shape: [batch, channels, height, width]
  if (input->shape.size() != 4) {
    throw std::runtime_error("BatchNorm2D expects 4D input");
  }

  int batch = input->shape[0];
  int channels = input->shape[1];
  int height = input->shape[2];
  int width = input->shape[3];

  if (channels != num_features) {
    throw std::runtime_error("BatchNorm2D channel mismatch");
  }

  // Cache input for backward
  last_input = input;
  normalized = Tensor::zeros(input->shape, false, input->is_cuda);
  batch_std_inv.resize(channels);

  auto output =
      Tensor::zeros(input->shape, true, input->is_cuda);

#ifdef USE_CUDA
  if (input->is_cuda) {
    if (!gamma->is_cuda) gamma->cuda();
    if (!beta->is_cuda) beta->cuda();
    if (!running_mean->is_cuda) running_mean->cuda();
    if (!running_var->is_cuda) running_var->cuda();

    if (training) {
        // Init saved stats
        if (!saved_mean || saved_mean->shape[0] != channels) {
            saved_mean = Tensor::zeros({channels}, false, true);
            saved_var = Tensor::zeros({channels}, false, true);
        }
        
        cuda::batchnorm_training_forward_cuda_device(
            input->data_ptr(), saved_mean->data_ptr(), saved_var->data_ptr(),
            running_mean->data_ptr(), running_var->data_ptr(),
            gamma->data_ptr(), beta->data_ptr(), output->data_ptr(),
            batch, channels, height, width, eps, momentum);
    } else {
        cuda::batchnorm_forward_cuda_device(
            input->data_ptr(), running_mean->data_ptr(),
            running_var->data_ptr(), gamma->data_ptr(),
            beta->data_ptr(), output->data_ptr(), batch,
            channels, height, width, eps);
    }
    return output;
  }
#endif

  if (training) {
    // Compute mean and variance per channel
    std::vector<float> mean(channels, 0.0f);
    std::vector<float> var(channels, 0.0f);
    int spatial_size = height * width;
    int n = batch * spatial_size;

    // Compute mean
    #pragma omp parallel for
    for (int c = 0; c < channels; ++c) {
      float sum = 0.0f;
      for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            int idx = ((b * channels + c) * height + h) * width + w;
            sum += input->data[idx];
          }
        }
      }
      mean[c] = sum / n;
    }

    // Compute variance
    #pragma omp parallel for
    for (int c = 0; c < channels; ++c) {
      float sum_sq = 0.0f;
      for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            int idx = ((b * channels + c) * height + h) * width + w;
            float diff = input->data[idx] - mean[c];
            sum_sq += diff * diff;
          }
        }
      }
      var[c] = sum_sq / n;
    }

    // Update running statistics
    for (int c = 0; c < channels; ++c) {
      running_mean->data[c] =
          (1.0f - momentum) * running_mean->data[c] + momentum * mean[c];
      running_var->data[c] =
          (1.0f - momentum) * running_var->data[c] + momentum * var[c];
    }

    // Normalize and cache
    #pragma omp parallel for
    for (int c = 0; c < channels; ++c) {
      float std_inv = 1.0f / std::sqrt(var[c] + eps);
      batch_std_inv[c] = std_inv;
      for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            int idx = ((b * channels + c) * height + h) * width + w;
            float x_hat = (input->data[idx] - mean[c]) * std_inv;
            normalized->data[idx] = x_hat;
            output->data[idx] = gamma->data[c] * x_hat + beta->data[c];
          }
        }
      }
    }
  } else {
    // Use running statistics for inference
    for (int c = 0; c < channels; ++c) {
      float std_inv = 1.0f / std::sqrt(running_var->data[c] + eps);
      batch_std_inv[c] = std_inv;
      for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            int idx = ((b * channels + c) * height + h) * width + w;
            float x_hat = (input->data[idx] - running_mean->data[c]) * std_inv;
            normalized->data[idx] = x_hat;
            output->data[idx] = gamma->data[c] * x_hat + beta->data[c];
          }
        }
      }
    }
  }

  return output;
}

TensorPtr BatchNorm2D::backward(const TensorPtr &grad_output) {
  int batch = last_input->shape[0];
  int channels = last_input->shape[1];
  int height = last_input->shape[2];
  int width = last_input->shape[3];
  int spatial = height * width;
  int n = batch * spatial;

  // Ensure grad buffers
  if (gamma->grad.size() != gamma->data.size())
    gamma->grad.resize(gamma->data.size(), 0.0f);
  if (beta->grad.size() != beta->data.size())
    beta->grad.resize(beta->data.size(), 0.0f);

  auto grad_input = Tensor::zeros(last_input->shape, false, last_input->is_cuda);

#ifdef USE_CUDA
  if (grad_output->is_cuda) {
      if (!saved_mean || !saved_var) {
          throw std::runtime_error("BatchNorm2D CUDA backward: saved stats not found (forward not called?)");
      }
      
      // Ensure grad buffers
      gamma->grad_ptr();
      beta->grad_ptr();
      
      cuda::batchnorm_backward_cuda_device(
          grad_output->data_ptr(), last_input->data_ptr(),
          saved_mean->data_ptr(), saved_var->data_ptr(),
          gamma->data_ptr(), grad_input->data_ptr(),
          gamma->grad_ptr(), beta->grad_ptr(),
          batch, channels, height, width, eps);
          
      return grad_input;
  }
#endif

  #pragma omp parallel for
  for (int c = 0; c < channels; ++c) {
    float g = gamma->data[c];
    float si = batch_std_inv[c];

    // Accumulate grad_gamma and grad_beta
    float dg = 0.0f, db = 0.0f;
    float sum_dy = 0.0f, sum_dy_xhat = 0.0f;
    for (int b = 0; b < batch; ++b) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          int idx = ((b * channels + c) * height + h) * width + w;
          float dy = grad_output->data[idx];
          float xh = normalized->data[idx];
          dg += dy * xh;
          db += dy;
          sum_dy += dy;
          sum_dy_xhat += dy * xh;
        }
      }
    }
    gamma->grad[c] += dg;
    beta->grad[c] += db;

    // Compute grad_input
    for (int b = 0; b < batch; ++b) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          int idx = ((b * channels + c) * height + h) * width + w;
          float dy = grad_output->data[idx];
          float xh = normalized->data[idx];
          grad_input->data[idx] = g * si / n *
              (n * dy - sum_dy - xh * sum_dy_xhat);
        }
      }
    }
  }

  return grad_input;
}

std::vector<TensorPtr> BatchNorm2D::parameters() {
  // Return running stats so they can be moved to CUDA recursively
  // They are not trainable (requires_grad=False), so optimizer will skip them if checked properly
  return {gamma, beta, running_mean, running_var};
}

// BatchNorm1D Implementation
BatchNorm1D::BatchNorm1D(int num_features, float eps, float momentum)
    : num_features(num_features), eps(eps), momentum(momentum) {

  gamma = Tensor::ones({num_features}, true, false);
  beta = Tensor::zeros({num_features}, true, false);
  running_mean = Tensor::zeros({num_features}, false, false);
  running_var = Tensor::ones({num_features}, false, false);
}

TensorPtr BatchNorm1D::forward(const TensorPtr &input) {
  if (!input) throw std::runtime_error("BatchNorm1D::forward: input is null");
  // Input shape: [batch, features]
  if (input->shape.size() != 2) {
    throw std::runtime_error("BatchNorm1D expects 2D input");
  }

  int batch = input->shape[0];
  int features = input->shape[1];

  if (features != num_features) {
    throw std::runtime_error("BatchNorm1D feature mismatch");
  }

  // Cache for backward
  last_input = input;
  normalized = Tensor::zeros(input->shape, false, input->is_cuda);
  batch_std_inv.resize(features);

  auto output =
      Tensor::zeros(input->shape, true, input->is_cuda);

#ifdef USE_CUDA
  if (input->is_cuda) {
      if (!gamma->is_cuda) gamma->cuda();
      if (!beta->is_cuda) beta->cuda();
      if (!running_mean->is_cuda) running_mean->cuda();
      if (!running_var->is_cuda) running_var->cuda();

      if (training) {
        if (!saved_mean || saved_mean->shape[0] != features) {
            saved_mean = Tensor::zeros({features}, false, true);
            saved_var = Tensor::zeros({features}, false, true);
        }
        cuda::batchnorm_training_forward_cuda_device(
            input->data_ptr(), saved_mean->data_ptr(), saved_var->data_ptr(),
            running_mean->data_ptr(), running_var->data_ptr(),
            gamma->data_ptr(), beta->data_ptr(), output->data_ptr(),
            batch, features, 1, 1, eps, momentum);
      } else {
        cuda::batchnorm_forward_cuda_device(
            input->data_ptr(), running_mean->data_ptr(),
            running_var->data_ptr(), gamma->data_ptr(),
            beta->data_ptr(), output->data_ptr(), batch,
            features, 1, 1, eps);
      }
      return output;
  }
#endif

  if (training) {
    std::vector<float> mean(features, 0.0f);
    std::vector<float> var(features, 0.0f);

    // Compute mean
    #pragma omp parallel for
    for (int f = 0; f < features; ++f) {
      float sum = 0.0f;
      for (int b = 0; b < batch; ++b) {
        sum += input->data[b * features + f];
      }
      mean[f] = sum / batch;
    }

    // Compute variance
    #pragma omp parallel for
    for (int f = 0; f < features; ++f) {
      float sum_sq = 0.0f;
      for (int b = 0; b < batch; ++b) {
        float diff = input->data[b * features + f] - mean[f];
        sum_sq += diff * diff;
      }
      var[f] = sum_sq / batch;
    }

    // Update running statistics
    for (int f = 0; f < features; ++f) {
      running_mean->data[f] =
          (1.0f - momentum) * running_mean->data[f] + momentum * mean[f];
      running_var->data[f] =
          (1.0f - momentum) * running_var->data[f] + momentum * var[f];
    }

    // Normalize and cache
    #pragma omp parallel for
    for (int f = 0; f < features; ++f) {
      float std_inv = 1.0f / std::sqrt(var[f] + eps);
      batch_std_inv[f] = std_inv;
      for (int b = 0; b < batch; ++b) {
        int idx = b * features + f;
        float x_hat = (input->data[idx] - mean[f]) * std_inv;
        normalized->data[idx] = x_hat;
        output->data[idx] = gamma->data[f] * x_hat + beta->data[f];
      }
    }
  } else {
    // Use running statistics
    for (int f = 0; f < features; ++f) {
      float std_inv = 1.0f / std::sqrt(running_var->data[f] + eps);
      batch_std_inv[f] = std_inv;
      for (int b = 0; b < batch; ++b) {
        int idx = b * features + f;
        float x_hat = (input->data[idx] - running_mean->data[f]) * std_inv;
        normalized->data[idx] = x_hat;
        output->data[idx] = gamma->data[f] * x_hat + beta->data[f];
      }
    }
  }

  return output;
}

TensorPtr BatchNorm1D::backward(const TensorPtr &grad_output) {
  int batch = last_input->shape[0];
  int features = last_input->shape[1];

  // Ensure grad buffers
  if (gamma->grad.size() != gamma->data.size())
    gamma->grad.resize(gamma->data.size(), 0.0f);
  if (beta->grad.size() != beta->data.size())
    beta->grad.resize(beta->data.size(), 0.0f);

  auto grad_input = Tensor::zeros(last_input->shape, false, last_input->is_cuda);

#ifdef USE_CUDA
  if (grad_output->is_cuda) {
      if (!saved_mean || !saved_var) {
          throw std::runtime_error("BatchNorm1D CUDA backward: saved stats not found");
      }
      
      gamma->grad_ptr();
      beta->grad_ptr();
      
      cuda::batchnorm_backward_cuda_device(
          grad_output->data_ptr(), last_input->data_ptr(),
          saved_mean->data_ptr(), saved_var->data_ptr(),
          gamma->data_ptr(), grad_input->data_ptr(),
          gamma->grad_ptr(), beta->grad_ptr(),
          batch, features, 1, 1, eps);
          
      return grad_input;
  }
#endif

  #pragma omp parallel for
  for (int f = 0; f < features; ++f) {
    float g = gamma->data[f];
    float si = batch_std_inv[f];

    float dg = 0.0f, db = 0.0f;
    float sum_dy = 0.0f, sum_dy_xhat = 0.0f;
    for (int b = 0; b < batch; ++b) {
      int idx = b * features + f;
      float dy = grad_output->data[idx];
      float xh = normalized->data[idx];
      dg += dy * xh;
      db += dy;
      sum_dy += dy;
      sum_dy_xhat += dy * xh;
    }
    gamma->grad[f] += dg;
    beta->grad[f] += db;

    for (int b = 0; b < batch; ++b) {
      int idx = b * features + f;
      float dy = grad_output->data[idx];
      float xh = normalized->data[idx];
      grad_input->data[idx] = g * si / batch *
          (batch * dy - sum_dy - xh * sum_dy_xhat);
    }
  }

  return grad_input;
}

std::vector<TensorPtr> BatchNorm1D::parameters() {
  return {gamma, beta, running_mean, running_var};
}

} // namespace deepnet
