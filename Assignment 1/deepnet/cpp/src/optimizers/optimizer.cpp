#include "optimizers/optimizer.hpp"
#include <cmath>
#ifdef USE_CUDA
#include "cuda/cuda_ops.hpp"
#endif

namespace deepnet {

// Base Optimizer
void Optimizer::zero_grad() {
  for (auto &param : parameters) {
    if (param->requires_grad) {
      param->zero_grad();
    }
  }
}

void Optimizer::add_parameters(const std::vector<TensorPtr> &params) {
  parameters.insert(parameters.end(), params.begin(), params.end());
}

// SGD Implementation
SGD::SGD(const std::vector<TensorPtr> &params, float lr, float momentum,
         float weight_decay, bool nesterov)
    : lr(lr), momentum(momentum), weight_decay(weight_decay),
      nesterov(nesterov) {
  parameters = params;

  if (momentum > 0.0f) {
    for (const auto &param : parameters) {
      velocity.push_back(Tensor::zeros(param->shape, false, param->is_cuda));
    }
  }
}

void SGD::step() {
  for (size_t i = 0; i < parameters.size(); ++i) {
    auto &param = parameters[i];
    if (!param->requires_grad)
      continue;

#ifdef USE_CUDA
    if (param->is_cuda) {
        // Ensure gradients are allocated and pointers are valid
        float* d_p = param->data_ptr();
        float* d_g = param->grad_ptr();
        float* d_v = (momentum > 0.0f) ? velocity[i]->data_ptr() : nullptr;
        
        cuda::sgd_update_cuda_device(d_p, d_v, d_g, lr, momentum, weight_decay,
                                   nesterov, param->numel());
        continue;
    }
#endif

    // Add weight decay
    if (weight_decay > 0.0f) {
      for (size_t j = 0; j < param->grad.size(); ++j) {
        param->grad[j] += weight_decay * param->data[j];
      }
    }

    if (momentum > 0.0f) {
      // Update velocity: v = momentum * v + grad
      for (size_t j = 0; j < param->data.size(); ++j) {
        velocity[i]->data[j] = momentum * velocity[i]->data[j] + param->grad[j];
      }

      if (nesterov) {
        // Nesterov: param -= lr * (momentum * v + grad)
        for (size_t j = 0; j < param->data.size(); ++j) {
          param->data[j] -=
              lr * (momentum * velocity[i]->data[j] + param->grad[j]);
        }
      } else {
        // Standard momentum: param -= lr * v
        for (size_t j = 0; j < param->data.size(); ++j) {
          param->data[j] -= lr * velocity[i]->data[j];
        }
      }
    } else {
      // Standard SGD: param -= lr * grad
      for (size_t j = 0; j < param->data.size(); ++j) {
        param->data[j] -= lr * param->grad[j];
      }
    }
  }
}

// Adam Implementation
Adam::Adam(const std::vector<TensorPtr> &params, float lr, float beta1,
           float beta2, float eps, float weight_decay)
    : lr(lr), beta1(beta1), beta2(beta2), eps(eps), weight_decay(weight_decay),
      t(0) {
  parameters = params;

  for (const auto &param : parameters) {
    m.push_back(Tensor::zeros(param->shape, false, param->is_cuda));
    v.push_back(Tensor::zeros(param->shape, false, param->is_cuda));
  }
}

void Adam::step() {
  t++;

  for (size_t i = 0; i < parameters.size(); ++i) {
    auto &param = parameters[i];
    if (!param->requires_grad)
      continue;

#ifdef USE_CUDA
    if (param->is_cuda) {
        float* d_p = param->data_ptr();
        float* d_g = param->grad_ptr();
        float* d_m = m[i]->data_ptr();
        float* d_v = v[i]->data_ptr();
        
        cuda::adam_update_cuda_device(d_p, d_m, d_v, d_g, lr, beta1, beta2, eps,
                                    weight_decay, t, param->numel());
        continue;
    }
#endif

    // Add weight decay
    if (weight_decay > 0.0f) {
      for (size_t j = 0; j < param->grad.size(); ++j) {
        param->grad[j] += weight_decay * param->data[j];
      }
    }

    // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grad
    for (size_t j = 0; j < param->data.size(); ++j) {
      m[i]->data[j] = beta1 * m[i]->data[j] + (1.0f - beta1) * param->grad[j];
    }

    // Update biased second moment estimate: v = beta2 * v + (1 - beta2) *
    // grad^2
    for (size_t j = 0; j < param->data.size(); ++j) {
      v[i]->data[j] = beta2 * v[i]->data[j] +
                      (1.0f - beta2) * param->grad[j] * param->grad[j];
    }

    // Compute bias correction
    float m_hat_scale = 1.0f / (1.0f - static_cast<float>(std::pow(beta1, t)));
    float v_hat_scale = 1.0f / (1.0f - static_cast<float>(std::pow(beta2, t)));

    // Update parameters: param -= lr * m_hat / (sqrt(v_hat) + eps)
    for (size_t j = 0; j < param->data.size(); ++j) {
      float m_hat = m[i]->data[j] * m_hat_scale;
      float v_hat = v[i]->data[j] * v_hat_scale;
      param->data[j] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
  }
}

// SGD Implementation Extensions
std::map<std::string, std::vector<std::vector<float>>> SGD::state_dict() {
    std::map<std::string, std::vector<std::vector<float>>> state;
    if (momentum > 0.0f) {
        std::vector<std::vector<float>> v_data;
        for (auto &v : velocity) {
            if (v->is_cuda) v->sync_to_cpu();
            v_data.push_back(v->data);
        }
        state["velocity"] = v_data;
    }
    return state;
}

void SGD::load_state_dict(const std::map<std::string, std::vector<std::vector<float>>> &state) {
    if (state.count("velocity") && momentum > 0.0f) {
        auto &v_data = state.at("velocity");
        for (size_t i = 0; i < std::min(v_data.size(), velocity.size()); ++i) {
            velocity[i]->data = v_data[i];
#ifdef USE_CUDA
            velocity[i]->cpu_dirty = true;
#endif
            if (velocity[i]->is_cuda) velocity[i]->sync_to_cuda();
        }
    }
}

// Adam Implementation Extensions
std::map<std::string, std::vector<std::vector<float>>> Adam::state_dict() {
    std::map<std::string, std::vector<std::vector<float>>> state;
    std::vector<std::vector<float>> m_data, v_data;
    for (auto &mt : m) {
        if (mt->is_cuda) mt->sync_to_cpu();
        m_data.push_back(mt->data);
    }
    for (auto &vt : v) {
        if (vt->is_cuda) vt->sync_to_cpu();
        v_data.push_back(vt->data);
    }
    state["m"] = m_data;
    state["v"] = v_data;
    
    // Encode t as a vector of size 1
    state["t"] = {{static_cast<float>(t)}};
    return state;
}

void Adam::load_state_dict(const std::map<std::string, std::vector<std::vector<float>>> &state) {
    if (state.count("m")) {
        auto &m_data = state.at("m");
        for (size_t i = 0; i < std::min(m_data.size(), m.size()); ++i) {
            m[i]->data = m_data[i];
#ifdef USE_CUDA
            m[i]->cpu_dirty = true;
#endif
            if (m[i]->is_cuda) m[i]->sync_to_cuda();
        }
    }
    if (state.count("v")) {
        auto &v_data = state.at("v");
        for (size_t i = 0; i < std::min(v_data.size(), v.size()); ++i) {
            v[i]->data = v_data[i];
#ifdef USE_CUDA
            v[i]->cpu_dirty = true;
#endif
            if (v[i]->is_cuda) v[i]->sync_to_cuda();
        }
    }
    if (state.count("t") && !state.at("t").empty() && !state.at("t")[0].empty()) {
        t = static_cast<int>(state.at("t")[0][0]);
    }
}

} // namespace deepnet
