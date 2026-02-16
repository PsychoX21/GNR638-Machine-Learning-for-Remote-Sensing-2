#pragma once

#include "../tensor.hpp"
#include <memory>
#include <vector>
#include <map>
#include <string>


namespace deepnet {

// Base Optimizer class
class Optimizer {
public:
  virtual ~Optimizer() = default;
  virtual void step() = 0;
  virtual void zero_grad();

  virtual void set_lr(float lr) = 0;
  virtual float get_lr() = 0;

  virtual std::map<std::string, std::vector<std::vector<float>>> state_dict() { return {}; }
  virtual void load_state_dict(const std::map<std::string, std::vector<std::vector<float>>> &state) {}

  void add_parameters(const std::vector<TensorPtr> &params);

protected:
  std::vector<TensorPtr> parameters;
};

// SGD Optimizer
class SGD : public Optimizer {
public:
  SGD(const std::vector<TensorPtr> &params, float lr = 0.01f,
      float momentum = 0.0f, float weight_decay = 0.0f, bool nesterov = false);

  void step() override;
  void set_lr(float lr) override { this->lr = lr; }
  float get_lr() override { return lr; }

  std::map<std::string, std::vector<std::vector<float>>> state_dict() override;
  void load_state_dict(const std::map<std::string, std::vector<std::vector<float>>> &state) override;

private:
  float lr, momentum, weight_decay;
  bool nesterov;
  std::vector<TensorPtr> velocity;
};

// Adam Optimizer
class Adam : public Optimizer {
public:
  Adam(const std::vector<TensorPtr> &params, float lr = 0.001f,
       float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
       float weight_decay = 0.0f);

  void step() override;
  void set_lr(float lr) override { this->lr = lr; }
  float get_lr() override { return lr; }

  std::map<std::string, std::vector<std::vector<float>>> state_dict() override;
  void load_state_dict(const std::map<std::string, std::vector<std::vector<float>>> &state) override;

private:
  float lr, beta1, beta2, eps, weight_decay;
  int t;                    // Timestep
  std::vector<TensorPtr> m; // First moment
  std::vector<TensorPtr> v; // Second moment
};

} // namespace deepnet
