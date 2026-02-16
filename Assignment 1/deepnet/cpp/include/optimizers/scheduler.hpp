#pragma once

#include <algorithm>
#include <cmath>

// Define M_PI if not already defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace deepnet {

// Base Learning Rate Scheduler class
class LRScheduler {
public:
  virtual ~LRScheduler() = default;
  virtual float get_lr(int epoch) = 0;
  virtual float step() = 0;

protected:
  float base_lr;
  int current_epoch;
};

// Step Learning Rate Scheduler
class StepLR : public LRScheduler {
public:
  StepLR(float lr, int step_size, float gamma = 0.1f)
      : step_size(step_size), gamma(gamma) {
    base_lr = lr;
    current_epoch = 0;
  }

  float get_lr(int epoch) override {
    int num_drops = epoch / step_size;
    return base_lr * static_cast<float>(std::pow(gamma, num_drops));
  }

  float step() override {
    current_epoch++;
    return get_lr(current_epoch);
  }

private:
  int step_size;
  float gamma;
};

// Exponential Learning Rate Scheduler
class ExponentialLR : public LRScheduler {
public:
  ExponentialLR(float lr, float gamma = 0.95f) : gamma(gamma) {
    base_lr = lr;
    current_epoch = 0;
  }

  float get_lr(int epoch) override { return base_lr * static_cast<float>(std::pow(gamma, epoch)); }

  float step() override {
    current_epoch++;
    return get_lr(current_epoch);
  }

private:
  float gamma;
};

// Cosine Annealing Learning Rate Scheduler
class CosineAnnealingLR : public LRScheduler {
public:
  CosineAnnealingLR(float lr, int T_max, float eta_min = 0.0f)
      : T_max(T_max), eta_min(eta_min) {
    base_lr = lr;
    current_epoch = 0;
  }

  float get_lr(int epoch) override {
    float cosine = static_cast<float>(std::cos(M_PI * epoch / T_max));
    return eta_min + (base_lr - eta_min) * (1.0f + cosine) / 2.0f;
  }

  float step() override {
    current_epoch++;
    return get_lr(current_epoch);
  }

private:
  int T_max;
  float eta_min;
};

// Reduce on Plateau Scheduler
class ReduceLROnPlateau : public LRScheduler {
public:
  ReduceLROnPlateau(float lr, float factor = 0.1f, int patience = 10,
                    float threshold = 1e-4f, float min_lr = 0.0f)
      : factor(factor), patience(patience), threshold(threshold),
        min_lr(min_lr), best_loss(1e10f), num_bad_epochs(0) {
    base_lr = lr;
    current_epoch = 0;
    current_lr = lr;
  }

  float get_lr(int epoch) override { return current_lr; }

  float step(float loss) {
    current_epoch++;

    if (loss < best_loss - threshold) {
      best_loss = loss;
      num_bad_epochs = 0;
    } else {
      num_bad_epochs++;
    }

    if (num_bad_epochs >= patience) {
      current_lr = std::max(current_lr * factor, min_lr);
      num_bad_epochs = 0;
    }

    return current_lr;
  }

  float step() override { return current_lr; }

private:
  float factor;
  int patience;
  float threshold;
  float min_lr;
  float best_loss;
  int num_bad_epochs;
  float current_lr;
};

} // namespace deepnet
