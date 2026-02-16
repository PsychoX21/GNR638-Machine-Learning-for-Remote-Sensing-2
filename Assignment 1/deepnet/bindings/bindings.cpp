#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "layers/batchnorm.hpp"
#include "layers/layer.hpp"
#include "layers/pooling.hpp"
#include "loss.hpp"
#include "optimizers/optimizer.hpp"
#include "optimizers/scheduler.hpp"
#include "tensor.hpp"
#include "cuda/cuda_utils.hpp"

namespace py = pybind11;
using namespace deepnet;

PYBIND11_MODULE(deepnet_backend, m) {
  m.doc() = "DeepNet: Custom deep learning framework backend";

  // Tensor class
  py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
      .def(py::init<>())
      .def(py::init<const std::vector<int> &, bool, bool>(), py::arg("shape"),
           py::arg("requires_grad") = false, py::arg("cuda") = false)
      .def(py::init<const std::vector<float> &, const std::vector<int> &, bool, bool>(),
           py::arg("data"), py::arg("shape"), py::arg("requires_grad") = false, 
           py::arg("cuda") = false)
      .def_property("data",
        [](Tensor &t) -> std::vector<float>& {
            if (t.is_cuda) t.sync_to_cpu();
            return t.data;
        },
        [](Tensor &t, const std::vector<float> &v) {
            t.data = v;
#ifdef USE_CUDA
            t.cpu_dirty = true;
            t.cuda_dirty = false;
            if (t.is_cuda) t.sync_to_cuda();
#endif
        })
      .def_property("grad",
        [](Tensor &t) -> std::vector<float>& {
            if (t.is_cuda) t.sync_to_cpu();
            return t.grad;
        },
        [](Tensor &t, const std::vector<float> &v) {
            t.grad = v;
#ifdef USE_CUDA
            t.cpu_dirty = true;
            t.cuda_dirty = false;
            if (t.is_cuda) t.sync_to_cuda();
#endif
        })
      .def_readwrite("shape", &Tensor::shape)
      .def_readwrite("requires_grad", &Tensor::requires_grad)
      .def_readwrite("is_cuda", &Tensor::is_cuda)

      // Factory methods
      .def_static("zeros", &Tensor::zeros, py::arg("shape"),
                  py::arg("requires_grad") = false, py::arg("cuda") = false)
      .def_static("ones", &Tensor::ones, py::arg("shape"),
                  py::arg("requires_grad") = false, py::arg("cuda") = false)
      .def_static("randn", &Tensor::randn, py::arg("shape"),
                  py::arg("mean") = 0.0f, py::arg("std") = 1.0f,
                  py::arg("requires_grad") = false, py::arg("cuda") = false)
      .def_static("from_data", &Tensor::from_data, py::arg("data"),
                  py::arg("shape"), py::arg("requires_grad") = false,
                  py::arg("cuda") = false)

      // Shape operations
      .def("size", py::overload_cast<>(&Tensor::size, py::const_))
      .def("size", py::overload_cast<int>(&Tensor::size, py::const_))
      .def("numel", &Tensor::numel)
      .def("ndim", &Tensor::ndim)
      .def("reshape", &Tensor::reshape)
      .def("view", &Tensor::view)
      .def("transpose", &Tensor::transpose)
      .def("flatten", &Tensor::flatten, py::arg("start_dim") = 0,
           py::arg("end_dim") = -1)

      // Operations
      .def("add", &Tensor::add)
      .def("sub", &Tensor::sub)
      .def("mul", &Tensor::mul)
      .def("div", &Tensor::div)
      .def("add_scalar", &Tensor::add_scalar)
      .def("mul_scalar", &Tensor::mul_scalar)
      .def("matmul", &Tensor::matmul)
      .def("mm", &Tensor::mm)

      // Reductions
      .def("sum", &Tensor::sum, py::arg("dim") = -1, py::arg("keepdim") = false)
      .def("mean", &Tensor::mean, py::arg("dim") = -1,
           py::arg("keepdim") = false)

      // Activations
      .def("relu", &Tensor::relu)
      .def("leaky_relu", &Tensor::leaky_relu, py::arg("negative_slope") = 0.01f)
      .def("tanh", &Tensor::tanh_)
      .def("sigmoid", &Tensor::sigmoid)

      // Math
      .def("exp", &Tensor::exp)
      .def("log", &Tensor::log)
      .def("pow", &Tensor::pow, py::arg("exponent"))
      .def("sqrt", &Tensor::sqrt)
      .def("max", &Tensor::max, py::arg("dim") = -1, py::arg("keepdim") = false)
      .def("min", &Tensor::min, py::arg("dim") = -1, py::arg("keepdim") = false)
      .def("permute", &Tensor::permute)

      // Autograd
      .def("backward", &Tensor::backward, py::arg("gradient") = nullptr)
      .def("zero_grad", &Tensor::zero_grad)
      .def("detach", &Tensor::detach)

      // CUDA
      .def("cuda", [](std::shared_ptr<Tensor> t) { t->cuda(); return t; })
      .def("cpu", [](std::shared_ptr<Tensor> t) { t->cpu(); return t; })
      .def("to", &Tensor::to)

       // Utility
      .def("copy_", &Tensor::copy_)
      .def("fill_", &Tensor::fill_)
      .def("uniform_", &Tensor::uniform_)
      .def("normal_", &Tensor::normal_)
      .def("clone", &Tensor::clone)
      .def("shape_str", &Tensor::shape_str)
      .def("print", &Tensor::print, py::arg("name") = "")

      // Operators
      .def("__add__", &Tensor::operator+)
      .def("__sub__", &Tensor::operator-)
      .def("__mul__", &Tensor::operator*)
      .def("__truediv__", &Tensor::operator/);

  // Base Layer
  py::class_<Layer, std::shared_ptr<Layer>>(m, "Layer")
      .def("forward", &Layer::forward)
      .def("backward", &Layer::backward)
      .def("parameters", &Layer::parameters)
      .def("train", &Layer::train)
      .def("eval", &Layer::eval);

  // Conv2D
  py::class_<Conv2D, Layer, std::shared_ptr<Conv2D>>(m, "Conv2D")
      .def(py::init<int, int, int, int, int, bool>(), py::arg("in_channels"),
           py::arg("out_channels"), py::arg("kernel_size"),
           py::arg("stride") = 1, py::arg("padding") = 0,
           py::arg("bias") = true)
      .def("forward", &Conv2D::forward)
      .def("parameters", &Conv2D::parameters);

  // Linear
  py::class_<Linear, Layer, std::shared_ptr<Linear>>(m, "Linear")
      .def(py::init<int, int, bool>(), py::arg("in_features"),
           py::arg("out_features"), py::arg("bias") = true)
      .def("forward", &Linear::forward)
      .def("parameters", &Linear::parameters);

  // Activations
  py::class_<ReLU, Layer, std::shared_ptr<ReLU>>(m, "ReLU")
      .def(py::init<>())
      .def("forward", &ReLU::forward);

  py::class_<LeakyReLU, Layer, std::shared_ptr<LeakyReLU>>(m, "LeakyReLU")
      .def(py::init<float>(), py::arg("negative_slope") = 0.01f)
      .def("forward", &LeakyReLU::forward);

  py::class_<Tanh, Layer, std::shared_ptr<Tanh>>(m, "Tanh")
      .def(py::init<>())
      .def("forward", &Tanh::forward);

  py::class_<Sigmoid, Layer, std::shared_ptr<Sigmoid>>(m, "Sigmoid")
      .def(py::init<>())
      .def("forward", &Sigmoid::forward);

  // Pooling
  py::class_<MaxPool2D, Layer, std::shared_ptr<MaxPool2D>>(m, "MaxPool2D")
      .def(py::init<int, int>(), py::arg("kernel_size"), py::arg("stride") = -1)
      .def("forward", &MaxPool2D::forward);

  py::class_<AvgPool2D, Layer, std::shared_ptr<AvgPool2D>>(m, "AvgPool2D")
      .def(py::init<int, int>(), py::arg("kernel_size"), py::arg("stride") = -1)
      .def("forward", &AvgPool2D::forward);

  // BatchNorm
  py::class_<BatchNorm2D, Layer, std::shared_ptr<BatchNorm2D>>(m, "BatchNorm2D")
      .def(py::init<int, float, float>(), py::arg("num_features"),
           py::arg("eps") = 1e-5f, py::arg("momentum") = 0.1f)
      .def("forward", &BatchNorm2D::forward)
      .def("parameters", &BatchNorm2D::parameters);

  py::class_<BatchNorm1D, Layer, std::shared_ptr<BatchNorm1D>>(m, "BatchNorm1D")
      .def(py::init<int, float, float>(), py::arg("num_features"),
           py::arg("eps") = 1e-5f, py::arg("momentum") = 0.1f)
      .def("forward", &BatchNorm1D::forward)
      .def("parameters", &BatchNorm1D::parameters);

  // Dropout
  py::class_<Dropout, Layer, std::shared_ptr<Dropout>>(m, "Dropout")
      .def(py::init<float>(), py::arg("p") = 0.5f)
      .def("forward", &Dropout::forward);

  // Flatten
  py::class_<Flatten, Layer, std::shared_ptr<Flatten>>(m, "Flatten")
      .def(py::init<int, int>(), py::arg("start_dim") = 1,
           py::arg("end_dim") = -1)
      .def("forward", &Flatten::forward);

  // Optimizers
  py::class_<Optimizer, std::shared_ptr<Optimizer>>(m, "Optimizer")
      .def("step", &Optimizer::step)
      .def("zero_grad", &Optimizer::zero_grad)
      .def("add_parameters", &Optimizer::add_parameters)
      .def("set_lr", &Optimizer::set_lr)
      .def("get_lr", &Optimizer::get_lr)
      .def("state_dict", &Optimizer::state_dict)
      .def("load_state_dict", &Optimizer::load_state_dict);

  py::class_<SGD, Optimizer, std::shared_ptr<SGD>>(m, "SGD")
      .def(
          py::init<const std::vector<TensorPtr> &, float, float, float, bool>(),
          py::arg("params"), py::arg("lr") = 0.01f, py::arg("momentum") = 0.0f,
          py::arg("weight_decay") = 0.0f, py::arg("nesterov") = false)
      .def("step", &SGD::step)
      .def("zero_grad", &SGD::zero_grad)
      .def("set_lr", &SGD::set_lr)
      .def("get_lr", &SGD::get_lr)
      .def("state_dict", &SGD::state_dict)
      .def("load_state_dict", &SGD::load_state_dict);

  py::class_<Adam, Optimizer, std::shared_ptr<Adam>>(m, "Adam")
      .def(py::init<const std::vector<TensorPtr> &, float, float, float, float,
                    float>(),
           py::arg("params"), py::arg("lr") = 0.001f, py::arg("beta1") = 0.9f,
           py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f,
           py::arg("weight_decay") = 0.0f)
      .def("step", &Adam::step)
      .def("zero_grad", &Adam::zero_grad)
      .def("set_lr", &Adam::set_lr)
      .def("get_lr", &Adam::get_lr)
      .def("state_dict", &Adam::state_dict)
      .def("load_state_dict", &Adam::load_state_dict);

  // Loss functions
  py::class_<CrossEntropyLoss>(m, "CrossEntropyLoss")
      .def(py::init<>())
      .def("forward", &CrossEntropyLoss::forward)
      .def("get_input_grad", &CrossEntropyLoss::get_input_grad);

  py::class_<MSELoss>(m, "MSELoss")
      .def(py::init<>())
      .def("forward", &MSELoss::forward);

  // Learning Rate Schedulers
  py::class_<LRScheduler, std::shared_ptr<LRScheduler>>(m, "LRScheduler")
      .def("get_lr", &LRScheduler::get_lr)
      .def("step", &LRScheduler::step);

  py::class_<StepLR, LRScheduler, std::shared_ptr<StepLR>>(m, "StepLR")
      .def(py::init<float, int, float>(), py::arg("lr"), py::arg("step_size"),
           py::arg("gamma") = 0.1f)
      .def("get_lr", &StepLR::get_lr)
      .def("step", &StepLR::step);

  py::class_<ExponentialLR, LRScheduler, std::shared_ptr<ExponentialLR>>(
      m, "ExponentialLR")
      .def(py::init<float, float>(), py::arg("lr"), py::arg("gamma") = 0.95f)
      .def("get_lr", &ExponentialLR::get_lr)
      .def("step", &ExponentialLR::step);

  py::class_<CosineAnnealingLR, LRScheduler,
             std::shared_ptr<CosineAnnealingLR>>(m, "CosineAnnealingLR")
      .def(py::init<float, int, float>(), py::arg("lr"), py::arg("T_max"),
           py::arg("eta_min") = 0.0f)
      .def("get_lr", &CosineAnnealingLR::get_lr)
      .def("step", &CosineAnnealingLR::step);

  py::class_<ReduceLROnPlateau, LRScheduler,
             std::shared_ptr<ReduceLROnPlateau>>(m, "ReduceLROnPlateau")
      .def(py::init<float, float, int, float, float>(), py::arg("lr"),
           py::arg("factor") = 0.1f, py::arg("patience") = 10,
           py::arg("threshold") = 1e-4f, py::arg("min_lr") = 0.0f)
      .def("get_lr", &ReduceLROnPlateau::get_lr)
      .def("step", py::overload_cast<float>(&ReduceLROnPlateau::step));

  // CUDA utilities
  m.def("is_cuda_available", &cuda::is_cuda_available,
        "Check if CUDA GPU is available");

  // Seeding
  m.def("manual_seed", &deepnet::manual_seed, "Set seed for C++ backend randomness");
}
