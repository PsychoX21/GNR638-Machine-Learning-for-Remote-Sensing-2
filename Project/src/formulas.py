from typing import Optional
import logging

logger = logging.getLogger(__name__)

FORMULA_TABLE = {
    "conv output": (
        "REFERENCE FORMULAS:\n"
        "• Conv2d output size: floor((input + 2*padding - dilation*(kernel-1) - 1) / stride + 1)\n"
        "• For standard conv (dilation=1): floor((input + 2*padding - kernel) / stride + 1)\n"
        "• Transposed conv output: (input - 1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1"
    ),
    "convolution": (
        "REFERENCE FORMULAS:\n"
        "• Conv2d output size: floor((input + 2*padding - dilation*(kernel-1) - 1) / stride + 1)\n"
        "• Conv2d parameters: out_channels * (in_channels/groups * kernel_h * kernel_w + 1 if bias)"
    ),
    "output shape": (
        "REFERENCE FORMULAS:\n"
        "• Conv2d output: floor((input + 2*padding - kernel) / stride + 1)\n"
        "• MaxPool2d/AvgPool2d output: floor((input + 2*padding - kernel) / stride + 1)\n"
        "• Linear output: (batch, out_features)\n"
        "• After flatten: (batch, channels * height * width)"
    ),
    "shape propagation": (
        "REFERENCE FORMULAS:\n"
        "• Conv2d output: floor((input + 2*padding - kernel) / stride + 1)\n"
        "• MaxPool2d output: floor((input + 2*padding - kernel) / stride + 1)\n"
        "• Default stride for MaxPool2d = kernel_size\n"
        "• Default padding = 0, default dilation = 1"
    ),

    # Parameter counting
    "parameter count": (
        "REFERENCE FORMULAS:\n"
        "• Linear(in, out): in*out + out (with bias), in*out (without bias)\n"
        "• Conv2d(in_ch, out_ch, k): out_ch * (in_ch * k * k) + out_ch (with bias)\n"
        "• BatchNorm(features): 2*features (weight + bias, running mean/var are buffers, not params)\n"
        "• LayerNorm(features): 2*features\n"
        "• Embedding(vocab, dim): vocab * dim\n"
        "• LSTM(input, hidden): 4 * ((input+hidden)*hidden + hidden) per layer, *2 if bidirectional\n"
        "• GRU(input, hidden): 3 * ((input+hidden)*hidden + hidden) per layer"
    ),
    "parameters": (
        "REFERENCE FORMULAS:\n"
        "• Linear(in, out) params = in*out + out (with bias)\n"
        "• Conv2d(in_ch, out_ch, k) params = out_ch * (in_ch * k * k) + out_ch (with bias)\n"
        "• Total trainable: sum(p.numel() for p in model.parameters() if p.requires_grad)"
    ),
    "trainable": (
        "REFERENCE FORMULAS:\n"
        "• Trainable parameters: those with requires_grad=True\n"
        "• BatchNorm has 2*features trainable params (gamma, beta)\n"
        "• Frozen layers: requires_grad=False → 0 trainable params from those layers"
    ),

    # Attention mechanism
    "attention": (
        "REFERENCE FORMULAS:\n"
        "• Scaled Dot-Product Attention: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V\n"
        "• Multi-Head Attention: d_k = d_model / num_heads\n"
        "• Self-attention params: 3 * d_model^2 (for Q,K,V projections) + d_model^2 (output projection)\n"
        "• Total MHA params: 4 * d_model^2 + 4 * d_model (with biases)"
    ),
    "self-attention": (
        "REFERENCE FORMULAS:\n"
        "• Q = XW_Q, K = XW_K, V = XW_V\n"
        "• Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V\n"
        "• d_k = d_model / num_heads\n"
        "• Complexity: O(n^2 * d) where n = sequence length"
    ),
    "transformer": (
        "REFERENCE FORMULAS:\n"
        "• Multi-Head Attention: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V\n"
        "• FFN: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2 (typically d_ff = 4 * d_model)\n"
        "• Layer: LN(x + MHA(x)) then LN(x + FFN(x)) [Post-LN] or x + MHA(LN(x)) [Pre-LN]\n"
        "• Positional encoding: PE(pos,2i) = sin(pos/10000^(2i/d_model)), PE(pos,2i+1) = cos(...)"
    ),

    # Normalization
    "batchnorm": (
        "REFERENCE FORMULAS:\n"
        "• Training: normalize using batch statistics (mean, var of current batch)\n"
        "• Inference: normalize using running statistics (accumulated during training)\n"
        "• y = (x - running_mean) / sqrt(running_var + eps) * gamma + beta\n"
        "• BatchNorm1d/2d/3d: gamma (weight) and beta (bias) are learnable\n"
        "• running_mean and running_var are NOT parameters (they are buffers)"
    ),
    "batch norm": (
        "REFERENCE FORMULAS:\n"
        "• Training: uses batch mean/var. Inference: uses running mean/var.\n"
        "• model.eval() switches to running statistics\n"
        "• Affine params: gamma (scale) and beta (shift) — 2 * num_features total"
    ),
    "layernorm": (
        "REFERENCE FORMULAS:\n"
        "• LayerNorm normalizes across the feature dimension (not batch)\n"
        "• Used in Transformers. Invariant to batch size.\n"
        "• y = (x - mean) / sqrt(var + eps) * gamma + beta\n"
        "• 2 * normalized_shape parameters (gamma + beta)"
    ),
    "normalization": (
        "REFERENCE FORMULAS:\n"
        "• BatchNorm: normalizes over batch dim, uses running stats at inference\n"
        "• LayerNorm: normalizes over feature dim, same at train/inference\n"
        "• GroupNorm: normalizes over groups of channels\n"
        "• InstanceNorm: normalizes over spatial dims per channel per sample"
    ),

    # Dropout
    "dropout": (
        "REFERENCE FORMULAS:\n"
        "• Training: randomly zero elements with probability p, scale remaining by 1/(1-p)\n"
        "• Inference: dropout is NOT applied (model.eval() or torch.no_grad())\n"
        "• Inverted dropout (PyTorch default): multiply by 1/(1-p) during training\n"
        "• Standard dropout: multiply by (1-p) during inference"
    ),

    # Loss functions
    "cross-entropy": (
        "REFERENCE FORMULAS:\n"
        "• CrossEntropyLoss = -log(softmax(logits)[correct_class])\n"
        "• = -logits[correct_class] + log(sum(exp(logits)))\n"
        "• softmax(x_i) = exp(x_i) / sum(exp(x_j))\n"
        "• NLLLoss expects log_softmax input: NLLLoss(log_softmax(x), target)"
    ),
    "loss function": (
        "REFERENCE FORMULAS:\n"
        "• MSE: mean((y_pred - y_true)^2)\n"
        "• BCE: -[y*log(p) + (1-y)*log(1-p)]\n"
        "• CrossEntropy: -sum(y_true * log(softmax(logits)))\n"
        "• Hinge: max(0, 1 - y*f(x))\n"
        "• KL Divergence: sum(p * log(p/q))"
    ),
    "softmax": (
        "REFERENCE FORMULAS:\n"
        "• softmax(x_i) = exp(x_i) / sum_j(exp(x_j))\n"
        "• Properties: outputs sum to 1, all values in (0,1)\n"
        "• Temperature: softmax(x_i/T) — higher T = softer distribution\n"
        "• LogSoftmax: log(softmax(x)) = x - log(sum(exp(x))) [numerically stable]"
    ),

    # Recurrent networks
    "lstm": (
        "REFERENCE FORMULAS:\n"
        "• LSTM gates: f_t = σ(W_f·[h_{t-1}, x_t] + b_f) [forget]\n"
        "• i_t = σ(W_i·[h_{t-1}, x_t] + b_i) [input]\n"
        "• o_t = σ(W_o·[h_{t-1}, x_t] + b_o) [output]\n"
        "• c̃_t = tanh(W_c·[h_{t-1}, x_t] + b_c) [candidate]\n"
        "• c_t = f_t * c_{t-1} + i_t * c̃_t\n"
        "• h_t = o_t * tanh(c_t)\n"
        "• Parameters per layer: 4 * ((input_size + hidden_size) * hidden_size + hidden_size)"
    ),
    "gru": (
        "REFERENCE FORMULAS:\n"
        "• GRU: z_t = σ(W_z·[h_{t-1}, x_t]) [update gate]\n"
        "• r_t = σ(W_r·[h_{t-1}, x_t]) [reset gate]\n"
        "• h̃_t = tanh(W·[r_t * h_{t-1}, x_t]) [candidate]\n"
        "• h_t = (1-z_t) * h_{t-1} + z_t * h̃_t\n"
        "• Parameters per layer: 3 * ((input_size + hidden_size) * hidden_size + hidden_size)"
    ),
    "rnn": (
        "REFERENCE FORMULAS:\n"
        "• Vanilla RNN: h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)\n"
        "• LSTM: 4 gates, cell state. Parameters: 4*(input+hidden)*hidden + 4*hidden\n"
        "• GRU: 3 gates, no cell state. Parameters: 3*(input+hidden)*hidden + 3*hidden\n"
        "• Bidirectional: doubles parameters and output dimension"
    ),

    # Pooling
    "pooling": (
        "REFERENCE FORMULAS:\n"
        "• MaxPool2d/AvgPool2d output: floor((input + 2*padding - kernel) / stride + 1)\n"
        "• Default stride = kernel_size\n"
        "• Global Average Pooling: reduces spatial dims to 1x1\n"
        "• Adaptive pooling: specify output size, not kernel size"
    ),
    "max pool": (
        "REFERENCE FORMULAS:\n"
        "• MaxPool2d output: floor((input + 2*padding - kernel) / stride + 1)\n"
        "• Default: stride = kernel_size, padding = 0\n"
        "• No learnable parameters"
    ),

    # Gradient / Backpropagation
    "gradient": (
        "REFERENCE FORMULAS:\n"
        "• Chain rule: dL/dx = dL/dy * dy/dx\n"
        "• ReLU gradient: 1 if x > 0, 0 if x <= 0\n"
        "• Sigmoid gradient: σ(x) * (1 - σ(x))\n"
        "• Tanh gradient: 1 - tanh²(x)\n"
        "• Softmax Jacobian: diag(s) - s*s^T"
    ),
    "backpropagation": (
        "REFERENCE FORMULAS:\n"
        "• Backprop applies chain rule layer by layer\n"
        "• dW = dL/dW = (dL/dy) * (dy/dW) = upstream_grad * local_grad\n"
        "• Vanishing gradient: occurs when gradients < 1 multiply through many layers\n"
        "• Exploding gradient: occurs when gradients > 1 multiply through many layers"
    ),

    # Optimization
    "optimizer": (
        "REFERENCE FORMULAS:\n"
        "• SGD: w = w - lr * grad\n"
        "• SGD + Momentum: v = μ*v - lr*grad; w = w + v\n"
        "• Adam: m = β1*m + (1-β1)*grad; v = β2*v + (1-β2)*grad²\n"
        "  w = w - lr * m_hat / (sqrt(v_hat) + eps)\n"
        "  where m_hat = m/(1-β1^t), v_hat = v/(1-β2^t)\n"
        "• Default Adam: β1=0.9, β2=0.999, eps=1e-8"
    ),
    "learning rate": (
        "REFERENCE FORMULAS:\n"
        "• Step decay: lr = lr_0 * factor^(epoch // step_size)\n"
        "• Cosine annealing: lr = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(π*t/T))\n"
        "• Warmup: linearly increase lr from 0 to lr_max over warmup steps"
    ),

    # Architectures
    "resnet": (
        "REFERENCE FORMULAS:\n"
        "• Skip connection: y = F(x) + x (identity shortcut)\n"
        "• If dimensions mismatch: y = F(x) + W_s*x (projection shortcut)\n"
        "• Bottleneck block: 1x1 conv → 3x3 conv → 1x1 conv + skip\n"
        "• ResNet solves vanishing gradient by allowing gradient flow through shortcuts"
    ),
    "skip connection": (
        "REFERENCE FORMULAS:\n"
        "• y = F(x) + x — gradient flows through addition unattenuated\n"
        "• Enables training very deep networks (100+ layers)\n"
        "• Used in ResNet, DenseNet (concatenation), U-Net"
    ),

    # Regularization
    "regularization": (
        "REFERENCE FORMULAS:\n"
        "• L1: loss += λ * sum(|w|)\n"
        "• L2 (weight decay): loss += λ * sum(w²)\n"
        "• Dropout: randomly zero with prob p during training\n"
        "• Data augmentation, early stopping, batch normalization"
    ),

    # Embedding / NLP
    "embedding": (
        "REFERENCE FORMULAS:\n"
        "• Embedding(num_embeddings, embedding_dim): lookup table\n"
        "• Parameters: num_embeddings * embedding_dim\n"
        "• Output shape: (..., embedding_dim) — adds last dimension"
    ),
    "word2vec": (
        "REFERENCE FORMULAS:\n"
        "• CBOW: predict center word from context\n"
        "• Skip-gram: predict context from center word\n"
        "• Negative sampling: approximate softmax with binary classification"
    ),

    # GAN
    "gan": (
        "REFERENCE FORMULAS:\n"
        "• min_G max_D [E[log(D(x))] + E[log(1 - D(G(z)))]]\n"
        "• Generator: minimize log(1-D(G(z))) or maximize log(D(G(z)))\n"
        "• Discriminator: maximize log(D(x)) + log(1-D(G(z)))\n"
        "• Mode collapse: generator produces limited variety"
    ),

    # VAE
    "vae": (
        "REFERENCE FORMULAS:\n"
        "• ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))\n"
        "• KL(N(μ,σ²) || N(0,1)) = -0.5 * sum(1 + log(σ²) - μ² - σ²)\n"
        "• Reparameterization: z = μ + σ * ε, where ε ~ N(0,1)"
    ),

    # Activation functions
    "activation": (
        "REFERENCE FORMULAS:\n"
        "• ReLU: max(0, x). Gradient: 1 if x > 0, 0 otherwise\n"
        "• Sigmoid: σ(x) = 1/(1+e^{-x}). Range: (0,1)\n"
        "• Tanh: (e^x - e^{-x})/(e^x + e^{-x}). Range: (-1,1)\n"
        "• LeakyReLU: max(αx, x) where α=0.01 typically\n"
        "• GELU: x * Φ(x) where Φ is standard normal CDF\n"
        "• Swish/SiLU: x * σ(x)"
    ),

    # MLP
    "mlp": (
        "REFERENCE FORMULAS:\n"
        "• MLP: sequence of Linear → Activation → Linear → ...\n"
        "• h = activation(W*x + b)\n"
        "• Universal approximation theorem: 1 hidden layer MLP can approximate any function\n"
        "• PyTorch: nn.Sequential(nn.Linear(in, h1), nn.ReLU(), nn.Linear(h1, out))"
    ),
}


def get_formula_injection(key_concept: str) -> Optional[str]:
    if not key_concept:
        return None

    key_lower = key_concept.lower().strip()

    matched = []
    for keyword, formula in FORMULA_TABLE.items():
        if keyword in key_lower or key_lower in keyword:
            matched.append(formula)

    if matched:
        return "\n\n".join(dict.fromkeys(matched))

    words = key_lower.split()
    for word in words:
        if len(word) < 3:
            continue
        for keyword, formula in FORMULA_TABLE.items():
            if word in keyword:
                return formula

    return None
