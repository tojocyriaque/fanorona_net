# Neural Network Implementation

This repository contains a Rust implementation of a feedforward neural network with backpropagation, designed for a classification task with two softmax output groups. The implementation uses the `rayon` crate for parallelized computations and includes utility functions for matrix and vector operations.

## Mathematical Documentation

### Notation
- **L**: Number of layers (`ln`), including input, hidden, and output layers.
- **ls**: Vector of layer sizes, where `ls[k]` is the number of neurons in layer `k`.
- **is**: Input size (number of features in the input vector).
- **W_k**: Weight matrix for layer `k`, where `W_k[i][j]` connects neuron `j` of layer `k-1` to neuron `i` of layer `k`.
- **b_k**: Bias vector for layer `k`.
- **lr**: Learning rate (`lr`), controlling the step size in parameter updates.
- **x**: Input vector of size `is`.
- **z_k**: Pre-activation values for layer `k`, computed as `z_k = W_k * a_{k-1} + b_k`.
- **a_k**: Activation values for layer `k`, where `a_k = σ(z_k)` for hidden layers (sigmoid activation) and `a_L = softmax(z_L)` for the output layer.
- **d* and a***: Target labels for the two output groups (e.g., digit and another attribute).
- **σ(u)**: Sigmoid function, defined as `σ(u) = 1 / (1 + e^(-u))`.
- **softmax(z)**: Softmax function for a vector `z`, defined as `softmax(z)_i = e^(z_i) / Σ_j e^(z_j)`.

### Initialization
The neural network is initialized with:
- A vector of layer sizes `ls` (including input and output layers).
- Input size `is`.
- Learning rate `lr`.

Weights `W_k` and biases `b_k` are initialized using utility functions (`init_matrixes` and `init_vectors`). Weights are typically initialized randomly, and biases are set to zero or small values.

#### Mathematical Representation
- **Weights**: For layer `k`, `W_k` is a matrix of size `ls[k] × ls[k-1]` (or `ls[k] × is` for the input layer).
- **Biases**: For layer `k`, `b_k` is a vector of size `ls[k]`.
- **Structure**: The network has `L = ls.len()` layers, with `ls[0]` neurons in the input layer, `ls[1]` to `ls[L-2]` for hidden layers, and `ls[L-1]` for the output layer.

### Feedforward Propagation
The feedforward process computes the output for an input vector `x`.

#### Steps
1. **Input Layer (k = 0)**:
   - Compute pre-activation: `z_0 = W_0 * x + b_0`.
   - Apply sigmoid activation: `a_0 = σ(z_0)`, where `σ(u) = 1 / (1 + e^(-u))` is applied element-wise.
2. **Hidden Layers (k = 1 to L-2)**:
   - For each layer `k`:
     - Compute pre-activation: `z_k = W_k * a_{k-1} + b_k`.
     - Apply sigmoid activation: `a_k = σ(z_k)`.
3. **Output Layer (k = L-1)**:
   - Compute pre-activation: `z_{L-1} = W_{L-1} * a_{L-2} + b_{L-1}`.
   - Split `z_{L-1}` into two parts: `z_{L-1}[0..9]` (first 9 elements) and `z_{L-1}[9..]` (remaining elements).
   - Apply softmax to each part:
     - `sf1 = softmax(z_{L-1}[0..9])`.
     - `sf2 = softmax(z_{L-1}[9..])`.
   - Concatenate results: `a_{L-1} = [sf1, sf2]`.

#### Output
The output `a_{L-1}` is a probability distribution over two groups of classes, with `sf1` and `sf2` representing normalized probabilities for each group.

### Backpropagation
Backpropagation computes gradients of the loss function with respect to `W_k` and `b_k` and updates them using gradient descent.

#### Loss Function
The loss is the negative log-likelihood of the predicted probabilities for the true labels `d*` and `a*`:
- `Loss = -log(P_d(d*) * P_a(a*)) = -log(a_{L-1}[d*]) - log(a_{L-1}[a*])`.

#### Steps
1. **Output Layer Gradient**:
   - Initialize gradient `dz_{L-1}`:
     - For index `i` in `a_{L-1}`:
       - If `i == d*`, set `dz_{L-1}[i] = a_{L-1}[i] - 1`.
       - If `i == a*`, set `dz_{L-1}[i] = a_{L-1}[i] - 1`.
       - Otherwise, `dz_{L-1}[i] = a_{L-1}[i]`.
   - This corresponds to the derivative of the loss with respect to `z_{L-1}` for softmax outputs.
2. **Hidden Layers Gradient**:
   - For each layer `k` from `L-2` down to `0`:
     - Compute weight gradients: `dw_k[i][j] = dz_k[i] * a_{k-1}[j]`, where `a_{-1} = x` for the input layer.
     - Compute bias gradients: `db_k[i] = dz_k[i]`.
     - If `k > 0`, compute activation gradient for the previous layer:
       - `da_{k-1}[i] = Σ_j (dz_k[j] * W_k[j][i])`.
       - `dz_{k-1}[i] = da_{k-1}[i] * a_{k-1}[i] * (1 - a_{k-1}[i])` (sigmoid derivative).
3. **Parameter Update**:
   - Update weights: `W_k[i][j] -= lr * dw_k[i][j]`.
   - Update biases: `b_k[i] -= lr * dz_k[i]`.

### Parallelization
The implementation uses `rayon` for parallel iteration (e.g., in sigmoid application), improving performance for large vectors.

### Prediction
The `predict` function outputs the most likely classes and their probabilities:
- Run feedforward to get `a_{L-1}`.
- For the first group (`sf[0..9]`):
  - Find `d* = argmax(sf[0..9])` and its probability `pd* = max(sf[0..9])`.
- For the second group (`sf[9..]`):
  - Find `a* = argmax(sf[9..])` and its probability `pa* = max(sf[9..])`.
- Return `((d*, pd*), (a*, pa*))`.

### Key Features
- **Sigmoid Activation**: Used for hidden layers to introduce non-linearity.
- **Softmax Output**: Normalizes outputs into probabilities for two class groups.
- **Backpropagation**: Computes gradients efficiently with parallelized operations.
- **Gradient Descent**: Updates parameters using the learning rate `lr`.

This implementation is tailored for a classification task with two output groups, using sigmoid and softmax activations for non-linear relationships and multi-class classification.
