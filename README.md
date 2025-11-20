# Neural Network For Classification from Scratch 🧠✨

This repository contains a **Neural Network implemented from scratch** in Python using **NumPy**.
It supports **binary classification**, **multiclass classification**, and **multilabel classification**.

## Features 🚀

* ⚡ Fully vectorized **feedforward** and **backpropagation**.
* 🎛️ Supports **ReLU, Sigmoid, Tanh, Identity, Softmax, Polynomial** activations.
* 🏋️ **Mini-batch training** with adjustable learning rate and epochs.
* 🏗️ Flexible network architecture: define neurons in hidden layers and output layer.
* 🐍 Easy-to-use Python class: `nueral_network`.
* 🧠 Can be used as **Binary Classifier**, **Multiclass Classifier**, or **Multilabel Classifier**.


## How It Works ⚙️

### 1️⃣ Forward Pass (Feedforward)

* Each layer computes a linear combination of inputs:

  ```
  Z = X * W + b
  ```
* Apply the **activation function**: ReLU, Sigmoid, Tanh, Softmax, Identity, or Polynomial.
* Output of one layer becomes the input to the next layer.

### 2️⃣ Backward Pass (Backpropagation)

* Compute **output error**:

  ```python
  dZ = output - target
  ```
* Propagate the error backwards through hidden layers using the **derivative of the activation function**.
* Update weights and biases using **gradient descent**:

  ```
  W := W - lr * (dZ.T * X) / batch_size
  b := b - lr * mean(dZ)
  ```

### 3️⃣ Training

* Supports **mini-batch training** for efficiency.
* Each epoch:

  1. Shuffle the dataset
  2. Split into batches
  3. Forward pass
  4. Backpropagation
  5. Update weights
* Repeat for the specified number of **epochs**.

### 4️⃣ Prediction

* **Multiclass**: returns the class with the highest probability (`argmax`).
* **Multilabel**: returns 1 if output ≥ 0.5, else 0.

### 5️⃣ Notes

* The combination of **softmax + cross-entropy** simplifies the gradient to `output - target`, so no need for explicit softmax derivative.
* Vectorized implementation allows efficient batch processing.
* Hidden layers, activations, learning rate, and batch size are fully configurable.

## Usage 🛠️

You can test and use the network via `main.py`:

* **Binary classification**: set `type='multiclass'` with 1 output neurons. Uncomment the `make_classification` line to generate data.
* **Multiclass classification**: set `type='multiclass'` with `output_neurone` equal to the number of classes. Uncomment the `make_classification` line.
* **Multilabel classification**: set `type='multilabel'` with `output_neurone` equal to the number of labels. Uncomment the `make_multilabel_classification` line.

```python
from nueral_network_for_classification import nueral_network
from sklearn.datasets import make_classification, make_multilabel_classification

if __name__ == "__main__":
    # Multi-label example
    x, y = make_multilabel_classification(n_features=10, n_labels=2, n_classes=2, n_samples=500, random_state=42)

    nn = nueral_network(neurone_in_each_hidden_layer=[32,16,8], output_neurone=2, lr=0.01, activation_function='sigmoid', type='multilabel')
    nn.fit(x, y, epochs=100, batch_size=64)
    predictions = nn.predict(x)

    print(f"The Prediction is : {predictions[:20]}")
    print(f"The Truth is      : {y[:20]}")
```

## Author ✍️

**Fathy Abderabbo**

## If you have any Note about the code Contact me . 
