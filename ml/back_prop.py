# From-scratch backprop tutorial in Python (2-2-1 MLP, sigmoid activations, BCE loss)
# - Single forward/backward demo with explicit numbers
# - Numerical gradient check
# - Tiny training loop on XOR to show learning
#
# No external deps besides numpy.

import numpy as np
from typing import Dict, Tuple, List

np.set_printoptions(precision=6, suppress=True)


# --- 1. Core Functions (Activation, Loss, and their Gradients) ---


def sigmoid(z: np.ndarray) -> np.ndarray:
    """The sigmoid activation function."""
    return 1 / (1 + np.exp(-z))


def sigmoid_grad(a: np.ndarray) -> np.ndarray:
    """Gradient of the sigmoid function, where 'a' is the activated output sigmoid(z)."""
    return a * (1 - a)


def bce_loss(y_hat: np.ndarray, y: np.ndarray) -> float:
    """Binary Cross-Entropy loss for a single example."""
    eps = 1e-12
    y_hat = np.clip(y_hat, eps, 1 - eps)
    # Use .item() to extract the scalar value from the resulting 1-element array,
    # which avoids the NumPy deprecation warning.
    return (-(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))).item()


# --- 2. Model Definition ---


class TinyMLP221:
    """
    A simple 2-2-1 Multi-Layer Perceptron with sigmoid activations.
    - Input layer: 2 neurons
    - Hidden layer: 2 neurons
    - Output layer: 1 neuron
    """

    def __init__(
        self,
        W1: np.ndarray = None,
        b1: np.ndarray = None,
        W2: np.ndarray = None,
        b2: np.ndarray = None,
        seed: int = 0,
    ):
        """Initializes weights and biases, either with provided values or random ones."""
        rng = np.random.default_rng(seed)
        # Layer 1 weights and biases (input -> hidden)
        self.W1 = (
            W1 if W1 is not None else rng.normal(0, 0.5, size=(2, 2))
        )  # Shape: (input_size, hidden_size)
        self.b1 = b1 if b1 is not None else np.zeros(2)  # Shape: (hidden_size,)
        # Layer 2 weights and biases (hidden -> output)
        self.W2 = (
            W2 if W2 is not None else rng.normal(0, 0.5, size=(2, 1))
        )  # Shape: (hidden_size, output_size)
        self.b2 = b2 if b2 is not None else np.zeros(1)  # Shape: (output_size,)

    def forward(self, x: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Performs a forward pass, returning intermediate values for backprop."""
        z1 = self.W1.T @ x + self.b1  # (2,)
        h = sigmoid(z1)  # (2,)
        z2 = self.W2.T @ h + self.b2  # (1,)
        y_hat = sigmoid(z2)  # (1,)
        cache = {"x": x, "z1": z1, "h": h, "z2": z2, "y_hat": y_hat}
        return cache, y_hat

    def backward(self, cache: Dict[str, np.ndarray], y: float) -> Dict[str, np.ndarray]:
        """Performs a backward pass to compute gradients using the chain rule."""
        x, z1, h, z2, y_hat = (
            cache["x"],
            cache["z1"],
            cache["h"],
            cache["z2"],
            cache["y_hat"],
        )
        # Ensure shapes
        y = np.array([y]).reshape(
            1,
        )

        dz2 = y_hat - y  # (1,)
        # Gradients for W2, b2
        dW2 = h.reshape(2, 1) @ dz2.reshape(1, 1)  # (2,1)
        db2 = dz2.copy()  # (1,)

        # Backprop to hidden
        dh = (self.W2 @ dz2).reshape(
            2,
        )  # (2,)
        dz1 = dh * sigmoid_grad(h)  # (2,)

        # Gradients for W1, b1
        dW1 = x.reshape(2, 1) @ dz1.reshape(
            1, 2
        )  # (2,2) but note broadcasting — ensure transpose later
        # We used W1.T @ x in forward, so be careful with shapes:
        # z1 = W1.T @ x + b1  => z1_i = sum_j W1[j,i] * x_j
        # dW1[j,i] = x_j * dz1_i  -> Our computed dW1 currently has shape (2,2) with dW1[j,i]. Good.
        db1 = dz1.copy()  # (2,)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return grads

    def apply_gradients(self, grads: Dict[str, np.ndarray], lr: float = 0.1):
        """Updates model parameters using the computed gradients."""
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]

    def calculate_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """Convenience function to calculate loss for a given input-output pair."""
        # Ensure y is passed as a NumPy array to be consistent with bce_loss type hints.
        _, y_hat = self.forward(x)
        return bce_loss(y_hat, y)


# --- 3. Main Execution and Demonstrations ---


def demo_single_pass():
    """Demonstrates a single forward and backward pass with fixed weights."""
    print("=" * 20, "1. Single Forward/Backward Pass", "=" * 20)
    # Toy constants from a tutorial text
    x = np.array([1.0, 0.0])
    y = 1.0
    W1 = np.array([[0.1, 0.2], [0.3, 0.4]])
    b1 = np.array([0.0, 0.0])
    W2 = np.array([[0.5], [-0.5]])
    b2 = np.array([0.0])

    net = TinyMLP221(W1=W1, b1=b1, W2=W2, b2=b2)

    # Forward pass
    cache, y_hat = net.forward(x)
    loss = net.calculate_loss(x, np.array([y]))

    print("\n--- Forward pass ---")
    print(f"x:     {x}")
    print(f"z1:    {cache['z1']}")
    print(f"h:     {cache['h']}")
    print(f"z2:    {cache['z2']}")
    print(f"y_hat: {y_hat}")
    print(f"Loss:  {loss:.6f}")

    # Backward pass
    grads = net.backward(cache, y)

    print("\n--- Backward pass (Gradients) ---")
    for name, grad in grads.items():
        print(f"d{name}:\n{grad}")
    return net, x, y


def demo_gradient_check(net: TinyMLP221, x: np.ndarray, y: float):
    """Performs a numerical gradient check to verify the backward pass."""
    print("\n" + "=" * 20, "2. Numerical Gradient Check", "=" * 20)

    def pack_params(model: TinyMLP221) -> np.ndarray:
        return np.concatenate(
            [p.flatten() for p in [model.W1, model.b1, model.W2, model.b2]]
        )

    def unpack_params(model: TinyMLP221, vec: np.ndarray):
        i = 0
        s_W1, s_b1, s_W2, s_b2 = (
            model.W1.size,
            model.b1.size,
            model.W2.size,
            model.b2.size,
        )
        model.W1 = vec[i : i + s_W1].reshape(model.W1.shape)
        i += s_W1
        model.b1 = vec[i : i + s_b1].reshape(model.b1.shape)
        i += s_b1
        model.W2 = vec[i : i + s_W2].reshape(model.W2.shape)
        i += s_W2
        model.b2 = vec[i : i + s_b2].reshape(model.b2.shape)

    # Calculate analytic gradient
    cache, _ = net.forward(x)
    analytic_grads = net.backward(cache, y)
    analytic_vec = pack_params(analytic_grads)

    # Calculate numerical gradient
    eps = 1e-5
    param_vec = pack_params(net)
    numeric_vec = np.zeros_like(param_vec)
    for i in range(param_vec.size):
        param_vec[i] += eps
        unpack_params(net, param_vec)
        loss_plus = net.calculate_loss(x, np.array([y]))

        param_vec[i] -= 2 * eps
        unpack_params(net, param_vec)
        loss_minus = net.calculate_loss(x, np.array([y]))

        numeric_vec[i] = (loss_plus - loss_minus) / (2 * eps)
        param_vec[i] += eps  # Restore original value

    unpack_params(net, param_vec)  # Restore original model state

    max_abs_diff = np.max(np.abs(analytic_vec - numeric_vec))
    print(
        f"Max absolute difference between analytic and numeric gradients: {max_abs_diff:.2e}"
    )
    assert max_abs_diff < 1e-6, "Gradient check failed!"
    print("Gradient check passed!")


def demo_xor_training():
    """Trains the MLP on the XOR problem to demonstrate learning."""
    print("\n" + "=" * 20, "3. Train on XOR", "=" * 20)
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    Y = np.array([0.0, 1.0, 1.0, 0.0])

    # Re-initialize a fresh model for training
    xor_net = TinyMLP221(seed=42)

    # --- Training Loop ---
    lr = 0.5
    epochs = 5000
    for ep in range(1, epochs + 1):
        epoch_loss = 0.0
        for x_i, y_i in zip(X, Y):
            cache, y_hat = xor_net.forward(x_i)
            epoch_loss += bce_loss(y_hat, np.array([y_i]))
            grads = xor_net.backward(cache, y_i)
            xor_net.apply_gradients(grads, lr=lr)

        if ep % 1000 == 0 or ep == 1:
            print(f"[Epoch {ep:4d}] Average Loss: {epoch_loss / len(X):.6f}")

    # --- Evaluation ---
    print("\n--- Predictions after training ---")
    for x_i, y_i in zip(X, Y):
        _, y_hat = xor_net.forward(x_i)
        prediction = 1 if y_hat[0] >= 0.5 else 0
        print(
            f"Input: {x_i}, Target: {int(y_i)}, Prediction: {prediction} (y_hat={y_hat[0]:.4f})"
        )


def main():
    """Runs all high-level steps of the tutorial."""
    # 1. Demonstrate a single forward/backward pass with fixed weights
    net, x, y = demo_single_pass()

    # 2. Verify the backward pass implementation with a numerical gradient check
    demo_gradient_check(net, x, y)

    # 3. Train a new network on the XOR problem to show that it can learn
    demo_xor_training()


if __name__ == "__main__":
    main()
