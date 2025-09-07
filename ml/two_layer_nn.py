import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. Define the Neural Network Architecture
class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        """
        Initializes the layers of the neural network.
        - input_size: The number of input features.
        - hidden_size1: The number of neurons in the first hidden layer.
        - hidden_size2: The number of neurons in the second hidden layer.
        - num_classes: The number of output classes.
        """
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the neural network.
        The final layer's output is raw logits. The softmax is applied by the loss function.
        """
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        # Note: We don't apply softmax here because nn.CrossEntropyLoss
        # expects raw logits and applies log_softmax internally for better
        # numerical stability.
        return out


def main():
    # 2. Set up Hyperparameters and Data
    input_size = 8  # n-dimensional binary vector
    hidden_size1 = 32
    hidden_size2 = 16
    num_classes = 2  # Binary output: 0 or 1
    batch_size = 128
    learning_rate = 0.02
    num_epochs = 100

    # Generate training data for the high-dimensional XOR/Parity problem
    # Input: n-dimensional binary vector (0s and 1s)
    # Output: 1 if the sum of inputs is odd, 0 otherwise.
    X_train = torch.randint(0, 2, (batch_size, input_size)).float()
    y_train = (torch.sum(X_train, dim=1) % 2).long()

    print("--- Sample Data ---")
    print("Sample Input (X_train[0]):", X_train[0].numpy())
    print("Sample Label (y_train[0]):", y_train[0].item())
    print("Input features (X_train shape):", X_train.shape)
    print("Target labels (y_train shape):", y_train.shape)
    print("-" * 20)

    # 3. Instantiate the Model, Loss, and Optimizer
    model = TwoLayerNet(input_size, hidden_size1, hidden_size2, num_classes)
    print("\n--- Model Architecture ---")
    print(model)
    print("-" * 20)

    # Loss function: CrossEntropyLoss is suitable for multi-class classification.
    # It combines nn.LogSoftmax and nn.NLLLoss in one single class.
    criterion = nn.CrossEntropyLoss()

    # Optimizer: Adam is often a good default choice for faster convergence.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Training Loop
    print("\n--- Training ---")
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward and optimize
        optimizer.zero_grad()  # Clear gradients from previous iteration
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # 5. Simple Evaluation after training
    print("\n--- Evaluation ---")
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for inference
        # Test with a new data point
        test_x = torch.tensor(
            [[1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]]
        )  # Sum = 5 (odd)
        true_y = 1
        outputs = model(test_x)
        print(f"Raw outputs (logits): {outputs.data.numpy().flatten()}")
        # Get the prediction from the max value of the output logits
        _, predicted_y = torch.max(outputs.data, 1)

        print(f"Input: {test_x.numpy().flatten()}")
        print(f"True y: {true_y}")
        print(f"Predicted y: {predicted_y.item()}")
        print(f"Correct: {true_y == predicted_y.item()}")


if __name__ == "__main__":
    main()
