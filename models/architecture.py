import numpy as np


# ReLU activation and its derivative
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return z > 0

def softmax(z):
    # z must be a NumPy array (if it's a Pandas DataFrame/Series)
    z = np.array(z)
    exp_Z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stability improvement
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true])  # Assuming y_true is integer labels
    loss = np.sum(log_likelihood) / m
    return loss


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001, dropout_rate=0.5):
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate  # Fraction of neurons to drop during training

        # Random initialization in the range [-1, 1]
        # Layer 1 (input to hidden layer 1)
        self.W1 = np.random.uniform(-1, 1, (input_size, hidden_size))  # Random initialization in [-1, 1]
        self.b1 = np.zeros((1, hidden_size))  # Bias for hidden layer 1

        # Layer 2 (hidden layer 1 to hidden layer 2)
        self.W2 = np.random.uniform(-1, 1, (hidden_size, hidden_size))  # Random initialization in [-1, 1]
        self.b2 = np.zeros((1, hidden_size))  # Bias for hidden layer 2

        # Layer 3 (hidden layer 2 to output)
        self.W3 = np.random.uniform(-1, 1, (hidden_size, output_size))  # Random initialization in [-1, 1]
        self.b3 = np.zeros((1, output_size))  # Bias for output layer

        # Initialization of activations and pre-activations
        self.Z1 = None
        self.A1 = None
        self.Z2 = None
        self.A2 = None
        self.Z3 = None
        self.A3 = None

    def save_weights(self, filename):
        np.savez(filename, W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3)
        print("Weights saved to", filename)

    def load_weights(self, filename):
        # Load weights and biases from a file
        data = np.load(filename)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.W3 = data['W3']
        self.b3 = data['b3']
        print("Weights loaded from", filename)

    def forward(self, X, training=True):
        # Layer 1 (input to first hidden layer)
        self.Z1 = X.dot(self.W1) + self.b1
        self.A1 = relu(self.Z1)  # ReLU activation for hidden layer 1

        # Apply dropout after first hidden layer (only during training)
        if training:
            self.A1 = self.dropout(self.A1)

        # Layer 2 (first hidden layer to second hidden layer)
        self.Z2 = self.A1.dot(self.W2) + self.b2
        self.A2 = relu(self.Z2)  # ReLU activation for hidden layer 2

        # Apply dropout after second hidden layer (only during training)
        if training:
            self.A2 = self.dropout(self.A2)

        # Output layer (second hidden layer to output)
        self.Z3 = self.A2.dot(self.W3) + self.b3
        self.A3 = softmax(self.Z3)  # Softmax activation for output layer

        return self.A3

    def dropout(self, A):
        # Dropout implementation
        mask = np.random.rand(*A.shape) < (1 - self.dropout_rate)  # Mask with 0's and 1's
        return A * mask  # Element-wise multiplication with mask

    def backward(self, X, y, lambda_reg=0.01):
        m = y.shape[0]

        # Output layer gradient
        one_hot_y = np.zeros_like(self.A3)
        one_hot_y[np.arange(m), y] = 1
        dZ3 = self.A3 - one_hot_y
        dW3 = (1 / m) * self.A2.T.dot(dZ3) + lambda_reg * self.W3  # Add weight decay to gradient
        db3 = (1 / m) * np.sum(dZ3, axis=0, keepdims=True)

        # Hidden layer 2 gradient
        dA2 = dZ3.dot(self.W3.T)
        dZ2 = dA2 * relu_derivative(self.Z2)
        dW2 = (1 / m) * self.A1.T.dot(dZ2) + lambda_reg * self.W2  # Add weight decay to gradient
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        # Hidden layer 1 gradient
        dA1 = dZ2.dot(self.W2.T)
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = (1 / m) * X.T.dot(dZ1) + lambda_reg * self.W1  # Add weight decay to gradient
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        # Update weights and biases with regularization
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    # Training function with batch training
    def train(self, X, y, epochs, batch_size, lambda_reg=0.01):
        for epoch in range(epochs):
            # Shuffle the data at the start of each epoch
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            # Process each batch
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                # Forward pass
                self.forward(X_batch)
                # Backward pass
                self.backward(X_batch, y_batch, lambda_reg=lambda_reg)

            # Calculate loss for the full dataset (optional, for monitoring)
            y_pred = self.forward(X)
            loss = cross_entropy_loss(y, y_pred)  # Pass only y and y_pred
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)