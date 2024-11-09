import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from data.processed.data_processing import get_loaders


# Define the neural network
class CatBreedClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.5):
        super(CatBreedClassifier, self).__init__()

        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

        # Define activation function and dropout
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Softmax for output layer
        self.dropout = nn.Dropout(p=dropout_prob)  # Dropout with probability p

    def forward(self, x):
        x = self.relu(self.fc1(x))  # First hidden layer with ReLU
        x = self.dropout(x)         # Apply dropout
        x = self.relu(self.fc2(x))  # Second hidden layer with ReLU
        x = self.dropout(x)         # Apply dropout
        x = self.relu(self.fc3(x))  # Third hidden layer with ReLU
        x = self.dropout(x)         # Apply dropout
        x = self.fc4(x)             # Output layer
        return self.softmax(x)

# Define training and evaluation function
def train_and_evaluate(model, criterion, optimizer, train_loader, val_loader, epochs=20):
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                outputs = model(X_val_batch)
                _, predicted = torch.max(outputs, 1)
                total += y_val_batch.size(0)
                correct += (predicted == y_val_batch).sum().item()
        accuracy = 100 * correct / total

        print(
            f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Validation Accuracy: {accuracy:.2f}%")

# Load data and prepare DataLoader
def prepare_data(file_path, batch_size=32):
    # Get loaders
    X_train, y_train, X_val, y_val, _, _ = get_loaders(file_path)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


if __name__ == '__main__':
    # Hyperparameters
    input_size = 25  # Based on your data
    hidden_size = 512
    output_size = 14  # Number of breeds (target classes)
    dropout_prob = 0.5  # Dropout probability

    # Initialize model, loss function, and optimizer
    model = CatBreedClassifier(input_size, hidden_size, output_size, dropout_prob)
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

    file_path = "balanced_train_data.xlsx"

    # Prepare data loaders and train the model
    train_loader, val_loader = prepare_data(file_path)
    train_and_evaluate(model, criterion, optimizer, train_loader, val_loader, epochs=50)
