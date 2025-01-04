import os
import pandas as pd
import torch
from dotenv import load_dotenv
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from data.processed.data_processing import load_data

load_dotenv()
train_data = os.getenv('BALANCED_DATASET')


class CustomDataset(Dataset):
    def __init__(self, X, y):
        # ensure X and y are NumPy arrays before converting to PyTorch tensors
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.to_numpy()

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def generate_positional_encoding(seq_len, d_model):
    """
    Generates positional encodings for a given sequence length and model dimension.
    """
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # Shape (seq_len, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))  # Shape (d_model/2,)
    positional_encoding = torch.zeros(seq_len, d_model)
    positional_encoding[:, 0::2] = torch.sin(position * div_term)  # Even indices: sin
    positional_encoding[:, 1::2] = torch.cos(position * div_term)  # Odd indices: cos
    return positional_encoding


class TransformerModel(nn.Module):
    def __init__(self, input_size=25, d_model=128, nhead=8, num_encoder_layers=3, num_classes=14):
        super(TransformerModel, self).__init__()

        # Ensure that d_model is divisible by nhead
        assert d_model % nhead == 0, f"embed_dim ({d_model}) must be divisible by num_heads ({nhead})"

        # Projection layer to match input size with d_model
        self.input_projection = nn.Linear(input_size, d_model)  # From 25 to d_model

        # Transformer Encoder with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Final classification layer to output num_classes
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # ensure the input tensor has the shape [batch_size, seq_len, d_model]
        batch_size = x.size(0)
        seq_len = 1  # sequence length is 1 for each instance in the batch

        # if your input has shape [batch_size, d_model], reshape it to [batch_size, seq_len, d_model]
        x = x.unsqueeze(1)  # shape: [batch_size, 1, d_model]

        # project input features to match d_model (e.g., 128)
        x = self.input_projection(x)  # shape: [batch_size, seq_len, d_model]

        # generate dynamic positional encoding based on seq_len
        positional_encoding = generate_positional_encoding(seq_len, x.size(-1)).unsqueeze(0).to(
            x.device)  # shape: (1, seq_len, d_model)

        # ensure the positional encoding matches the batch size
        positional_encoding = positional_encoding.repeat(batch_size, 1, 1)  # shape: (batch_size, seq_len, d_model)

        x += positional_encoding  # add positional encoding to input

        # pass through Transformer Encoder
        x = self.encoder(x)  # Shape: (batch_size, seq_len, d_model)

        # take the output of the last time step (or average across all time steps)
        x = x.mean(dim=1)  # global average pooling, shape: (batch_size, d_model)

        # Classifier
        x = self.classifier(x)  # shape: (batch_size, num_classes)

        return x


def train_model(model, train_loader, val_loader, epochs, learning_rate, weights_file):
    # set up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # move the model to the appropriate device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0

        # loop over batches from the train loader
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # zero the gradients
            optimizer.zero_grad()

            # perform forward pass
            outputs = model(X_batch)

            loss = criterion(outputs, y_batch.long())

            # backward pass and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        val_loss = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # save the trained model's weights to the specified file
    torch.save(model.state_dict(), weights_file)


def evaluate_model(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    # disable gradient calculation (for faster computation during evaluation)
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass to get predictions
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.long())

            val_loss += loss.item()

            # Get predicted class by finding the class with maximum probability
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    # compute average validation loss and accuracy
    val_loss /= len(val_loader)
    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')

    return val_loss

def load_model(weights_file, input_size, output_size, d_model, n_head):
    model = TransformerModel(input_size, output_size, d_model, n_head)
    model.load_state_dict(torch.load(weights_file))
    return model


if __name__ == '__main__':

    X_in, y_out, race_desc = load_data(train_data)
    X_train, X_val, y_train, y_val = train_test_split(X_in, y_out, test_size=0.2)

    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=25, shuffle=False)

    input_size = 25  # number of features
    output_size = 14  # number of classes
    d_model = 128  # embedding size
    attention_heads = 4  # number of attention heads

    epochs = 50
    learning_rate = 0.001
    weights_file = "transformer_weights.pth"

    model = TransformerModel(input_size=input_size, d_model=d_model, nhead=attention_heads, num_classes=output_size)
    # print(model)

    train_model(model, train_loader, val_loader, epochs, learning_rate, weights_file)
